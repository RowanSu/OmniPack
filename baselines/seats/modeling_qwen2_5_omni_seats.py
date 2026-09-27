import os
from typing import List, Optional, Tuple, Union
import torch

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
)
from .pre_llm_units import video_windivprune, audio_windivprune
from .inner_llm_units import top_down_token_selection, remove_non_textual_tokens


def _seats_debug_requested() -> bool:
    return os.getenv("SEATS_DEBUG_TOKEN_COUNTS", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _seats_debug_limit() -> int:
    try:
        return max(0, int(os.getenv("SEATS_DEBUG_MAX_SAMPLES", "1")))
    except ValueError:
        return 1


def _mask_count(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


def _seats_debug_context() -> str:
    return (
        f"rank={os.getenv('RANK', '?')} local_rank={os.getenv('LOCAL_RANK', '?')}"
        f" pid={os.getpid()}"
    )


def _print_token_counts(prefix: str, *, video: int, audio: int, text: int, total: int, extra: str = "") -> None:
    suffix = f" {extra}" if extra else ""
    print(
        f"[SEATS-DEBUG] {_seats_debug_context()} {prefix}"
        f" total={total} video={video} audio={audio} text={text}{suffix}",
        flush=True,
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / max(int(denominator), 1)


def _planned_keep_count(token_count: int, keep_ratio: float) -> int:
    if token_count <= 0:
        return 0
    return max(1, round(float(keep_ratio) * token_count))


def Qwen2_5OmniThinkerTextModel_forward_seats(
    self: Qwen2_5OmniThinkerTextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    seats_cfg = getattr(self, "seats_global_config", None)
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states, causal_mask, position_ids, past_key_values,
                output_attentions, use_cache, cache_position, position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        # ===== SEATS inner-LLM hooks (prefill only) =====
        if (seats_cfg is not None
            and (cache_position is not None and cache_position[0] == 0)  # prefill only
        ):
            seq_before = hidden_states.shape[1]
            keep_mask = None
            drop_kind = None
            video_before = _mask_count(self.seats_video_token_mask)
            audio_before = _mask_count(self.seats_audio_token_mask)
            text_before = _mask_count(self.seats_text_token_mask)

            # --- Middle-block: Top-down token budget allocation + Query-guided token selection ---
            if (layer_idx + 1) in seats_cfg.get("drop_layers", []):
                drop_kind = "progressive"
                if seats_cfg.get("debug_enabled", False):
                    seats_cfg.setdefault("debug_triggered_layers", []).append(layer_idx + 1)
                (hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values, keep_mask) = \
                    top_down_token_selection(
                    hidden_states=hidden_states,
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    layer_idx=layer_idx,
                    decoder_layers=self.layers,
                    progressive_config=seats_cfg,
                    audio_token_mask=self.seats_audio_token_mask,
                    text_token_mask=self.seats_text_token_mask,
                    video_token_mask=self.seats_video_token_mask,
                )

            # --- Late-block: remove all non-textual tokens (before late_block_layer) ---
            late_block_layer = seats_cfg.get("late_block_layer")
            if late_block_layer is not None and (layer_idx + 1) == late_block_layer - 1:
                drop_kind = "late-removal"
                (hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values, keep_mask) = \
                    remove_non_textual_tokens(
                    hidden_states=hidden_states,
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    layer_idx=layer_idx,
                    audio_token_mask=self.seats_audio_token_mask,
                    text_token_mask=self.seats_text_token_mask,
                    video_token_mask=self.seats_video_token_mask,
                )

            # 同步 token masks (任何一次 drop 都要更新)
            if keep_mask is not None and hidden_states.shape[1] < seq_before:
                self.seats_audio_token_mask = self.seats_audio_token_mask[keep_mask]
                self.seats_video_token_mask = self.seats_video_token_mask[keep_mask]
                self.seats_text_token_mask = self.seats_text_token_mask[keep_mask]

            if keep_mask is not None and seats_cfg.get("debug_enabled", False):
                video_after = _mask_count(self.seats_video_token_mask)
                audio_after = _mask_count(self.seats_audio_token_mask)
                text_after = _mask_count(self.seats_text_token_mask)
                drop_step = seats_cfg.get("drop_layers", []).index(layer_idx + 1) \
                    if (layer_idx + 1) in seats_cfg.get("drop_layers", []) else None
                expected = ""
                if drop_step is not None:
                    v_list = seats_cfg.get("video_keep_ratio_list", [])
                    a_list = seats_cfg.get("audio_keep_ratio_list", [])
                    if drop_step < len(v_list) and drop_step < len(a_list):
                        configured_v_keep = float(v_list[drop_step])
                        configured_a_keep = float(a_list[drop_step])
                        planned_video_after = _planned_keep_count(video_before, configured_v_keep)
                        planned_audio_after = _planned_keep_count(audio_before, configured_a_keep)
                        expected = (
                            f" configured_step_keep=(v={configured_v_keep:.4f},a={configured_a_keep:.4f})"
                            f" planned_after=(v={planned_video_after},a={planned_audio_after})"
                            f" count_error=(v={video_after - planned_video_after:+d},"
                            f"a={audio_after - planned_audio_after:+d})"
                        )
                raw_video = int(seats_cfg.get("raw_video_tokens", video_before))
                raw_audio = int(seats_cfg.get("raw_audio_tokens", audio_before))
                pre_video = int(seats_cfg.get("pre_llm_video_tokens", video_before))
                pre_audio = int(seats_cfg.get("pre_llm_audio_tokens", audio_before))
                _print_token_counts(
                    f"sample={seats_cfg.get('debug_sample', 0)} inner layer={layer_idx + 1}"
                    f" kind={drop_kind} before",
                    video=video_before,
                    audio=audio_before,
                    text=text_before,
                    total=seq_before,
                )
                _print_token_counts(
                    f"sample={seats_cfg.get('debug_sample', 0)} inner layer={layer_idx + 1}"
                    f" kind={drop_kind} after",
                    video=video_after,
                    audio=audio_after,
                    text=text_after,
                    total=hidden_states.shape[1],
                    extra=(
                        f"actual_step_keep=(v={_safe_ratio(video_after, video_before):.4f},"
                        f"a={_safe_ratio(audio_after, audio_before):.4f})"
                        f" cumulative_from_pre_llm=(v={_safe_ratio(video_after, pre_video):.4f},"
                        f"a={_safe_ratio(audio_after, pre_audio):.4f})"
                        f" cumulative_from_raw=(v={_safe_ratio(video_after, raw_video):.4f},"
                        f"a={_safe_ratio(audio_after, raw_audio):.4f}){expected}"
                    ),
                )

    if seats_cfg is not None and seats_cfg.get("debug_enabled", False):
        expected_layers = list(seats_cfg.get("drop_layers", []))
        triggered_layers = list(seats_cfg.get("debug_triggered_layers", []))
        final_video = _mask_count(self.seats_video_token_mask)
        final_audio = _mask_count(self.seats_audio_token_mask)
        raw_video = int(seats_cfg.get("raw_video_tokens", final_video))
        raw_audio = int(seats_cfg.get("raw_audio_tokens", final_audio))
        status = "OK" if triggered_layers == expected_layers else "MISMATCH"
        print(
            f"[SEATS-DEBUG] {_seats_debug_context()}"
            f" sample={seats_cfg.get('debug_sample', 0)} inner-summary"
            f" expected_layers={expected_layers} triggered_layers={triggered_layers}"
            f" trigger_count={len(triggered_layers)} status={status}"
            f" final_video={final_video} final_audio={final_audio}"
            f" final_from_raw=(v={_safe_ratio(final_video, raw_video):.4f},"
            f"a={_safe_ratio(final_audio, raw_audio):.4f})",
            flush=True,
        )

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )



def Qwen2_5OmniThinkerForConditionalGeneration_forward_seats(
    self: Qwen2_5OmniThinkerForConditionalGeneration,
    input_ids: Optional[torch.LongTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
    audio_feature_lengths: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    use_audio_in_video: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    video_second_per_grid: Optional[torch.LongTensor] = None,
) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        # 1. Extract the input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # 2. Merge text , audios , image and video
    if input_ids is not None and input_ids.shape[1] != 1:  # Prefill
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_mask_emb = (
                (input_ids == self.config.audio_token_id)
                .unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask_emb, audio_features)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_mask_emb = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask_emb, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_mask_emb = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask_emb, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
    else:
        audio_feature_lengths = None

    if attention_mask is not None and position_ids is None:
        if (
            cache_position is None
            or (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
        ):
            delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
                use_audio_in_video,
                audio_feature_lengths,
                video_second_per_grid,
            )
            rope_deltas = rope_deltas - delta0
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length = input_ids.shape
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # ===== SEATS Stage I: Pre-LLM winDivPrune =====
    is_prefill = input_ids is not None and input_ids.shape[1] > 1 and inputs_embeds is not None
    seats_config = getattr(self, "seats_config", None)
    debug_sample = int(getattr(self, "_seats_debug_prefill_count", 0))
    debug_enabled = is_prefill and _seats_debug_requested() and debug_sample < _seats_debug_limit()
    if is_prefill:
        self._seats_debug_prefill_count = debug_sample + 1
    if is_prefill and seats_config is not None and (
        seats_config.video_encoder_ratio < 1.0 or seats_config.audio_encoder_ratio < 1.0
    ):
        device = inputs_embeds.device
        spatial_merge_unit = self.config.vision_config.spatial_merge_size ** 2
        raw_video_tokens = int((input_ids[0] == self.config.video_token_id).sum().item())
        raw_audio_tokens = int((input_ids[0] == self.config.audio_token_id).sum().item())
        raw_total_tokens = int(input_ids.shape[1])
        raw_text_tokens = raw_total_tokens - raw_video_tokens - raw_audio_tokens

        if debug_enabled:
            _print_token_counts(
                f"sample={debug_sample} pre-llm before",
                video=raw_video_tokens,
                audio=raw_audio_tokens,
                text=raw_text_tokens,
                total=raw_total_tokens,
                extra=(
                    f"configured_encoder_keep=(v={seats_config.video_encoder_ratio:.4f},"
                    f"a={seats_config.audio_encoder_ratio:.4f})"
                ),
            )

        # Pre-LLM winDivPrune (video): attention-weighted DivPrune within window-per-modality
        video_attn_mean = getattr(self.visual, "_vz_attn_mean", None)
        assert video_attn_mean is not None, "video attn stats missing for SEATS pre-LLM"
        inputs_embeds, video_mask = video_windivprune(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            video_attn_mean=video_attn_mean,
            video_token_id=self.config.video_token_id,
            video_ratio=seats_config.video_encoder_ratio,
            spatial_merge_unit=spatial_merge_unit,
            video_grid_thw=video_grid_thw,
            grid_in_window=seats_config.grid_in_window,
        )

        # Pre-LLM winDivPrune (audio)
        audio_attn_mean = getattr(self.audio_tower, "_vz_attn_mean", None)
        assert audio_attn_mean is not None, "audio attn stats missing for SEATS pre-LLM"
        inputs_embeds, audio_mask = audio_windivprune(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            audio_attn_mean=audio_attn_mean,
            audio_token_id=self.config.audio_token_id,
            audio_ratio=seats_config.audio_encoder_ratio,
            sec_in_audio_window=seats_config.sec_in_audio_window,
        )

        global_mask = video_mask & audio_mask

        inputs_embeds = inputs_embeds[:, global_mask, :]
        if attention_mask is not None:
            attention_mask = attention_mask[..., global_mask]
        if position_ids is not None:
            position_ids = position_ids[..., global_mask]

        # ===== Stash token masks + inner-LLM config onto self.model for TextModel.forward =====
        flat = input_ids.reshape(-1)
        kept_ids = flat[global_mask]
        audio_token_mask = (kept_ids == self.config.audio_token_id)
        video_token_mask = (kept_ids == self.config.video_token_id)
        self.model.seats_audio_token_mask = audio_token_mask
        self.model.seats_video_token_mask = video_token_mask
        self.model.seats_text_token_mask = ~audio_token_mask & ~video_token_mask
        if debug_enabled:
            kept_video_tokens = _mask_count(video_token_mask)
            kept_audio_tokens = _mask_count(audio_token_mask)
            kept_text_tokens = _mask_count(self.model.seats_text_token_mask)
            _print_token_counts(
                f"sample={debug_sample} pre-llm after",
                video=kept_video_tokens,
                audio=kept_audio_tokens,
                text=kept_text_tokens,
                total=int(inputs_embeds.shape[1]),
                extra=(
                    f"actual_keep=(v={kept_video_tokens / max(raw_video_tokens, 1):.4f},"
                    f"a={kept_audio_tokens / max(raw_audio_tokens, 1):.4f})"
                ),
            )
            print(
                f"[SEATS-DEBUG] {_seats_debug_context()} sample={debug_sample} schedule"
                f" drop_layers={list(seats_config.drop_layers)}"
                f" video_step_keep={list(seats_config.video_inner_progressive_ratio_list or [])}"
                f" audio_step_keep={list(seats_config.audio_inner_progressive_ratio_list or [])}"
                f" late_block_layer={seats_config.late_block_layer}"
                f" inter_window_softmax_temp={float(seats_config.inter_window_softmax_temp):.4f}",
                flush=True,
            )
        self.model.seats_global_config = {
            "drop_layers": list(seats_config.drop_layers),
            "video_keep_ratio_list": list(seats_config.video_inner_progressive_ratio_list or []),
            "audio_keep_ratio_list": list(seats_config.audio_inner_progressive_ratio_list or []),
            "inter_window_softmax_temp": float(seats_config.inter_window_softmax_temp),
            "late_block_layer": int(seats_config.late_block_layer) if seats_config.late_block_layer else None,
            "debug_enabled": debug_enabled,
            "debug_sample": debug_sample,
            "raw_video_tokens": raw_video_tokens,
            "raw_audio_tokens": raw_audio_tokens,
            "pre_llm_video_tokens": _mask_count(video_token_mask),
            "pre_llm_audio_tokens": _mask_count(audio_token_mask),
            "debug_triggered_layers": [],
        }
    else:
        # decode step or no compression: ensure layer hook is inert
        self.model.seats_global_config = None

    outputs = self.model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
        )

    if not return_dict:
        output = (logits,) + outputs
        return (loss,) + output if loss is not None else output

    return Qwen2_5OmniThinkerCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
