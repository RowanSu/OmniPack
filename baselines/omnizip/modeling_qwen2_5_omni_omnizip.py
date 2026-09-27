"""Qwen2.5-Omni runtime hooks for the official OmniZip algorithm."""

from types import MethodType
from typing import Optional

import torch

from .omnizip_core import omnizip


def _collect_last_audio_attention(thinker):
    """Register the same last-layer audio-attention statistic used upstream."""

    def hook(module, args, kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_states is None:
            return
        with torch.no_grad():
            query_states = module.q_proj(hidden_states).reshape(hidden_states.shape[0], module.num_heads, -1)
            key_states = module.k_proj(hidden_states).reshape(hidden_states.shape[0], module.num_heads, -1)
            query_states = query_states.permute(1, 0, 2)
            key_states = key_states.permute(1, 0, 2)

            # Match the official OmniZip implementation: accumulate the
            # per-key attention importance in head/query chunks instead of
            # materializing the full [num_heads, seq_len, seq_len] matrix.
            # This is mathematically equivalent for token ranking while the
            # temporary attention tensor is bounded by [4, 512, seq_len].
            num_heads, seq_len, head_dim = query_states.shape
            scale = head_dim**-0.5
            token_importance = torch.zeros(
                seq_len,
                device=query_states.device,
                dtype=torch.float32,
            )
            head_chunk = 4
            query_chunk = 512
            key_states_t = key_states.transpose(-1, -2)
            for head_start in range(0, num_heads, head_chunk):
                query_head = query_states[head_start : head_start + head_chunk]
                key_head_t = key_states_t[head_start : head_start + head_chunk]
                for query_start in range(0, seq_len, query_chunk):
                    attn_chunk = torch.matmul(
                        query_head[:, query_start : query_start + query_chunk, :],
                        key_head_t,
                    ) * scale
                    attn_chunk = torch.softmax(attn_chunk, dim=-1)
                    token_importance.add_(attn_chunk.sum(dim=(0, 1), dtype=torch.float32))
                    del attn_chunk

            attn_mean = token_importance / (num_heads * seq_len)
            if attn_mean.shape[0] % 2 == 1:
                attn_mean = attn_mean[:-1]
            if attn_mean.shape[0] >= 2:
                attn_mean = attn_mean.view(-1, 2).mean(dim=-1)
            thinker._omnizip_audio_attn = attn_mean

    attention = thinker.audio_tower.layers[-1].self_attn
    return attention.register_forward_pre_hook(hook, with_kwargs=True)


def Qwen2_5OmniThinkerForConditionalGeneration_forward_omnizip(
    self,
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
    past_key_values=None,
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
):
    original_forward = self._omnizip_original_forward
    original_text_forward = self.model.forward
    self._omnizip_audio_attn = None
    hook_handle = None

    is_prefill = input_ids is not None and input_ids.shape[1] != 1
    if is_prefill and input_features is not None and pixel_values_videos is not None:
        hook_handle = _collect_last_audio_attention(self)

    def compressed_text_forward(model_self, *args, **kwargs):
        current_embeds = kwargs.get("inputs_embeds")
        current_positions = kwargs.get("position_ids")
        current_mask = kwargs.get("attention_mask")
        attn_logits = self._omnizip_audio_attn
        if (
            is_prefill
            and current_embeds is not None
            and current_positions is not None
            and attn_logits is not None
            and video_grid_thw is not None
        ):
            current_embeds, global_mask = omnizip(
                input_embeds=current_embeds,
                attn_logits=attn_logits,
                input_ids=input_ids,
                audio_token_id=self.config.audio_token_id,
                video_token_id=self.config.video_token_id,
                video_grid_thw=video_grid_thw,
                audio_tokens_per_sec=25,
                merging_ratio_audio=self.omnizip_config["rho_audio"],
                merging_ratio_v=self.omnizip_config["rho_video"],
                contextual_ratio=self.omnizip_config["contextual_ratio"],
                g=self.omnizip_config["g"],
            )
            kwargs["inputs_embeds"] = current_embeds[:, global_mask, :]
            if current_mask is not None:
                kwargs["attention_mask"] = current_mask[:, global_mask]
            kwargs["position_ids"] = current_positions[:, :, global_mask]
        return original_text_forward(*args, **kwargs)

    self.model.forward = MethodType(compressed_text_forward, self.model)
    try:
        return original_forward(
            input_ids=input_ids,
            input_features=input_features,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            rope_deltas=rope_deltas,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_audio_in_video=use_audio_in_video,
            cache_position=cache_position,
            video_second_per_grid=video_second_per_grid,
        )
    finally:
        self.model.forward = original_text_forward
        if hook_handle is not None:
            hook_handle.remove()
