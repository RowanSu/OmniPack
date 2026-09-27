"""Qwen2.5-Omni forward replacement for the FastVID video-only baseline."""

from typing import Callable, List, Optional, Tuple, Union

import torch

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from baselines.utils import audio_intact_reallocate


VideoCompressor = Callable[
    [torch.Tensor, torch.Tensor, int, float, object],
    Tuple[torch.Tensor, torch.Tensor],
]


def compress_video_sequence(
    *,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    video_embeds: torch.Tensor,
    video_grid_thw: torch.Tensor,
    video_token_id: int,
    spatial_merge_size: int,
    retention_ratio: float,
    compressor: VideoCompressor,
    method_config: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress each video independently and map representatives into the AV sequence.

    The returned global mask only removes video positions. Audio, image, and text
    positions therefore remain bit-for-bit aligned with the original prefill.
    """
    if inputs_embeds.shape[0] != 1:
        raise ValueError("video-only compressors currently require batch_size=1")
    if isinstance(video_embeds, (list, tuple)):
        video_embeds = torch.cat(video_embeds, dim=0)

    video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    split_sizes = (
        video_grid_thw.prod(dim=-1) // int(spatial_merge_size) ** 2
    ).tolist()
    if int(sum(split_sizes)) != int(video_embeds.shape[0]):
        raise ValueError(
            "video_grid_thw does not match encoded video tokens: "
            f"grid={sum(split_sizes)}, embeds={video_embeds.shape[0]}"
        )
    if video_positions.numel() != video_embeds.shape[0]:
        raise ValueError(
            "video placeholders do not match encoded video tokens: "
            f"placeholders={video_positions.numel()}, embeds={video_embeds.shape[0]}"
        )

    compressed_chunks = []
    representative_global = []
    offset = 0
    for grid, features in zip(video_grid_thw, torch.split(video_embeds, split_sizes)):
        compressed, representative = compressor(
            features,
            grid,
            int(spatial_merge_size),
            float(retention_ratio),
            method_config,
        )
        representative = representative.to(device=features.device, dtype=torch.long).flatten()
        compressed = compressed.to(device=features.device, dtype=features.dtype)
        if compressed.ndim != 2 or compressed.shape[0] != representative.numel():
            raise ValueError(
                "compressor must return [K,D] features and K representative indices, got "
                f"features={tuple(compressed.shape)}, indices={tuple(representative.shape)}"
            )
        if representative.numel() == 0:
            representative = torch.zeros(1, device=features.device, dtype=torch.long)
            compressed = features[:1]
        if representative.min() < 0 or representative.max() >= features.shape[0]:
            raise IndexError("compressor returned an out-of-range representative index")
        compressed_chunks.append(compressed)
        representative_global.append(representative + offset)
        offset += features.shape[0]

    compressed_video = torch.cat(compressed_chunks, dim=0)
    representative_global = torch.cat(representative_global, dim=0)
    kept_video_positions = video_positions[representative_global]

    global_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=inputs_embeds.device)
    global_mask[video_positions] = False
    global_mask[kept_video_positions] = True

    pruned_embeds = inputs_embeds[:, global_mask, :]
    pruned_ids = input_ids[:, global_mask]
    compressed_mask = (
        (pruned_ids == video_token_id)
        .unsqueeze(-1)
        .expand_as(pruned_embeds)
    )
    if int(compressed_mask[..., 0].sum().item()) != compressed_video.shape[0]:
        raise ValueError("representative video positions are not unique")
    pruned_embeds = pruned_embeds.masked_scatter(
        compressed_mask,
        compressed_video.to(pruned_embeds.device, pruned_embeds.dtype),
    )
    return pruned_embeds, global_mask


def Qwen2_5OmniThinkerForConditionalGeneration_forward_fastvid(
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
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    video_embeds = None
    if input_ids is not None and input_ids.shape[1] != 1:
        if input_features is not None:
            audio_features = self.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_mask = (input_ids == self.config.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_mask.to(inputs_embeds.device),
                audio_features.to(inputs_embeds.device, inputs_embeds.dtype),
            )

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.to(inputs_embeds.device),
                image_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
            )

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            flat_video_embeds = torch.cat(video_embeds, dim=0) if isinstance(video_embeds, (list, tuple)) else video_embeds
            video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.to(inputs_embeds.device),
                flat_video_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
            )

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
    else:
        audio_feature_lengths = None

    if attention_mask is not None and position_ids is None:
        if cache_position is None or cache_position[0] == 0 or self.rope_deltas is None:
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
            position_ids = position_ids.view(1, -1).expand(batch_size, -1).add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    cfg = getattr(self, "video_compressor_config", None)
    is_prefill = input_ids is not None and input_ids.shape[1] > 1
    if (
        is_prefill
        and cfg is not None
        and video_embeds is not None
        and video_grid_thw is not None
        and cfg.video_ratio < 1.0
    ):
        if input_ids.shape[0] != 1:
            raise ValueError(f"{cfg.method} only supports batch_size=1")
        video_token_num = int((input_ids[0] == self.config.video_token_id).sum().item())
        audio_token_num = int((input_ids[0] == self.config.audio_token_id).sum().item())
        video_ratio, audio_ratio = audio_intact_reallocate(
            cfg.video_ratio,
            cfg.audio_ratio,
            video_token_num,
            audio_token_num,
        )
        if audio_ratio < 1.0:
            raise ValueError(
                f"{cfg.method} is video-only, but the configured budget would require "
                f"audio_ratio={audio_ratio:.4f}; pass audio_ratio=1.0"
            )
        inputs_embeds, global_mask = compress_video_sequence(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            video_embeds=video_embeds,
            video_grid_thw=video_grid_thw,
            video_token_id=self.config.video_token_id,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
            retention_ratio=video_ratio,
            compressor=cfg.compressor,
            method_config=cfg,
        )
        self._video_compressor_last_ratio = video_ratio
        if attention_mask is not None:
            attention_mask = attention_mask[..., global_mask]
        if position_ids is not None:
            position_ids = position_ids[..., global_mask]

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
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size)
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
