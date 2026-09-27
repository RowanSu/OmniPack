"""Qwen2.5-Omni runtime hook for the official OmniSIFT algorithm."""

from types import MethodType
from typing import Optional

import torch
import torch.nn as nn

from .compression_units import similarity_pruning


class TransformerAudioSelector(nn.Module):
    """Official OmniSIFT video-guided audio selector."""

    def __init__(self, d_model=3584, nhead=8, num_layers=1, hidden_dim=512):
        super().__init__()
        self.v_proj = nn.Linear(d_model, hidden_dim)
        self.a_proj = nn.Linear(d_model, hidden_dim)
        self.v_norm = nn.LayerNorm(hidden_dim)
        self.a_norm = nn.LayerNorm(hidden_dim)
        self.post_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            batch_first=True,
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, v_embeds, a_embeds):
        v = self.v_norm(self.v_proj(v_embeds)).unsqueeze(0)
        a = self.a_norm(self.a_proj(a_embeds)).unsqueeze(0)
        attn_output, _ = self.cross_attn(query=a, key=v, value=v)
        x = self.post_attn_norm(attn_output + a)
        scores = self.score_head(x).squeeze(0).squeeze(-1)
        return torch.sigmoid(scores)


def Qwen2_5OmniThinkerForConditionalGeneration_forward_omnisift(
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
    original_forward = self._omnisift_original_forward
    original_text_forward = self.model.forward
    is_prefill = input_ids is not None and input_ids.shape[1] != 1

    def compressed_text_forward(model_self, *args, **kwargs):
        current_embeds = kwargs.get("inputs_embeds")
        current_positions = kwargs.get("position_ids")
        current_mask = kwargs.get("attention_mask")
        if (
            is_prefill
            and input_features is not None
            and pixel_values_videos is not None
            and current_embeds is not None
            and current_positions is not None
        ):
            current_embeds, global_mask, video_count, audio_count = similarity_pruning(
                self.compressor,
                current_embeds,
                input_ids,
                self.config.audio_token_id,
                self.config.video_token_id,
                current_positions,
                merging_ratio_a=self.compression_config["rho_audio"],
                merging_ratio_v=self.compression_config["rho_video"],
            )
            kwargs["inputs_embeds"] = torch.stack(
                [current_embeds[b, global_mask[b]] for b in range(current_embeds.size(0))],
                dim=0,
            )
            if current_mask is not None:
                kwargs["attention_mask"] = torch.stack(
                    [current_mask[b, global_mask[b]] for b in range(current_mask.size(0))],
                    dim=0,
                )
            kwargs["position_ids"] = torch.stack(
                [current_positions[:, b, global_mask[b]] for b in range(global_mask.size(0))],
                dim=1,
            )
            self.compressed_v_tokens_num = video_count
            self.compressed_a_tokens_num = audio_count
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