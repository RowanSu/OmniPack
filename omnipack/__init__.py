from dataclasses import dataclass
from typing import Optional

from torch import nn

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
    Qwen2_5OmniVisionFlashAttention2,
    Qwen2_5OmniVisionBlock,
    Qwen2_5OmniVisionEncoder,
    Qwen2_5OmniAudioFlashAttention2,
    Qwen2_5OmniAudioEncoderLayer,
    Qwen2_5OmniAudioEncoder,
)

from baselines.visionzip.modeling_qwen2_5_omni_visionzip import (
    Qwen2_5OmniVisionFlashAttention2_forward_visionzip,
    Qwen2_5OmniVisionBlock_forward_visionzip,
    Qwen2_5OmniVisionEncoder_forward_visionzip,
    Qwen2_5OmniAudioFlashAttention2_forward_visionzip,
    Qwen2_5OmniAudioEncoderLayer_forward_visionzip,
    Qwen2_5OmniAudioEncoder_forward_visionzip,
)
from .modeling_qwen2_5_omni_omnipack import (
    Qwen2_5OmniThinkerTextModel_forward_omnipack,
    Qwen2_5OmniThinkerForConditionalGeneration_forward_omnipack,
)


@dataclass
class OmniPackConfig:
    method: str = "omnipack"
    video_ratio: float = 0.30
    audio_ratio: float = 0.65
    video_encoder_ratio: float = 0.42
    audio_encoder_ratio: float = 0.91
    grid_in_window: int = 2
    sec_in_audio_window: int = 2
    stage2_layer: int = 22
    video_stage2_keep_ratio: float = 1.0
    audio_stage2_keep_ratio: float = 1.0


def omnipack(
    model: nn.Module,
    video_ratio: float = 0.30,
    audio_ratio: float = 0.65,
    video_encoder_ratio: float = 0.42,
    audio_encoder_ratio: float = 0.91,
    grid_in_window: int = 2,
    sec_in_audio_window: int = 2,
    stage2_layer: int = 22,
    video_stage2_keep_ratio: float = 1.0,
    audio_stage2_keep_ratio: float = 1.0,
) -> nn.Module:
    if type(model) is Qwen2_5OmniForConditionalGeneration:
        # Reuse VisionZip encoder hooks so both modalities expose _vz_attn_mean.
        Qwen2_5OmniVisionFlashAttention2.forward = Qwen2_5OmniVisionFlashAttention2_forward_visionzip
        Qwen2_5OmniVisionBlock.forward = Qwen2_5OmniVisionBlock_forward_visionzip
        Qwen2_5OmniVisionEncoder.forward = Qwen2_5OmniVisionEncoder_forward_visionzip
        Qwen2_5OmniAudioFlashAttention2.forward = Qwen2_5OmniAudioFlashAttention2_forward_visionzip
        Qwen2_5OmniAudioEncoderLayer.forward = Qwen2_5OmniAudioEncoderLayer_forward_visionzip
        Qwen2_5OmniAudioEncoder.forward = Qwen2_5OmniAudioEncoder_forward_visionzip
        Qwen2_5OmniThinkerTextModel.forward = Qwen2_5OmniThinkerTextModel_forward_omnipack
        Qwen2_5OmniThinkerForConditionalGeneration.forward = Qwen2_5OmniThinkerForConditionalGeneration_forward_omnipack
    else:
        raise NotImplementedError(f"OmniPack is not supported for {type(model)} yet.")

    cfg = OmniPackConfig(
        method="omnipack",
        video_ratio=video_ratio,
        audio_ratio=audio_ratio,
        video_encoder_ratio=video_encoder_ratio,
        audio_encoder_ratio=audio_encoder_ratio,
        grid_in_window=grid_in_window,
        sec_in_audio_window=sec_in_audio_window,
        stage2_layer=stage2_layer,
        video_stage2_keep_ratio=video_stage2_keep_ratio,
        audio_stage2_keep_ratio=audio_stage2_keep_ratio,
    )
    # Store the OmniPack configuration on the patched Thinker.
    setattr(model.thinker, "omnipack_config", cfg)
    setattr(model.thinker.visual, "grid_in_window", grid_in_window)
    setattr(model.thinker.audio_tower, "sec_in_audio_window", sec_in_audio_window)
    return model
