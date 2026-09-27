# OmniZip baseline entry point for Qwen2.5-Omni.

from dataclasses import dataclass

from torch import nn

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from .modeling_qwen2_5_omni_omnizip import (
    Qwen2_5OmniThinkerForConditionalGeneration_forward_omnizip,
)


@dataclass
class OmniZipConfig:
    rho_audio: float = 0.3
    rho_video: float = 0.6
    contextual_ratio: float = 0.05
    g: int = 3


def omnizip(
    model: nn.Module,
    video_ratio: float = 0.4,
    audio_ratio: float = 0.7,
    contextual_ratio: float = 0.05,
    g: int = 3,
) -> nn.Module:
    if type(model) is not Qwen2_5OmniForConditionalGeneration:
        raise NotImplementedError(f"OmniZip baseline supports Qwen2.5-Omni only, got {type(model)}.")

    if not hasattr(model.thinker, "_omnizip_original_forward"):
        model.thinker._omnizip_original_forward = model.thinker.forward
        Qwen2_5OmniThinkerForConditionalGeneration.forward = (
            Qwen2_5OmniThinkerForConditionalGeneration_forward_omnizip
        )
    model.thinker.omnizip_config = {
        "rho_audio": 1.0 - float(audio_ratio),
        "rho_video": 1.0 - float(video_ratio),
        "contextual_ratio": float(contextual_ratio),
        "g": int(g),
    }
    return model

