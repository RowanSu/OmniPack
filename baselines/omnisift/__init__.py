# OmniSIFT baseline entry point for Qwen2.5-Omni.

from dataclasses import dataclass

from torch import nn

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from .modeling_qwen2_5_omni_omnisift import (
    Qwen2_5OmniThinkerForConditionalGeneration_forward_omnisift,
    TransformerAudioSelector,
)


@dataclass
class OmniSIFTConfig:
    rho_audio: float = 0.3
    rho_video: float = 0.7


def omnisift(
    model: nn.Module,
    video_ratio: float = 0.3,
    audio_ratio: float = 0.7,
) -> nn.Module:
    if type(model) is not Qwen2_5OmniForConditionalGeneration:
        raise NotImplementedError(f"OmniSIFT baseline supports Qwen2.5-Omni only, got {type(model)}.")

    if not hasattr(model.thinker, "_omnisift_original_forward"):
        model.thinker._omnisift_original_forward = model.thinker.forward
        Qwen2_5OmniThinkerForConditionalGeneration.forward = (
            Qwen2_5OmniThinkerForConditionalGeneration_forward_omnisift
        )
    if not hasattr(model.thinker, "compressor"):
        reference = next(model.thinker.parameters())
        model.thinker.compressor = TransformerAudioSelector(
            d_model=model.config.get_text_config().hidden_size
        ).to(device=reference.device, dtype=reference.dtype)
        model.thinker.compressor.eval()

    model.thinker.compression_config = {
        "rho_audio": 1.0 - float(audio_ratio),
        "rho_video": 1.0 - float(video_ratio),
    }
    return model
