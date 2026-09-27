from dataclasses import dataclass, field
from typing import Callable

from torch import nn

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from .modeling_qwen2_5_omni_vidcom2 import (
    Qwen2_5OmniThinkerForConditionalGeneration_forward_vidcom2,
)
from .vidcom2_units import compress_vidcom2


@dataclass
class VidCom2Config:
    method: str = "vidcom2"
    video_ratio: float = 1.0
    audio_ratio: float = 1.0
    low_var_ratio: float = 0.5
    temperature: float = 0.01
    compressor: Callable = field(default=compress_vidcom2, repr=False)


def vidcom2(model: nn.Module, video_ratio=0.30, audio_ratio=0.65, low_var_ratio=0.5, temperature=0.01):
    if type(model) is not Qwen2_5OmniForConditionalGeneration:
        raise NotImplementedError(f"VidCom2 is not supported for {type(model)} yet.")
    Qwen2_5OmniThinkerForConditionalGeneration.forward = Qwen2_5OmniThinkerForConditionalGeneration_forward_vidcom2
    model.thinker.video_compressor_config = VidCom2Config(
        video_ratio=float(video_ratio), audio_ratio=float(audio_ratio),
        low_var_ratio=float(low_var_ratio), temperature=float(temperature),
    )
    return model
