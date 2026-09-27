from dataclasses import dataclass, field
from typing import Callable

from torch import nn

from models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from .fastvid_units import compress_fastvid
from .modeling_qwen2_5_omni_fastvid import (
    Qwen2_5OmniThinkerForConditionalGeneration_forward_fastvid,
)


@dataclass
class FastVIDConfig:
    method: str = "fastvid"
    video_ratio: float = 1.0
    audio_ratio: float = 1.0
    dyseg_c: int = 8
    dyseg_tau: float = 0.84
    dtm_p: int = 4
    compressor: Callable = field(default=compress_fastvid, repr=False)


def fastvid(model: nn.Module, video_ratio=0.30, audio_ratio=0.65, dyseg_c=8, dyseg_tau=0.84, dtm_p=4):
    if type(model) is not Qwen2_5OmniForConditionalGeneration:
        raise NotImplementedError(f"FastVID is not supported for {type(model)} yet.")
    Qwen2_5OmniThinkerForConditionalGeneration.forward = Qwen2_5OmniThinkerForConditionalGeneration_forward_fastvid
    model.thinker.video_compressor_config = FastVIDConfig(
        video_ratio=float(video_ratio), audio_ratio=float(audio_ratio),
        dyseg_c=int(dyseg_c), dyseg_tau=float(dyseg_tau), dtm_p=int(dtm_p),
    )
    return model
