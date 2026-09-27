"""VidCom2 Gaussian outlier selection for Qwen2.5-Omni video grids."""

import torch

from .vidcom2 import (
    _map_linear_offset,
    compute_gaussian_scores,
    compute_scales,
    select_low_var_channels,
    select_outlier_indices,
)


def compress_vidcom2(flat_features, grid_thw, spatial_merge_size, retention_ratio, config):
    t, h, w = [int(x) for x in grid_thw.tolist()]
    frame_tokens = (h * w) // (spatial_merge_size ** 2)
    total = int(flat_features.shape[0])
    if t <= 0 or frame_tokens <= 0 or total != t * frame_tokens:
        indices = torch.arange(total, device=flat_features.device, dtype=torch.long)
        return flat_features, indices
    selected = select_low_var_channels(flat_features, ratio=float(config.low_var_ratio))
    video_score, frame_score = compute_gaussian_scores(selected, frame_tokens)
    scales = compute_scales(
        -video_score.mean(dim=-1),
        float(max(0.0, min(1.0, retention_ratio))),
        temp=float(config.temperature),
    )
    local_indices = select_outlier_indices(video_score + frame_score, scales, frame_tokens)
    keep = _map_linear_offset(local_indices, frame_tokens).to(flat_features.device, torch.long)
    keep = torch.unique(keep, sorted=True)
    if keep.numel() == 0:
        keep = torch.zeros(1, device=flat_features.device, dtype=torch.long)
    return flat_features[keep], keep
