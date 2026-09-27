"""FastVID dynamic-segmentation selection for Qwen2.5-Omni video grids."""

import torch

from .fastvid_algo import fastvid_compression


def compress_fastvid(flat_features, grid_thw, spatial_merge_size, retention_ratio, config):
    t, h, w = [int(x) for x in grid_thw.tolist()]
    frame_tokens = (h * w) // (spatial_merge_size ** 2)
    total = int(flat_features.shape[0])
    if t <= 0 or frame_tokens <= 0 or total != t * frame_tokens:
        indices = torch.arange(total, device=flat_features.device, dtype=torch.long)
        return flat_features, indices
    keep = fastvid_compression(
        video_embeds=flat_features,
        importance_scores=flat_features.float().norm(dim=-1),
        grid_thw=grid_thw.view(1, 3).to(flat_features.device),
        config_args={
            "retention_ratio": float(retention_ratio),
            "dyseg_c": int(config.dyseg_c),
            "dyseg_tau": float(config.dyseg_tau),
            "dtm_p": int(config.dtm_p),
        },
    )
    keep = torch.unique(keep.to(flat_features.device, torch.long), sorted=True)
    if keep.numel() == 0:
        keep = torch.zeros(1, device=flat_features.device, dtype=torch.long)
    return flat_features[keep], keep
