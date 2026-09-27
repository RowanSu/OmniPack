# OmniPack pre-LLM compression
from typing import Optional, Tuple

import torch


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() * b.float()).sum(dim=-1) / (
        a.float().norm(dim=-1) * b.float().norm(dim=-1) + 1e-6
    )


def _align_scores(scores: Optional[torch.Tensor], n: int, device: torch.device) -> Optional[torch.Tensor]:
    if scores is None:
        return None
    scores = scores.flatten().to(device=device, dtype=torch.float32)
    if scores.numel() < n:
        scores = torch.cat([scores, torch.zeros(n - scores.numel(), device=device)], dim=0)
    elif scores.numel() > n:
        scores = scores[:n]
    return scores


def _minmax(x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    if x.numel() == 0:
        return x
    if dim is None:
        return (x - x.min()) / (x.max() - x.min() + 1e-6)
    return (x - x.min(dim=dim, keepdim=True).values) / (
        x.max(dim=dim, keepdim=True).values - x.min(dim=dim, keepdim=True).values + 1e-6
    )


@torch.no_grad()
def _cluster_dpc_knn(
    features: torch.Tensor,
    cluster_num: int,
    k: int = 7,
    st_coords: Optional[torch.Tensor] = None,
    salience: Optional[torch.Tensor] = None,
    coord_weight: float = 0.20,
) -> torch.Tensor:
    total = int(features.shape[0])
    cluster_num = max(1, min(int(cluster_num), total))
    if cluster_num >= total:
        return torch.arange(total, device=features.device)

    feat = torch.nn.functional.normalize(features.float(), dim=-1)
    dist = 1.0 - torch.mm(feat, feat.transpose(0, 1))
    if st_coords is not None and coord_weight > 0:
        coord_dist = _minmax(torch.cdist(st_coords.float(), st_coords.float(), p=2), dim=-1)
        dist = dist + float(coord_weight) * coord_dist
    dist.fill_diagonal_(0.0)

    knn_k = max(1, min(int(k), total - 1))
    knn_dist = torch.topk(
        dist + torch.eye(total, device=features.device, dtype=dist.dtype) * 1e6,
        knn_k,
        dim=-1,
        largest=False,
    ).values
    density = torch.exp(-(knn_dist ** 2).mean(dim=-1))
    if salience is not None:
        density = density * (1.0 + 0.35 * _minmax(salience.float()))
    density = density + torch.arange(total, device=features.device, dtype=density.dtype) * 1e-12

    higher_density = density.unsqueeze(0) > density.unsqueeze(1)
    dist_to_higher = dist.masked_fill(~higher_density, float("inf"))
    delta = dist_to_higher.min(dim=1).values
    peak = torch.isinf(delta)
    if peak.any():
        delta[peak] = dist.max(dim=1).values[peak]

    score = density * delta
    centers = torch.topk(score, cluster_num, largest=True).indices
    return centers.sort().values


def _global_retention_pool(
    features: torch.Tensor,
    salience: torch.Tensor,
    ratio: float,
    st_coords: Optional[torch.Tensor] = None,
    dominant_fraction: float = 0.25,
    redundancy_tau: float = 0.70,
    knn_k: int = 7,
) -> torch.Tensor:
    total = int(features.shape[0])
    keep_total = max(1, min(total, round(float(ratio) * total)))
    if keep_total >= total:
        return torch.ones(total, dtype=torch.bool, device=features.device)

    salience = _minmax(salience.float()) + 1e-6
    order = torch.argsort(salience, descending=True)
    normed = torch.nn.functional.normalize(features.float(), dim=-1)
    keep = []

    dominant_target = max(1, min(keep_total, round(keep_total * float(dominant_fraction))))
    for idx in order.tolist():
        idx_t = torch.tensor(idx, device=features.device, dtype=torch.long)
        if not keep:
            keep.append(idx_t)
        else:
            kept_idx = torch.stack(keep)
            max_sim = torch.matmul(normed[idx_t], normed[kept_idx].transpose(0, 1)).max()
            if max_sim < redundancy_tau or len(keep) < dominant_target:
                keep.append(idx_t)
        if len(keep) >= dominant_target:
            break

    keep_mask = torch.zeros(total, dtype=torch.bool, device=features.device)
    if keep:
        keep_mask[torch.stack(keep)] = True
    contextual_num = keep_total - int(keep_mask.sum().item())
    remaining = (~keep_mask).nonzero(as_tuple=True)[0]
    if contextual_num > 0 and remaining.numel() > 0:
        centers_local = _cluster_dpc_knn(
            features[remaining],
            contextual_num,
            k=knn_k,
            st_coords=st_coords[remaining] if st_coords is not None else None,
            salience=salience[remaining],
        )
        keep_mask[remaining[centers_local]] = True
    return keep_mask


def _temporal_coords(total: int, device: torch.device) -> torch.Tensor:
    if total <= 1:
        return torch.zeros(total, 1, dtype=torch.float32, device=device)
    return (torch.arange(total, device=device, dtype=torch.float32) / float(total - 1)).unsqueeze(-1)


def _soft_merge_dropped_to_kept(
    features: torch.Tensor,
    keep_mask: torch.Tensor,
    salience: Optional[torch.Tensor] = None,
    st_coords: Optional[torch.Tensor] = None,
    coord_weight: float = 0.08,
    center_beta: float = 0.62,
) -> torch.Tensor:
    kept_idx = keep_mask.nonzero(as_tuple=True)[0]
    drop_idx = (~keep_mask).nonzero(as_tuple=True)[0]
    if kept_idx.numel() == 0 or drop_idx.numel() == 0:
        return features

    feat_norm = torch.nn.functional.normalize(features.float(), dim=-1)
    affinity = torch.matmul(feat_norm[drop_idx], feat_norm[kept_idx].transpose(0, 1))
    if st_coords is not None and coord_weight > 0:
        coord_dist = torch.cdist(st_coords[drop_idx].float(), st_coords[kept_idx].float(), p=2)
        affinity = affinity - float(coord_weight) * _minmax(coord_dist, dim=-1)
    if salience is not None:
        affinity = affinity + 0.10 * _minmax(salience[kept_idx].float()).unsqueeze(0)

    target_rel = torch.argmax(affinity, dim=-1)
    merged = features.float().clone()
    accum = merged[kept_idx].clone()
    counts = torch.ones(kept_idx.numel(), dtype=torch.float32, device=features.device)
    if salience is None:
        drop_weight = torch.ones(drop_idx.numel(), dtype=torch.float32, device=features.device)
    else:
        drop_weight = 0.5 + 0.5 * _minmax(salience[drop_idx].float())

    accum.index_add_(0, target_rel, features[drop_idx].float() * drop_weight.unsqueeze(-1))
    counts.index_add_(0, target_rel, drop_weight)
    cluster_mean = accum / counts.unsqueeze(-1)
    merged[kept_idx] = float(center_beta) * features[kept_idx].float() + (1.0 - float(center_beta)) * cluster_mean
    return merged.to(dtype=features.dtype)


def _video_event_bias(features: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
    total = int(features.shape[0])
    full_tokens = (total // tokens_per_frame) * tokens_per_frame
    bias = torch.zeros(total, dtype=torch.float32, device=features.device)
    if full_tokens <= tokens_per_frame:
        return bias

    frames = full_tokens // tokens_per_frame
    aligned = features[:full_tokens].reshape(frames, tokens_per_frame, -1)
    delta = torch.zeros(frames, tokens_per_frame, dtype=torch.float32, device=features.device)
    adjacent_change = 1.0 - _cosine(
        aligned[:-1].reshape(-1, aligned.shape[-1]),
        aligned[1:].reshape(-1, aligned.shape[-1]),
    ).reshape(frames - 1, tokens_per_frame)
    delta[:-1] = torch.maximum(delta[:-1], adjacent_change)
    delta[1:] = torch.maximum(delta[1:], adjacent_change)

    curvature = torch.zeros_like(delta)
    if frames >= 3:
        prev_step = aligned[1:-1] - aligned[:-2]
        next_step = aligned[2:] - aligned[1:-1]
        curvature[1:-1] = 1.0 - _cosine(
            prev_step.reshape(-1, prev_step.shape[-1]),
            next_step.reshape(-1, next_step.shape[-1]),
        ).reshape(frames - 2, tokens_per_frame)

    frame_mean = aligned.mean(dim=1, keepdim=True).expand_as(aligned)
    spatial_detail = 1.0 - _cosine(
        aligned.reshape(-1, aligned.shape[-1]),
        frame_mean.reshape(-1, frame_mean.shape[-1]),
    ).reshape(frames, tokens_per_frame)

    evidence = torch.maximum(delta, spatial_detail)
    bias[:full_tokens] = (_minmax(evidence) + 0.35 * _minmax(curvature)).reshape(-1)
    return bias


def _video_spatiotemporal_coords(
    total: int,
    grid_h: int,
    grid_w: int,
    spatial_merge_unit: int,
    device: torch.device,
) -> torch.Tensor:
    merge_size = max(1, int(round(float(spatial_merge_unit) ** 0.5)))
    merged_h = max(1, grid_h // merge_size)
    merged_w = max(1, grid_w // merge_size)
    tokens_per_frame = max(1, merged_h * merged_w)

    idx = torch.arange(total, device=device)
    t = idx // tokens_per_frame
    p = idx % tokens_per_frame
    h = p // merged_w
    w = p % merged_w

    t = t.float() / max(float(t.max().item()), 1.0)
    h = h.float() / max(float(merged_h - 1), 1.0)
    w = w.float() / max(float(merged_w - 1), 1.0)
    return torch.stack([t, h, w], dim=-1)


def _mix_attention_with_bias(
    attn_scores: Optional[torch.Tensor],
    bias: torch.Tensor,
    n: int,
    device: torch.device,
    bias_weight: float,
) -> torch.Tensor:
    attn = _align_scores(attn_scores, n, device)
    if attn is None:
        return 1.0 + bias_weight * _minmax(bias)
    attn_norm = _minmax(attn)
    bias_norm = _minmax(bias)
    return attn_norm * (1.0 + bias_weight * bias_norm) + 1e-6

def _ratio_adaptive_bias_weight(
    ratio: float,
    base_weight: float,
    min_scale: float = 0.35,
    low_ratio: float = 0.14,
    high_ratio: float = 0.42,
) -> float:
    scale = (float(ratio) - low_ratio) / max(high_ratio - low_ratio, 1e-6)
    scale = max(0.0, min(1.0, scale))
    return base_weight * (min_scale + (1.0 - min_scale) * scale)


def _audio_boundary_salience(
    features: torch.Tensor,
    attn_scores: Optional[torch.Tensor],
) -> torch.Tensor:
    total = int(features.shape[0])
    salience = torch.zeros(total, dtype=torch.float32, device=features.device)
    attn = _align_scores(attn_scores, total, features.device)
    if attn is not None:
        salience += _minmax(attn)
    if total >= 2:
        change = 1.0 - _cosine(features[:-1], features[1:])
        novelty = torch.zeros(total, dtype=torch.float32, device=features.device)
        novelty[:-1] = torch.maximum(novelty[:-1], change)
        novelty[1:] = torch.maximum(novelty[1:], change)
        salience += 0.25 * _minmax(novelty)
    return salience


def video_spatiotemporal_merge(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    video_attn_mean: torch.Tensor,
    video_token_id: int,
    video_ratio: float,
    spatial_merge_unit: int,
    video_grid_thw: torch.Tensor,
    grid_in_window: int = 2,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert inputs_embeds.dim() == 3 and inputs_embeds.shape[0] == 1, "supports batch_size=1 only"
    device = inputs_embeds.device
    global_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)
    video_positions = (input_ids[0] == video_token_id).nonzero(as_tuple=True)[0]
    total = int(video_positions.numel())
    if total == 0:
        return inputs_embeds, global_mask

    features = inputs_embeds[:, video_positions, :]
    grid_h = int(video_grid_thw[0, 1].item())
    grid_w = int(video_grid_thw[0, 2].item())
    tokens_per_frame = int(grid_h * grid_w // spatial_merge_unit)
    event_bias = _video_event_bias(features[0], tokens_per_frame)
    st_coords = _video_spatiotemporal_coords(total, grid_h, grid_w, spatial_merge_unit, device)
    salience = _mix_attention_with_bias(
        video_attn_mean,
        event_bias,
        total,
        device,
        bias_weight=_ratio_adaptive_bias_weight(video_ratio, 0.35),
    )
    keep_video = _global_retention_pool(
        features[0],
        salience,
        video_ratio,
        st_coords=st_coords,
        dominant_fraction=0.25,
        redundancy_tau=0.70,
        knn_k=7,
    )

    merged_features = _soft_merge_dropped_to_kept(
        features[0],
        keep_video,
        salience=salience,
        st_coords=st_coords,
        coord_weight=0.10,
        center_beta=0.62,
    )
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[0, video_positions, :] = merged_features
    global_mask[video_positions] = keep_video
    return inputs_embeds, global_mask


def audio_temporal_merge(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    audio_attn_mean: torch.Tensor,
    audio_token_id: int,
    audio_ratio: float,
    sec_in_audio_window: int = 2,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert inputs_embeds.dim() == 3 and inputs_embeds.shape[0] == 1, "supports batch_size=1 only"
    device = inputs_embeds.device
    global_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=device)
    audio_positions = (input_ids[0] == audio_token_id).nonzero(as_tuple=True)[0]
    total = int(audio_positions.numel())
    if total == 0:
        return inputs_embeds, global_mask

    features = inputs_embeds[:, audio_positions, :]
    salience = _audio_boundary_salience(features[0], audio_attn_mean)
    t_coords = _temporal_coords(total, device)
    keep_audio = _global_retention_pool(
        features[0],
        salience,
        audio_ratio,
        st_coords=t_coords,
        dominant_fraction=0.35,
        redundancy_tau=0.75,
        knn_k=7,
    )

    merged_features = _soft_merge_dropped_to_kept(
        features[0],
        keep_audio,
        salience=salience,
        st_coords=t_coords,
        coord_weight=0.05,
        center_beta=0.65,
    )
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[0, audio_positions, :] = merged_features
    global_mask[audio_positions] = keep_audio
    return inputs_embeds, global_mask