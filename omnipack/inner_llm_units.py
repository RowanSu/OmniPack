# OmniPack inner-LLM compression
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def apply_token_dropping(
    hidden_states: torch.Tensor,
    keep_mask: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values,
    num_layers_to_prune: int,
):
    keep_indices = keep_mask.nonzero(as_tuple=True)[0]
    hidden_states = hidden_states[:, keep_indices, :]
    position_ids = position_ids[:, :, keep_indices]
    cos, sin = position_embeddings
    position_embeddings = (cos[:, :, keep_indices, :], sin[:, :, keep_indices, :])
    cache_position = cache_position[keep_indices]

    if causal_mask is not None:
        causal_mask = causal_mask[:, :, keep_indices, :][:, :, :, keep_indices]

    if past_key_values is not None:
        if hasattr(past_key_values, "layers"):
            max_layers = min(num_layers_to_prune + 1, len(past_key_values.layers))
            for i in range(max_layers):
                layer_cache = past_key_values.layers[i]
                if layer_cache.get_seq_length() == 0:
                    continue
                layer_cache.keys = layer_cache.keys[:, :, keep_indices, :]
                layer_cache.values = layer_cache.values[:, :, keep_indices, :]
        elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            max_layers = min(num_layers_to_prune + 1, len(past_key_values.key_cache))
            for i in range(max_layers):
                if past_key_values.key_cache[i] is None:
                    continue
                past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, keep_indices, :]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, keep_indices, :]
            if hasattr(past_key_values, "_seen_tokens"):
                past_key_values._seen_tokens -= int((~keep_mask).sum().item())

    return hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values


@torch.no_grad()
def compute_text_query_scores_full(
    hidden_states: torch.Tensor,
    text_token_mask: torch.Tensor,
) -> torch.Tensor:
    text_positions = text_token_mask.nonzero(as_tuple=True)[0]
    if text_positions.numel() == 0:
        return torch.zeros(hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device)
    query = hidden_states[:, text_positions, :].float().mean(dim=1)
    query = F.normalize(query, dim=-1)
    tokens = F.normalize(hidden_states.float(), dim=-1)
    return (tokens * query[:, None, :]).sum(dim=-1).mean(dim=0)


def _minmax(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def _cosine_to_proto(feat: torch.Tensor, proto: Optional[torch.Tensor]) -> torch.Tensor:
    if feat.numel() == 0:
        return torch.empty(0, device=feat.device, dtype=torch.float32)
    if proto is None:
        return torch.zeros(feat.shape[0], device=feat.device, dtype=torch.float32)
    feat_n = F.normalize(feat.float(), dim=-1)
    proto_n = F.normalize(proto.float(), dim=-1)
    return (feat_n * proto_n).sum(dim=-1)


def _text_token_similarity(
    features: torch.Tensor,
    text_features: Optional[torch.Tensor],
) -> torch.Tensor:
    if features.numel() == 0:
        return torch.empty(0, device=features.device, dtype=torch.float32)
    if text_features is None or text_features.numel() == 0:
        return torch.zeros(features.shape[0], device=features.device, dtype=torch.float32)
    feat_n = F.normalize(features.float(), dim=-1)
    text_n = F.normalize(text_features.float(), dim=-1)
    return torch.matmul(feat_n, text_n.transpose(0, 1)).max(dim=-1).values


def _modality_relevance(
    features: torch.Tensor,
    query_scores: torch.Tensor,
    cross_scores: torch.Tensor,
) -> torch.Tensor:
    feat_n = F.normalize(features.float(), dim=-1)
    dist = 1.0 - torch.matmul(feat_n, feat_n.transpose(0, 1))
    density = torch.exp(-dist.mean(dim=-1))
    return _minmax(query_scores.float()) + _minmax(cross_scores.float()) + _minmax(density)


def _select_modality_keep(
    features: torch.Tensor,
    relevance: torch.Tensor,
    keep_n: int,
) -> torch.Tensor:
    n = int(features.shape[0])
    keep_n = max(0, min(int(keep_n), n))
    mask = torch.zeros(n, dtype=torch.bool, device=features.device)
    if keep_n <= 0:
        return mask
    if keep_n >= n:
        mask[:] = True
        return mask

    feat_n = F.normalize(features.float(), dim=-1)
    dist = 1.0 - torch.matmul(feat_n, feat_n.transpose(0, 1))

    # Keep a strong relevance seed, then use max-min diversity under relevance pressure.
    first = torch.argmax(relevance)
    selected = [first]
    min_dist = dist[first].clone()
    min_dist[first] = -1.0
    for _ in range(1, keep_n):
        score = min_dist * (1.0 + _minmax(relevance))
        idx = torch.argmax(score)
        selected.append(idx)
        min_dist = torch.minimum(min_dist, dist[idx])
        min_dist[idx] = -1.0
    mask[torch.stack(selected)] = True
    return mask


def _merge_pruned_to_kept(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    keep_local: torch.Tensor,
    relevance: torch.Tensor,
) -> torch.Tensor:
    if keep_local.all() or (~keep_local).sum().item() == 0 or keep_local.sum().item() == 0:
        return hidden_states

    kept_positions = positions[keep_local]
    pruned_positions = positions[~keep_local]
    kept_feat = hidden_states[0, kept_positions]
    pruned_feat = hidden_states[0, pruned_positions]

    kept_n = F.normalize(kept_feat.float(), dim=-1)
    pruned_n = F.normalize(pruned_feat.float(), dim=-1)
    sim = torch.matmul(pruned_n, kept_n.transpose(0, 1))
    best_sim, assign = sim.max(dim=-1)

    pruned_rel = _minmax(relevance[~keep_local].float())
    sim_weight = ((best_sim + 1.0) * 0.5).clamp(0.0, 1.0).square()
    weights = sim_weight * (0.5 + 0.5 * pruned_rel)

    accum = torch.zeros_like(kept_feat.float())
    denom = torch.zeros(kept_feat.shape[0], 1, device=kept_feat.device, dtype=torch.float32)
    accum.index_add_(0, assign, pruned_feat.float() * weights.unsqueeze(-1))
    denom.index_add_(0, assign, weights.unsqueeze(-1))

    updated = (kept_feat.float() + accum) / (1.0 + denom)
    hidden_states = hidden_states.clone()
    hidden_states[0, kept_positions] = updated.to(dtype=hidden_states.dtype)
    return hidden_states


@torch.no_grad()
def stage2_token_selection(
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    position_ids: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values,
    layer_idx: int,
    decoder_layers,
    progressive_config: Dict,
    audio_token_mask: torch.Tensor,
    text_token_mask: torch.Tensor,
    video_token_mask: torch.Tensor,
):
    """OmniPack Stage II: one-shot text-query and audio-video mutual pruning."""
    video_keep_ratio = float(progressive_config.get("video_stage2_keep_ratio", 1.0))
    audio_keep_ratio = float(progressive_config.get("audio_stage2_keep_ratio", 1.0))

    seq_len = hidden_states.shape[1]
    device = hidden_states.device
    global_keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)

    n_video = int(video_token_mask.sum().item())
    n_audio = int(audio_token_mask.sum().item())
    if n_video == 0 and n_audio == 0:
        return hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values, global_keep_mask

    scores_full = compute_text_query_scores_full(
        hidden_states=hidden_states,
        text_token_mask=text_token_mask,
    )

    hs = hidden_states[0]
    video_positions = video_token_mask.nonzero(as_tuple=True)[0]
    audio_positions = audio_token_mask.nonzero(as_tuple=True)[0]
    text_positions = text_token_mask.nonzero(as_tuple=True)[0]
    video_feat = hs[video_positions] if n_video > 0 else None
    audio_feat = hs[audio_positions] if n_audio > 0 else None
    text_feat = hs[text_positions] if text_positions.numel() > 0 else None
    video_proto = video_feat.mean(dim=0) if n_video > 0 else None
    audio_proto = audio_feat.mean(dim=0) if n_audio > 0 else None

    if n_video > 0:
        keep_n = max(1, min(n_video, round(video_keep_ratio * n_video)))
        query_scores = torch.maximum(scores_full[video_token_mask], _text_token_similarity(video_feat, text_feat))
        cross_scores = _cosine_to_proto(video_feat, audio_proto)
        relevance = _modality_relevance(video_feat, query_scores, cross_scores)
        keep_local = _select_modality_keep(video_feat, relevance, keep_n)
        hidden_states = _merge_pruned_to_kept(hidden_states, video_positions, keep_local, relevance)
        global_keep_mask[video_positions[~keep_local]] = False

    if n_audio > 0:
        keep_n = max(1, min(n_audio, round(audio_keep_ratio * n_audio)))
        query_scores = torch.maximum(scores_full[audio_token_mask], _text_token_similarity(audio_feat, text_feat))
        cross_scores = _cosine_to_proto(audio_feat, video_proto)
        relevance = _modality_relevance(audio_feat, query_scores, cross_scores)
        keep_local = _select_modality_keep(audio_feat, relevance, keep_n)
        hidden_states = _merge_pruned_to_kept(hidden_states, audio_positions, keep_local, relevance)
        global_keep_mask[audio_positions[~keep_local]] = False

    hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values = apply_token_dropping(
        hidden_states=hidden_states,
        keep_mask=global_keep_mask,
        causal_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        cache_position=cache_position,
        past_key_values=past_key_values,
        num_layers_to_prune=layer_idx,
    )
    return hidden_states, causal_mask, position_ids, position_embeddings, cache_position, past_key_values, global_keep_mask