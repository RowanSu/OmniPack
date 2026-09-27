import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def similarity_pruning(
    compressioner,
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    audio_token_id: int,
    video_token_id: int,
    position_ids: Optional[torch.LongTensor] = None,
    merging_ratio_a: float = 0.3,
    merging_ratio_v: float = 0.5,
):
    # print("try spatial-temporal-audio-video-cross-modality-compression")
    device = inputs_embeds.device
    batch_size, seq_len, _ = inputs_embeds.shape
    global_keep_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        sample_input_ids = input_ids[i]
        video_token_mask = (sample_input_ids == video_token_id)
        audio_token_mask = (sample_input_ids == audio_token_id)
        
        sample_pos = position_ids[:, i, :] 
        time_ids = sample_pos[0, :]
        height_ids = sample_pos[1, :]
        width_ids = sample_pos[2, :]
        
        unique_frames = sorted(time_ids[video_token_mask].unique().tolist())
        if not unique_frames:
            continue

        frames_per_group = 2
        for idx in range(0, len(unique_frames), frames_per_group):
            # 1. Locate video frames in the current chunk.
            v_frames_in_chunk = unique_frames[idx : idx + frames_per_group]
            t_start = v_frames_in_chunk[0]
            t_end = unique_frames[idx + frames_per_group] if (idx + frames_per_group) < len(unique_frames) else (time_ids.max().item() + 1)

            # --- A. Video compression (spatial for F1, temporal for F2) ---
            
            # First-frame indices.
            f1_indices = torch.nonzero((time_ids == t_start) & video_token_mask, as_tuple=True)[0]
            k_v = int(len(f1_indices) * merging_ratio_v) # Number of tokens to drop.

            # 1. Spatial Pruning on Frame 1
            if len(f1_indices) > 0 and k_v > 0:
                f1_embeds = inputs_embeds[i, f1_indices]
                # Compute the similarity between each patch and the frame-level mean feature.
                f1_mean = f1_embeds.mean(dim=0, keepdim=True)
                sim_spatial = F.cosine_similarity(f1_embeds, f1_mean, dim=-1)
                _, drop_s_idx = torch.topk(sim_spatial, k=k_v, largest=True)
                global_keep_mask[i, f1_indices[drop_s_idx]] = False

            # 2. Temporal Pruning on Frame 2
            if len(v_frames_in_chunk) > 1:
                f2_t = v_frames_in_chunk[1]
                f2_indices = torch.nonzero((time_ids == f2_t) & video_token_mask, as_tuple=True)[0]
                
                if len(f1_indices) == len(f2_indices) and k_v > 0:
                    f1_embeds_full = inputs_embeds[i, f1_indices]
                    f2_embeds_full = inputs_embeds[i, f2_indices]
                    # Compute similarity between each F2 patch and the corresponding F1 patch.
                    sim_temporal = F.cosine_similarity(f1_embeds_full, f2_embeds_full, dim=-1)
                    _, drop_t_idx = torch.topk(sim_temporal, k=k_v, largest=True)
                    global_keep_mask[i, f2_indices[drop_t_idx]] = False

            # --- B. Cross-modal compression (audio selection) ---
            
            # All audio tokens in the current chunk.
            a_idx_chunk = torch.nonzero((time_ids >= t_start) & (time_ids < t_end) & audio_token_mask, as_tuple=True)[0]
            
            # Use the remaining video tokens after video compression as context.
            v_idx_remain = torch.nonzero(
                (time_ids >= t_start) & (time_ids < t_end) & video_token_mask & global_keep_mask[i],
                as_tuple=True
            )[0]
            
            if len(v_idx_remain) > 0 and len(a_idx_chunk) > 1:
                v_embeds_ctx = inputs_embeds[i, v_idx_remain]
                a_embeds_chunk = inputs_embeds[i, a_idx_chunk]
                
                # Selector scores.
                audio_scores = compressioner(v_embeds_ctx, a_embeds_chunk)
                
                n_keep_a = len(a_idx_chunk) - int(len(a_idx_chunk) * merging_ratio_a)
                if n_keep_a > 0:
                    _, topk_keep_a_idx = torch.topk(audio_scores, k=n_keep_a, largest=True)
                    
                    hard_mask = torch.zeros_like(audio_scores)
                    hard_mask.scatter_(dim=-1, index=topk_keep_a_idx, value=1.0)
                    
                    # STE gradient.
                    ste_mask = (hard_mask - audio_scores).detach() + audio_scores
                    inputs_embeds[i, a_idx_chunk] = inputs_embeds[i, a_idx_chunk] * ste_mask.unsqueeze(-1)
                    
                    # Update the global mask.
                    drop_a_indices = a_idx_chunk[torch.nonzero(hard_mask == 0, as_tuple=True)[0]]
                    global_keep_mask[i, drop_a_indices] = False

    # Debug statistics.
    video_count = (global_keep_mask & (input_ids == video_token_id)).sum().item()
    audio_count = (global_keep_mask & (input_ids == audio_token_id)).sum().item()

    return inputs_embeds, global_keep_mask, video_count, audio_count

