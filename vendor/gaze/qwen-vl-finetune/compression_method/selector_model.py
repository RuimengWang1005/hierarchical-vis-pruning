# -*- coding: utf-8 -*-
"""
Qwen2.5-VL BBox监督的Token选择训练模块

【与原版本的架构差异对比】
====================================

原版本(/home/bbhan/projects/myproject/qwen-vl-finetune/compression_method/selector_model.py):
- 仅使用视觉特征进行Token选择
- 依赖约束损失（soft vs hard mask）进行训练
- 固定全局预算分配
- 无内容感知的Token选择

这个版本(visionselectorbbox):
- 仅使用视觉特征进行Token选择
- BBox监督损失（BCE + Dice）替代约束损失
- 基于内容的动态预算分配

【核心改进详解】
=================
1. 【BBox监督集成】：
   - 边界框信息指导Token选择和预算分配
   - BCE + Dice损失提供直接内容监督

2. 【动态预算策略】：
   - 基于BBox前景比例的智能预算分配
   - 替代原版本的固定全局预算

【训练机制革新】
===============
- 原版本：L_constraint 仅强制形式约束（分数接近0/1）
- 这个版本：L_bbox 提供内容监督（前景vs背景识别）

【文件结构差异】
===============
新增功能模块：
- bbox_supervision.py: BBox监督工具函数
- _compute_ref_token_mask_supervision_loss_impl: token-mask监督损失计算
- 文本特征提取和对齐函数（_group_tokens_by_sample等）

增强功能：
- qwen25vl_vision_tower_forward_selector: 支持BBox预算
- qwen25vl_generation_forward_selector: 集成BBox损失
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None
logger = logging.get_logger(__name__)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from copy import deepcopy
from torch import vmap
from torch.func import grad
from torch.autograd import Function

# 【新增】导入BBox监督工具函数
# 与原版本不同：原版本没有BBox监督功能
from compression_method.bbox_supervision import compute_bbox_rank_loss_full

# ============================================================================
# 可微分TopK实现 - 与原版本相同的基础机制
# ============================================================================
# 注意：这部分实现与原版本保持一致，提供可微分的Token选择功能
# 唯一区别：这个版本专门为BBox监督进行了优化，支持基于内容的Token选择

sigmoid = torch.sigmoid
sigmoid_grad = vmap(vmap(grad(sigmoid)))

class TopK(Function):
    @staticmethod
    def forward(ctx, xs, k):
        """
        可微分TopK前向传播
        
        与原版本功能相同，通过二分查找找到合适的阈值，使得恰好k个Token被选中。
        返回的选择概率可以作为软掩码用于梯度反向传播。
        """
        ts, ps = _find_ts(xs, k)
        ctx.save_for_backward(xs, ts)
        return ps

    @staticmethod
    def backward(ctx, grad_output):
        """
        可微分TopK反向传播
        
        计算选择概率对输入Token分数的梯度，使得Token选择过程可以端到端训练。
        使用向量-雅可比积(vjp)高效计算梯度。
        """
        xs, ts = ctx.saved_tensors
        # v = sigmoid'(x + t)
        v = sigmoid_grad(xs + ts)
        s = v.sum(dim=1, keepdims=True)
        # Jacobian is -vv.T/s + diag(v)
        uv = grad_output * v
        t1 = - uv.sum(dim=1, keepdims=True) * v / s
        return t1 + uv, None

@torch.no_grad()
def _find_ts(xs, k):
    """
    通过二分查找找到使得sigmoid(xs + t)恰好有k个Token被选中的阈值t
    
    Args:
        xs: Token重要性分数 [batch_size, num_tokens]
        k: 要选择的Token数量
        
    Returns:
        ts: 找到的阈值
        sigmoid(xs + ts): 选择概率掩码
    """
    b, n = xs.shape
    assert 0 < k < n
    # Lo应该足够小，使得所有sigmoid都在0区域
    # Hi应该足够大，使得所有sigmoid都在1区域
    lo = -xs.max(dim=1, keepdims=True).values - 10
    hi = -xs.min(dim=1, keepdims=True).values + 10
    for _ in range(64):
        mid = (hi + lo)/2
        mask = sigmoid(xs + mid).sum(dim=1) < k
        lo[mask] = mid[mask]
        hi[~mask] = mid[~mask]
    ts = (lo + hi)/2
    return ts, sigmoid(xs + ts)

topk = TopK.apply
# ============================================================================

def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.to(q.dtype).chunk(2, dim=-1)[0].contiguous()
    sin = sin.to(q.dtype).chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q, cos, sin).type_as(q)
    k_embed = apply_rotary_emb(k, cos, sin).type_as(k)
    return q_embed, k_embed

# [Fixed]: Helper to derive token length from grid_thw entry with spatial_merge_size
def _get_visual_token_len(thw: torch.Tensor, merge_size: int = 2) -> int:
    thw_long = thw.to(dtype=torch.long)
    t = thw_long[0].item()
    h = thw_long[1].item()
    w = thw_long[2].item()
    # Keep train-time token length rule aligned with inference-time selector logic.
    h_m = max(int(h) // merge_size, 1)
    w_m = max(int(w) // merge_size, 1)
    t_m = max(int(t), 1)
    return int(t_m * h_m * w_m)


def _grid_thw_to_token_lengths(grid_thw: Optional[torch.Tensor], merge_size: int) -> List[int]:
    if grid_thw is None:
        return []
    if grid_thw.dim() == 1:
        grid_thw = grid_thw.unsqueeze(0)
    return [_get_visual_token_len(entry, merge_size=merge_size) for entry in grid_thw]


def _split_foc_spa_budget(
    k_tokens: int,
    spatial_ratio: float,
) -> Tuple[int, int]:
    k = max(1, int(k_tokens))
    r = max(0.0, min(float(spatial_ratio), 1.0))
    k_spa = int(round(k * r))
    k_spa = max(0, min(k_spa, k))
    k_foc = k - k_spa
    if k_foc <= 0:
        # Keep at least one focal token for stability.
        k_foc = 1
        k_spa = k - k_foc
    return k_foc, k_spa


def _normalize_cosine_params(
    penalty_weight: float,
    similarity_threshold: float,
) -> Tuple[float, float]:
    weight = max(0.0, float(penalty_weight))
    threshold = max(-1.0, min(1.0, float(similarity_threshold)))
    return weight, threshold


def _robust_iqr(x: torch.Tensor) -> torch.Tensor:
    x = x.to(dtype=torch.float32)
    if x.numel() <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    q1 = torch.quantile(x, 0.25)
    q3 = torch.quantile(x, 0.75)
    return torch.clamp(q3 - q1, min=0.0)


def _compute_scale_matched_lambda(
    scores: torch.Tensor,
    penalty_base: torch.Tensor,
    alpha: float,
    eps: float = 1e-6,
    max_lambda: float = 100.0,
) -> float:
    if alpha <= 0.0 or scores.numel() == 0 or penalty_base.numel() == 0:
        return 0.0

    score_scale = _robust_iqr(scores)
    if float(score_scale) <= eps:
        score_scale = torch.clamp(scores.max() - scores.min(), min=0.0)

    penalty_scale = _robust_iqr(penalty_base)
    if float(penalty_scale) <= eps:
        penalty_scale = torch.clamp(penalty_base.max() - penalty_base.min(), min=0.0)

    if float(penalty_scale) <= eps:
        return 0.0

    lam = alpha * (score_scale / (penalty_scale + eps))
    lam = torch.clamp(lam, min=0.0, max=max_lambda)
    return float(lam.item())


@torch.no_grad()
def _select_indices_with_cosine_penalty(
    scores: torch.Tensor,
    token_features: Optional[torch.Tensor],
    k: int,
    penalty_weight: float,
    similarity_threshold: float,
    reference_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Greedy diversity selection with cosine-similarity penalty.

    At each step, select argmax(score - penalty), where penalty is proportional to
    max cosine similarity to already selected/reference tokens.
    """
    total = int(scores.numel())
    if total == 0 or k <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)

    k = min(int(k), total)
    if k >= total:
        return torch.arange(total, dtype=torch.long, device=scores.device)

    penalty_weight, similarity_threshold = _normalize_cosine_params(
        penalty_weight, similarity_threshold
    )

    if (
        penalty_weight <= 0.0
        or token_features is None
        or token_features.dim() != 2
        or token_features.shape[0] != total
    ):
        return torch.topk(scores, k=k, dim=0, largest=True).indices

    feat = F.normalize(
        token_features.to(dtype=torch.float32),
        p=2,
        dim=-1,
        eps=1e-6,
    )
    base_scores = scores.to(dtype=torch.float32)
    max_sim = torch.zeros(total, device=scores.device, dtype=torch.float32)

    if (
        reference_features is not None
        and isinstance(reference_features, torch.Tensor)
        and reference_features.dim() == 2
        and reference_features.numel() > 0
        and reference_features.shape[1] == feat.shape[1]
    ):
        ref_feat = F.normalize(
            reference_features.to(dtype=torch.float32),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        max_sim = torch.matmul(feat, ref_feat.transpose(0, 1)).max(dim=1).values

    selected_mask = torch.zeros(total, dtype=torch.bool, device=scores.device)
    selected: List[int] = []
    has_reference = (
        reference_features is not None
        and isinstance(reference_features, torch.Tensor)
        and reference_features.dim() == 2
        and reference_features.numel() > 0
        and reference_features.shape[1] == feat.shape[1]
    )
    lambda_eff = 0.0

    # Single-shot adaptive lambda: calibrate once per sample, then keep it fixed
    # while the greedy loop updates only the similarity penalty.
    if has_reference:
        penalty_base = torch.relu(max_sim - similarity_threshold)
        lambda_eff = _compute_scale_matched_lambda(
            base_scores,
            penalty_base,
            alpha=penalty_weight,
        )
    else:
        seed_idx = int(torch.argmax(base_scores).item())
        selected_mask[seed_idx] = True
        selected.append(seed_idx)
        if len(selected) < k:
            sim_to_new = torch.matmul(feat, feat[seed_idx])
            max_sim = torch.maximum(max_sim, sim_to_new)
            penalty_base = torch.relu(max_sim - similarity_threshold)
            candidate_mask = ~selected_mask
            lambda_eff = _compute_scale_matched_lambda(
                base_scores[candidate_mask],
                penalty_base[candidate_mask],
                alpha=penalty_weight,
            )

    while len(selected) < k:
        penalty_base = torch.relu(max_sim - similarity_threshold)
        penalized = base_scores - lambda_eff * penalty_base
        penalized = penalized.masked_fill(selected_mask, float("-inf"))

        next_idx = int(torch.argmax(penalized).item())
        if selected_mask[next_idx]:
            break

        selected_mask[next_idx] = True
        selected.append(next_idx)
        if len(selected) >= k:
            break

        sim_to_new = torch.matmul(feat, feat[next_idx])
        max_sim = torch.maximum(max_sim, sim_to_new)

    if len(selected) < k:
        remaining = torch.where(~selected_mask)[0]
        if remaining.numel() > 0:
            need = min(k - len(selected), int(remaining.numel()))
            top_rel = torch.topk(base_scores[remaining], k=need, largest=True).indices
            selected.extend(remaining[top_rel].tolist())

    if not selected:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    return torch.tensor(selected, dtype=torch.long, device=scores.device)


@torch.no_grad()
def _select_spatial_indices_grid_rounds(
    scores_non_foc: torch.Tensor,
    non_foc_within: torch.Tensor,
    cell_id: torch.Tensor,
    budget: int,
    token_features_non_foc: Optional[torch.Tensor] = None,
    reference_features: Optional[torch.Tensor] = None,
    cosine_penalty_weight: float = 0.0,
    cosine_similarity_threshold: float = 0.85,
) -> torch.Tensor:
    """
    Select spatial tokens with grid-aware rounds:
    1) coverage round: pick top-1 from each non-empty cell;
    2) subsequent rounds: compete among each cell's next candidate by score.
    """
    device = scores_non_foc.device
    total = int(non_foc_within.numel())
    if budget <= 0 or total == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    budget = min(int(budget), total)

    def _pick_from_candidates(
        candidate_rel: torch.Tensor,
        k: int,
        ref_feats: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cand_scores = scores_non_foc[candidate_rel]
        cand_features = None
        if token_features_non_foc is not None:
            cand_features = token_features_non_foc[candidate_rel]
        chosen_rel = _select_indices_with_cosine_penalty(
            cand_scores,
            cand_features,
            k,
            cosine_penalty_weight,
            cosine_similarity_threshold,
            reference_features=ref_feats,
        )
        if chosen_rel.numel() != k:
            chosen_rel = torch.topk(cand_scores, k=k, largest=True).indices
        return candidate_rel[chosen_rel]

    unique_cells = torch.unique(cell_id, sorted=True)
    cell_rel_lists: List[torch.Tensor] = []
    cell_ptrs: List[int] = []
    first_round_rel: List[int] = []

    for cell in unique_cells.tolist():
        rel_idxs = torch.where(cell_id == cell)[0]
        if rel_idxs.numel() == 0:
            continue
        rel_order = torch.argsort(scores_non_foc[rel_idxs], descending=True)
        rel_sorted = rel_idxs[rel_order]
        cell_rel_lists.append(rel_sorted)
        cell_ptrs.append(1)
        first_round_rel.append(int(rel_sorted[0].item()))

    if len(first_round_rel) == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    first_candidates = torch.tensor(first_round_rel, dtype=torch.long, device=device)
    if budget <= int(first_candidates.numel()):
        chosen_first = _pick_from_candidates(first_candidates, budget, reference_features)
        return non_foc_within[chosen_first]

    selected_rel: List[int] = list(first_round_rel)
    remaining = budget - len(selected_rel)

    while remaining > 0:
        round_rel: List[int] = []
        round_cells: List[int] = []
        for cell_idx, rel_sorted in enumerate(cell_rel_lists):
            ptr = cell_ptrs[cell_idx]
            if ptr < int(rel_sorted.numel()):
                round_rel.append(int(rel_sorted[ptr].item()))
                round_cells.append(cell_idx)

        if len(round_rel) == 0:
            break

        round_candidates = torch.tensor(round_rel, dtype=torch.long, device=device)
        ref_feats = reference_features
        if token_features_non_foc is not None and len(selected_rel) > 0:
            selected_rel_tensor = torch.tensor(selected_rel, dtype=torch.long, device=device)
            selected_feats = token_features_non_foc[selected_rel_tensor]
            if ref_feats is None or ref_feats.numel() == 0:
                ref_feats = selected_feats
            else:
                ref_feats = torch.cat([ref_feats, selected_feats], dim=0)

        chosen = _pick_from_candidates(round_candidates, 1, ref_feats)
        chosen_rel = int(chosen[0].item())
        selected_rel.append(chosen_rel)

        chosen_cell = round_cells[round_rel.index(chosen_rel)]
        cell_ptrs[chosen_cell] += 1
        remaining -= 1

    if len(selected_rel) == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    selected_rel_tensor = torch.tensor(selected_rel[:budget], dtype=torch.long, device=device)
    return non_foc_within[selected_rel_tensor]


def _compute_foc_spa_masks(
    scores_slice: torch.Tensor,
    k_tokens: int,
    grid_thw_entry: Optional[torch.Tensor],
    merge_size: int,
    spatial_ratio: float,
    token_features_slice: Optional[torch.Tensor] = None,
    cosine_penalty_weight: float = 0.0,
    cosine_similarity_threshold: float = 0.85,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    length = int(scores_slice.numel())
    device = scores_slice.device
    dtype = scores_slice.dtype

    soft_foc = torch.zeros((length,), device=device, dtype=dtype)
    soft_spa = torch.zeros((length,), device=device, dtype=dtype)
    hard_foc = torch.zeros((length,), device=device, dtype=dtype)
    hard_spa = torch.zeros((length,), device=device, dtype=dtype)

    if length == 0:
        return soft_foc, soft_spa, hard_foc, hard_spa

    if k_tokens >= length:
        hard_foc.fill_(1.0)
        soft_foc.fill_(1.0)
        return soft_foc, soft_spa, hard_foc, hard_spa

    k_foc, k_spa = _split_foc_spa_budget(k_tokens, spatial_ratio)
    k_foc = max(1, min(k_foc, length - 1))
    k_spa = max(0, min(k_spa, length - k_foc))

    features_slice = None
    if (
        isinstance(token_features_slice, torch.Tensor)
        and token_features_slice.dim() == 2
        and token_features_slice.shape[0] == length
    ):
        features_slice = token_features_slice

    if 0 < k_foc < length:
        foc_indices = _select_indices_with_cosine_penalty(
            scores_slice,
            features_slice,
            k_foc,
            cosine_penalty_weight,
            cosine_similarity_threshold,
        )
        if foc_indices.numel() != k_foc:
            foc_indices = scores_slice.topk(k_foc, dim=0).indices
        hard_foc.scatter_(0, foc_indices, 1.0)
        soft_foc = topk(scores_slice.unsqueeze(0), k_foc).squeeze(0)
    else:
        hard_foc.fill_(1.0)
        soft_foc.fill_(1.0)
        return soft_foc, soft_spa, hard_foc, hard_spa

    if k_spa <= 0:
        return soft_foc, soft_spa, hard_foc, hard_spa

    non_foc_within = torch.where(hard_foc == 0)[0]
    if non_foc_within.numel() == 0:
        return soft_foc, soft_spa, hard_foc, hard_spa

    scores_non_foc = scores_slice[non_foc_within]
    features_non_foc = None
    if features_slice is not None:
        features_non_foc = features_slice[non_foc_within]
    foc_features = None
    if features_slice is not None and foc_indices.numel() > 0:
        foc_features = features_slice[foc_indices]

    k_spa = min(k_spa, int(non_foc_within.numel()))

    use_grid = grid_thw_entry is not None and grid_thw_entry.numel() >= 3
    if not use_grid:
        if k_spa >= scores_non_foc.numel():
            hard_spa[non_foc_within] = 1.0
            soft_spa[non_foc_within] = 1.0
            return soft_foc, soft_spa, hard_foc, hard_spa
        spa_indices_rel = _select_indices_with_cosine_penalty(
            scores_non_foc,
            features_non_foc,
            k_spa,
            cosine_penalty_weight,
            cosine_similarity_threshold,
            reference_features=foc_features,
        )
        if spa_indices_rel.numel() != k_spa:
            spa_indices_rel = scores_non_foc.topk(k_spa, dim=0).indices
        hard_spa[non_foc_within[spa_indices_rel]] = 1.0
        soft_vals = topk(scores_non_foc.unsqueeze(0), k_spa).squeeze(0)
        soft_spa[non_foc_within] = soft_vals
        return soft_foc, soft_spa, hard_foc, hard_spa

    t_i, h_i, w_i = grid_thw_entry.to(dtype=torch.long)[:3].tolist()
    h_m = max(int(h_i) // merge_size, 1)
    w_m = max(int(w_i) // merge_size, 1)
    h_tot = max(int(t_i) * int(h_m), 1)
    w_tot = max(int(w_m), 1)

    grid_cols = max(1, int(math.ceil(math.sqrt(float(k_spa) * (float(w_tot) / float(h_tot))))))
    grid_rows = max(1, int(math.ceil(float(k_spa) / float(grid_cols))))
    row_step = max(1, int(math.ceil(float(h_tot) / float(grid_rows))))
    col_step = max(1, int(math.ceil(float(w_tot) / float(grid_cols))))
    num_cells = int(grid_rows * grid_cols)

    row = (non_foc_within // w_tot).clamp(min=0, max=h_tot - 1)
    col = (non_foc_within % w_tot).clamp(min=0, max=w_tot - 1)
    cell_r = (row // row_step).clamp(min=0, max=grid_rows - 1)
    cell_c = (col // col_step).clamp(min=0, max=grid_cols - 1)
    cell_id = cell_r * grid_cols + cell_c

    selected_within = _select_spatial_indices_grid_rounds(
        scores_non_foc=scores_non_foc,
        non_foc_within=non_foc_within,
        cell_id=cell_id,
        budget=k_spa,
        token_features_non_foc=features_non_foc,
        reference_features=foc_features,
        cosine_penalty_weight=cosine_penalty_weight,
        cosine_similarity_threshold=cosine_similarity_threshold,
    )

    hard_spa[selected_within] = 1.0

    cell_scores = torch.full((num_cells,), -1e9, device=device, dtype=dtype)
    for cell in range(num_cells):
        rel_idxs = torch.where(cell_id == cell)[0]
        if rel_idxs.numel() == 0:
            continue
        cell_scores[cell] = scores_non_foc[rel_idxs].max()

    valid_cells = cell_scores > -1e8
    valid_count = int(valid_cells.sum().item())
    if valid_count == 0:
        return soft_foc, soft_spa, hard_foc, hard_spa

    if k_spa >= valid_count or num_cells <= 1:
        cell_soft = torch.zeros((num_cells,), device=device, dtype=dtype)
        cell_soft[valid_cells] = 1.0
    else:
        cell_soft = topk(cell_scores.unsqueeze(0), k_spa).squeeze(0)

    for cell in range(num_cells):
        rel_idxs = torch.where(cell_id == cell)[0]
        if rel_idxs.numel() == 0:
            continue
        token_weights = torch.softmax(scores_non_foc[rel_idxs], dim=0)
        soft_spa[non_foc_within[rel_idxs]] = cell_soft[cell] * token_weights

    return soft_foc, soft_spa, hard_foc, hard_spa


def qwen25vl_vision_tower_forward_selector(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    ref_token_masks: Optional[List[Optional[torch.Tensor]]] = None,
) -> torch.Tensor:
    """
    【重要改进】视觉Transformer前向传播函数 - BBox监督版本
    
    【与原版本的关键区别】：
    1. 使用ref_token_masks：直接以token级mask监督Token选择（不走“bbox一轮对话”）
    2. 增加了基于BBox的动态预算分配机制
    3. 返回值增加learned_scores用于BBox损失计算
    
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.
        ref_token_masks: Token级参考mask，按拼接后的图像顺序对齐（每张图一个mask）

    Returns:
        tuple: 【修改】返回8个元素而非3个
            - hidden_states_new: 处理后的视觉特征
            - img_mask: 软选择掩码
            - constraint_img_mask: 硬选择掩码
            - learned_scores: Token重要性分数，用于BBox监督损失
            - soft_foc_mask: focal soft mask
            - soft_spa_mask: spatial soft mask
            - hard_foc_mask: focal hard mask
            - hard_spa_mask: spatial hard mask
    """
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    # ---------------------------BBox监督与Token选择---------------------------
    hidden_states_unsqueezed = hidden_states.unsqueeze(0)

    # Get spatial_merge_size from self, default to 2 if not found
    merge_size = getattr(self, 'spatial_merge_size', 2)

    learned_scores = self.importance_scorer(hidden_states_unsqueezed).squeeze(0)
    
    total_tokens = learned_scores.shape[0]
    device = learned_scores.device
    dtype = learned_scores.dtype
    spatial_ratio = float(
        getattr(
            self,
            "spatial_ratio",
            getattr(
                self,
                "contextual_ratio",
                float(os.getenv("SELECTOR_SPATIAL_RATIO", os.getenv("SELECTOR_CONTEXTUAL_RATIO", "0.1"))),
            ),
        )
    )
    cosine_penalty_weight = float(
        getattr(
            self,
            "cosine_penalty_weight",
            float(os.getenv("SELECTOR_COSINE_PENALTY_WEIGHT", "0.15")),
        )
    )
    cosine_penalty_weight = max(0.0, cosine_penalty_weight)
    cosine_similarity_threshold = float(
        getattr(
            self,
            "cosine_penalty_threshold",
            float(os.getenv("SELECTOR_COSINE_PENALTY_THRESHOLD", "0.85")),
        )
    )
    cosine_similarity_threshold = max(-1.0, min(1.0, cosine_similarity_threshold))
    debug_enabled = getattr(self, "bbox_debug", False) or os.getenv("BBOX_DEBUG") == "1"

    # 【新增】推导每张图像的Token长度以支持每图像预算分配
    # 与原版本不同：原版本使用全局预算，这里支持基于BBox的动态预算
    token_lengths: List[int] = _grid_thw_to_token_lengths(grid_thw, merge_size) if grid_thw is not None else []
    use_bbox_budget = (
        ref_token_masks is not None  # 【新增】需要token级mask信息
        and grid_thw is not None
        and isinstance(grid_thw, torch.Tensor)
        and len(token_lengths) == grid_thw.shape[0]
        and sum(token_lengths) == total_tokens
    )
    debug_reason = "ref+grid aligned" if use_bbox_budget else "missing ref/grid alignment"
    if use_bbox_budget and ref_token_masks is not None and len(ref_token_masks) != len(token_lengths):
        use_bbox_budget = False
        debug_reason = "ref length mismatch"
    if debug_enabled:
        debug_pre = {
            "has_ref": ref_token_masks is not None,
            "ref_len": len(ref_token_masks) if ref_token_masks is not None else 0,
            "grid_shape0": grid_thw.shape[0] if isinstance(grid_thw, torch.Tensor) else None,
            "token_len_len": len(token_lengths),
            "token_len_sum": sum(token_lengths) if token_lengths else 0,
            "total_tokens": total_tokens,
        }

    if use_bbox_budget:
        # 【新增】BBox预算策略参数 - 与原版本完全不同的预算分配方式
        # 原版本：使用固定的全局预算比例
        # 这个版本：基于BBox内容的动态预算分配
        min_keep_ratio = getattr(self, "bbox_keep_min_ratio", 0.1)
        max_keep_ratio = getattr(self, "bbox_keep_max_ratio", 0.6)

        soft_foc_masks: List[torch.Tensor] = []
        soft_spa_masks: List[torch.Tensor] = []
        hard_foc_masks: List[torch.Tensor] = []
        hard_spa_masks: List[torch.Tensor] = []
        k_tokens_per_image: List[int] = []
        k_ratio_per_image: List[float] = []
        k_foc_per_image: List[int] = []
        k_spa_per_image: List[int] = []
        offset = 0

        ref_used = 0
        for img_idx, length in enumerate(token_lengths):
            scores_slice = learned_scores[offset:offset + length]
            ref_mask = ref_token_masks[img_idx] if (ref_token_masks is not None and img_idx < len(ref_token_masks)) else None

            # 回退到默认全局预算
            k_ratio = self.budgets
            if ref_mask is not None:
                flat_mask = _flatten_ref_token_mask(ref_mask, length)
                if flat_mask is not None and flat_mask.numel() == length:
                    ref_used += 1
                    fg_ratio = flat_mask.to(dtype=torch.float32).mean().item()
                    k_ratio = max(min_keep_ratio, min(fg_ratio, max_keep_ratio))

            k_tokens = max(1, int(math.ceil(length * k_ratio)))
            if k_tokens >= length:
                k_tokens = length - 1 if length > 1 else 1
            k_tokens_per_image.append(k_tokens)
            k_ratio_per_image.append(k_tokens / max(length, 1))
            k_foc_dbg, k_spa_dbg = _split_foc_spa_budget(k_tokens, spatial_ratio)
            k_foc_dbg = max(1, min(k_foc_dbg, length - 1))
            k_spa_dbg = max(0, min(k_spa_dbg, length - k_foc_dbg))
            k_foc_per_image.append(k_foc_dbg)
            k_spa_per_image.append(k_spa_dbg)

            grid_entry = None
            if isinstance(grid_thw, torch.Tensor) and grid_thw.numel() > 0:
                if grid_thw.dim() == 1 and grid_thw.numel() >= 3:
                    grid_entry = grid_thw
                elif grid_thw.dim() > 1 and img_idx < grid_thw.shape[0]:
                    grid_entry = grid_thw[img_idx]

            soft_foc, soft_spa, hard_foc, hard_spa = _compute_foc_spa_masks(
                scores_slice,
                k_tokens,
                grid_entry,
                merge_size,
                spatial_ratio,
                token_features_slice=hidden_states[offset:offset + length],
                cosine_penalty_weight=cosine_penalty_weight,
                cosine_similarity_threshold=cosine_similarity_threshold,
            )

            soft_foc_masks.append(soft_foc)
            soft_spa_masks.append(soft_spa)
            hard_foc_masks.append(hard_foc)
            hard_spa_masks.append(hard_spa)
            offset += length

        soft_foc_mask = torch.cat(soft_foc_masks, dim=0)
        soft_spa_mask = torch.cat(soft_spa_masks, dim=0)
        hard_foc_mask = torch.cat(hard_foc_masks, dim=0)
        hard_spa_mask = torch.cat(hard_spa_masks, dim=0)
        img_mask = torch.clamp(soft_foc_mask + soft_spa_mask, 0.0, 1.0)
        constraint_img_mask = torch.clamp(hard_foc_mask + hard_spa_mask, 0.0, 1.0)

        if debug_enabled:
            rank_ok = True
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank_ok = torch.distributed.get_rank() == 0
            if rank_ok:
                msg = (
                    "[BBoxBudget] used=True "
                    f"imgs={len(token_lengths)} "
                    f"ref_used={ref_used} "
                    f"k_tokens={k_tokens_per_image} "
                    f"k_foc={k_foc_per_image} "
                    f"k_spa={k_spa_per_image} "
                    f"k_ratio_min={float(min(k_ratio_per_image) if k_ratio_per_image else 0.0):.4f} "
                    f"k_ratio_max={float(max(k_ratio_per_image) if k_ratio_per_image else 0.0):.4f} "
                    f"cos_w={cosine_penalty_weight:.3f} cos_thr={cosine_similarity_threshold:.2f} "
                    f"min/max={min_keep_ratio:.2f}/{max_keep_ratio:.2f} "
                    f"reason={debug_reason} "
                    f"pre={debug_pre}"
                )
                logger.info(msg)
                print(msg)
    else:
        # 【回退策略】使用原版本的固定全局预算方法
        # 与原版本相同的逻辑，但仅在BBox不可用时使用
        k = int(total_tokens * self.budgets)
        k = max(1, k) # 确保至少保留1个Token
        grid_entry = None
        if isinstance(grid_thw, torch.Tensor) and grid_thw.numel() > 0:
            if grid_thw.dim() == 1 and grid_thw.numel() >= 3:
                grid_entry = grid_thw
            elif grid_thw.dim() > 1 and grid_thw.shape[0] == 1:
                grid_entry = grid_thw[0]

        soft_foc_mask, soft_spa_mask, hard_foc_mask, hard_spa_mask = _compute_foc_spa_masks(
            learned_scores,
            k,
            grid_entry,
            merge_size,
            spatial_ratio,
            token_features_slice=hidden_states,
            cosine_penalty_weight=cosine_penalty_weight,
            cosine_similarity_threshold=cosine_similarity_threshold,
        )
        img_mask = torch.clamp(soft_foc_mask + soft_spa_mask, 0.0, 1.0)
        constraint_img_mask = torch.clamp(hard_foc_mask + hard_spa_mask, 0.0, 1.0)
        if debug_enabled:
            k_foc_dbg, k_spa_dbg = _split_foc_spa_budget(k, spatial_ratio)
            k_foc_dbg = max(1, min(k_foc_dbg, total_tokens - 1))
            k_spa_dbg = max(0, min(k_spa_dbg, total_tokens - k_foc_dbg))
            rank_ok = True
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank_ok = torch.distributed.get_rank() == 0
            if rank_ok:
                msg = (
                    "[BBoxBudget] used=False "
                    f"total_tokens={total_tokens} "
                    f"k={k} k_foc={k_foc_dbg} k_spa={k_spa_dbg} budget={float(self.budgets):.3f} "
                    f"cos_w={cosine_penalty_weight:.3f} cos_thr={cosine_similarity_threshold:.2f} "
                    f"reason={debug_reason} "
                    f"pre={debug_pre}"
                )
                logger.info(msg)
                print(msg)

    img_mask_expanded = img_mask.unsqueeze(1).expand(-1, hidden_states_unsqueezed.shape[-1])
    hidden_states_new = img_mask_expanded*hidden_states
    hidden_states_new = hidden_states_new.type(hidden_states.dtype)

    # -------------------------------------------------------------------------------
    # 【修改】返回值增加 learned_scores 与 foc/spa masks 用于监督
    return (
        hidden_states_new,
        img_mask,
        constraint_img_mask,
        learned_scores,
        soft_foc_mask,
        soft_spa_mask,
        hard_foc_mask,
        hard_spa_mask,
    )


def _flatten_ref_token_mask(
    mask: torch.Tensor,
    token_len: int,
) -> Optional[torch.Tensor]:
    """
    Normalize ref_token_masks to a flat 1D mask aligned with the vision token order.

    Expected input mask shapes:
      - [H, W] (GlimpsePrune-style token grid after merge)
      - [T, H, W]
      - [num_tokens]

    If token_len is a multiple of H*W, we repeat the spatial mask across T frames.
    """
    if mask is None or not isinstance(mask, torch.Tensor):
        return None
    if token_len <= 0:
        return None

    if mask.dim() == 1:
        flat = mask
    elif mask.dim() in (2, 3):
        flat = mask.reshape(-1)
    else:
        return None

    if flat.numel() == token_len:
        return flat

    if flat.numel() > 0 and token_len % flat.numel() == 0:
        repeat = token_len // flat.numel()
        expanded = flat.repeat(repeat)
        if expanded.numel() == token_len:
            return expanded

    return None


def _compute_mask_token_supervision_loss_impl(
    learned_scores: torch.Tensor,
    mask_token_labels: List[Optional[torch.Tensor]],
    mask_loss_valid: List[bool],
    grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    rank_margin: float = 0.1,
) -> Optional[torch.Tensor]:
    """
    Compute the original full pairwise ranking loss from three-value Mask labels.

    Labels are token-aligned after spatial merge: 1=positive, 0=reliable negative,
    -1=ignore. Each image is averaged first, then the image losses are averaged.
    """
    if mask_token_labels is None or len(mask_token_labels) == 0:
        return None
    if grid_thw is None or not isinstance(grid_thw, torch.Tensor) or grid_thw.numel() == 0:
        return None

    token_lengths = _grid_thw_to_token_lengths(grid_thw, spatial_merge_size)
    if not token_lengths:
        return None

    # Split scores by image based on token_lengths
    offset = 0
    per_image_losses: List[torch.Tensor] = []

    for img_idx, token_len in enumerate(token_lengths):
        if offset + token_len > learned_scores.shape[0]:
            break
        scores = learned_scores[offset : offset + token_len]
        offset += token_len

        labels = mask_token_labels[img_idx]
        is_valid = img_idx < len(mask_loss_valid) and bool(mask_loss_valid[img_idx])
        if not is_valid:
            per_image_losses.append(scores.sum() * 0.0)
            continue

        flat_labels = _flatten_ref_token_mask(labels, token_len)
        if flat_labels is None or flat_labels.numel() != scores.numel():
            raise ValueError(
                f"Mask label length mismatch for image {img_idx}: "
                f"labels={None if flat_labels is None else flat_labels.numel()} scores={scores.numel()}"
            )

        loss = compute_bbox_rank_loss_full(scores, flat_labels, margin=rank_margin)
        if loss is None:
            raise ValueError(f"Valid Mask supervision has no positive or reliable negative tokens for image {img_idx}")
        per_image_losses.append(loss)

    if not per_image_losses:
        return None

    return torch.stack(per_image_losses).mean()


def qwen25vl_generation_forward_selector(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    # 【新增】Token-level mask监督参数
    ref_token_masks: Optional[List[Optional[torch.Tensor]]] = None,
    mask_token_labels: Optional[List[Optional[torch.Tensor]]] = None,
    mask_loss_valid: Optional[List[bool]] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            # 【调用增强的视觉模型】传入BBox信息
            # 与原版本不同：这里支持BBox监督的Token选择
            (
                image_embeds,
                img_mask,
                constraint_img_mask,
                learned_scores,
                soft_foc_mask,
                soft_spa_mask,
                hard_foc_mask,
                hard_spa_mask,
            ) = self.visual(
                pixel_values,
                grid_thw=image_grid_thw,
                ref_token_masks=ref_token_masks,
            )
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds, _, _, _, _, _, _, _ = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            # Only move to device if we might use it, and handle potentially weird shapes downstream
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        ce_loss_val = loss.item()

        # --------------------------------约束+Mask监督损失计算---------------------------------------------
        constraint_val = None
        mask_loss_val = None

        if pixel_values is not None:
            # 约束项：focal/spatial 软硬掩码分别对齐
            constraint_loss = (
                F.binary_cross_entropy(soft_foc_mask, hard_foc_mask)
                + F.binary_cross_entropy(soft_spa_mask, hard_spa_mask)
            )
            loss += self.regularization_weight * constraint_loss
            constraint_val = constraint_loss.item()

            mask_loss = None
            if mask_token_labels is not None and mask_loss_valid is not None and image_grid_thw is not None:
                mask_loss = _compute_mask_token_supervision_loss_impl(
                    learned_scores,
                    mask_token_labels,
                    mask_loss_valid,
                    image_grid_thw,
                    spatial_merge_size=getattr(self.visual, 'spatial_merge_size', 2),
                    rank_margin=getattr(self, 'bbox_rank_margin', 0.2),
                )

            if mask_loss is not None:
                mask_loss_weight = getattr(self, 'bbox_loss_weight', 1.0)
                mask_loss_val = mask_loss.item()
                loss += mask_loss_weight * mask_loss

            # 调试输出：统一打印 CE/constraint/mask
            constraint_str = f"{constraint_val:.4f}" if constraint_val is not None else "None"
            mask_str = f"{mask_loss_val:.4f}" if mask_loss_val is not None else "None"
            logger.info(
                "[LOSS_DEBUG] ce_loss=%.4f constraint=%s (w=%.3f) mask_loss=%s (w=%.3f) total=%.4f",
                ce_loss_val,
                constraint_str,
                self.regularization_weight,
                mask_str,
                getattr(self, 'bbox_loss_weight', 1.0),
                loss.item(),
            )
            print(
                f"[LOSS_DEBUG] ce_loss={ce_loss_val:.4f} "
                f"constraint={constraint_str} (w={self.regularization_weight:.3f}) "
                f"mask_loss={mask_str} (w={getattr(self, 'bbox_loss_weight', 1.0):.3f}) "
                f"total={loss.item():.4f}"
            )
            # 缓存本步损失拆分，供 Trainer 回调写入 log_history
            self._last_loss_components = {
                "ce_loss": ce_loss_val,
                "constraint_loss": constraint_val,
                "mask_loss": mask_loss_val,
                "total_loss": loss.item(),
            }
        else:
            logger.info("[LOSS_DEBUG] ce_loss=%.4f constraint=None mask_loss=None total=%.4f", ce_loss_val, loss.item())
            print(f"[LOSS_DEBUG] ce_loss={ce_loss_val:.4f} constraint=None mask_loss=None total={loss.item():.4f}")
            self._last_loss_components = {
                "ce_loss": ce_loss_val,
                "constraint_loss": None,
                "mask_loss": None,
                "total_loss": loss.item(),
            }
        # -------------------------------------------------------------------------------

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
