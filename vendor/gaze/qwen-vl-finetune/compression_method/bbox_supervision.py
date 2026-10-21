# -*- coding: utf-8 -*-
"""
BBox Supervision Utilities for Vision Token Selection

This module provides functions to:
1. Rasterize normalized BBox coordinates to token-level binary masks
2. Compute BCE + Dice loss for foreground/background classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


def _viscot_square_norm_to_hw_norm(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    h_tokens: int,
    w_tokens: int,
) -> Tuple[float, float, float, float]:
    """
    VISCOT conversation bboxes are normalized after padding the image to a square
    canvas with side `s = max(w, h)` and centered padding.

    This converts square-normalized coords back to (w, h)-normalized coords using
    only the aspect ratio.
    """
    if h_tokens <= 0 or w_tokens <= 0:
        return x1, y1, x2, y2

    aspect = float(w_tokens) / float(h_tokens)  # w/h
    if aspect < 1.0:
        # Portrait: pad left/right.
        pad_x = (1.0 - aspect) / 2.0
        x1 = (x1 - pad_x) / aspect
        x2 = (x2 - pad_x) / aspect
    else:
        # Landscape (or square): pad top/bottom.
        inv_aspect = 1.0 / aspect
        pad_y = (1.0 - inv_aspect) / 2.0
        y1 = (y1 - pad_y) * aspect
        y2 = (y2 - pad_y) * aspect

    return x1, y1, x2, y2


def bbox_to_token_mask(
    bbox_normalized: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
    assume_square_padded: bool = True,
) -> torch.Tensor:
    """
    Convert normalized bounding box coordinates to a binary token-level mask.

    The viscot dataset provides normalized bbox in [x1, y1, x2, y2] format where
    coordinates are in [0, 1] range. In VISCOT, conversation bboxes are normalized
    in a square-padded coordinate system (pad to `s=max(w,h)` then normalize by `s`).
    This function optionally converts to (w,h)-normalized coords using the token grid
    aspect ratio before rasterizing to the token grid after spatial merge.

    Args:
        bbox_normalized: Normalized bbox [x1, y1, x2, y2] in range [0, 1]
                        Shape: [4] or [N, 4]
        grid_thw: Grid dimensions [T, H, W] before spatial merge
                  Shape: [3] or [N, 3]
        spatial_merge_size: Spatial merge factor (default 2 for Qwen2.5-VL)
        assume_square_padded: If True, treat bbox as VISCOT square-padded normalized
                              coords and convert to (w,h)-normalized coords.

    Returns:
        Binary mask of shape [num_tokens] where 1 indicates foreground (inside bbox)
        For multiple bboxes, returns shape [N, num_tokens]

    Example:
        >>> bbox = torch.tensor([0.3, 0.2, 0.7, 0.8])  # normalized coords
        >>> grid_thw = torch.tensor([1, 28, 28])  # 1 frame, 28x28 patches
        >>> mask = bbox_to_token_mask(bbox, grid_thw, spatial_merge_size=2)
        >>> # Returns mask of shape [196] (14x14 tokens after merge)
    """
    # Handle single bbox case
    if bbox_normalized.dim() == 1:
        bbox_normalized = bbox_normalized.unsqueeze(0)
        grid_thw = grid_thw.unsqueeze(0) if grid_thw.dim() == 1 else grid_thw
        single_bbox = True
    else:
        single_bbox = False

    device = bbox_normalized.device
    dtype = bbox_normalized.dtype

    masks = []

    for i in range(bbox_normalized.shape[0]):
        bbox = bbox_normalized[i]  # [x1, y1, x2, y2]

        # Get grid dimensions
        if grid_thw.shape[0] > i:
            t, h, w = grid_thw[i].long().tolist()
        else:
            # Use first grid_thw for all bboxes if only one provided
            t, h, w = grid_thw[0].long().tolist()

        # Token grid size after spatial merge
        h_tokens = h // spatial_merge_size
        w_tokens = w // spatial_merge_size
        num_tokens = t * h_tokens * w_tokens

        # Convert normalized coords to token grid coords
        x1, y1, x2, y2 = bbox.tolist()

        # VISCOT conversation bboxes are normalized after square padding; undo it.
        if assume_square_padded:
            x1, y1, x2, y2 = _viscot_square_norm_to_hw_norm(
                x1, y1, x2, y2, h_tokens=h_tokens, w_tokens=w_tokens
            )

        # Clamp to valid range
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))

        # Convert to token indices
        col_start = int(x1 * w_tokens)
        col_end = int(x2 * w_tokens)
        row_start = int(y1 * h_tokens)
        row_end = int(y2 * h_tokens)

        # Ensure at least one token is selected
        col_end = max(col_start + 1, col_end)
        row_end = max(row_start + 1, row_end)

        # Clamp to valid range
        col_start = max(0, min(w_tokens - 1, col_start))
        col_end = max(1, min(w_tokens, col_end))
        row_start = max(0, min(h_tokens - 1, row_start))
        row_end = max(1, min(h_tokens, row_end))

        # Create 2D mask for each temporal frame
        mask_2d = torch.zeros((h_tokens, w_tokens), device=device, dtype=dtype)
        mask_2d[row_start:row_end, col_start:col_end] = 1.0

        # Expand to all temporal frames and flatten
        mask_3d = mask_2d.unsqueeze(0).expand(t, -1, -1)  # [T, H, W]
        mask_flat = mask_3d.reshape(-1)  # [num_tokens]

        masks.append(mask_flat)

    # Stack masks
    result = torch.stack(masks, dim=0) if len(masks) > 1 else masks[0].unsqueeze(0)

    if single_bbox:
        return result.squeeze(0)
    return result


def compute_bbox_supervision_loss(
    scores: torch.Tensor,
    gt_mask: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    smooth: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute BCE + Dice loss for bbox supervision.

    Args:
        scores: Predicted importance scores from Scorer, shape [num_tokens]
                Should be raw scores (will be normalized to [0, 1] via sigmoid)
        gt_mask: Ground truth binary mask from bbox, shape [num_tokens]
                 1 = foreground (inside bbox), 0 = background
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        smooth: Smoothing factor for Dice loss to avoid division by zero

    Returns:
        Tuple of (total_loss, bce_loss, dice_loss)

    Loss details:
        - BCE Loss: Pixel-level binary cross entropy
          $$L_{BCE} = -\\frac{1}{N} \\sum_{i} [y_i \\log(p_i) + (1-y_i)\\log(1-p_i)]$$

        - Dice Loss: Region-level similarity
          $$L_{Dice} = 1 - \\frac{2 \\sum_i p_i y_i + \\epsilon}{\\sum_i p_i + \\sum_i y_i + \\epsilon}$$
    """
    # Normalize scores to [0, 1] probability
    probs = torch.sigmoid(scores)

    # Ensure same device and dtype
    gt_mask = gt_mask.to(probs.device, dtype=probs.dtype)

    # BCE Loss (pixel-level)
    bce_loss = F.binary_cross_entropy(probs, gt_mask, reduction='mean')

    # Dice Loss (region-level) - handles class imbalance
    intersection = (probs * gt_mask).sum()
    union = probs.sum() + gt_mask.sum()
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_score

    # Total loss
    total_loss = bce_weight * bce_loss + dice_weight * dice_loss

    return total_loss, bce_loss, dice_loss


def compute_bbox_rank_loss_full(
    scores: torch.Tensor,
    gt_mask: torch.Tensor,
    margin: float = 0.2,
) -> Optional[torch.Tensor]:
    """
    Full pairwise ranking loss for three-value Mask labels.

    Label 1 is foreground, label 0 is reliable background, and label -1 is ignored.

    L = -mean(logsigmoid(s_pos - s_neg - margin))
      = mean(softplus(-(s_pos - s_neg - margin)))
    """
    if scores is None or gt_mask is None:
        return None

    gt_mask = gt_mask.to(scores.device)

    pos_scores = scores[gt_mask == 1]
    neg_scores = scores[gt_mask == 0]

    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return None

    # Full pairwise differences: [P, N]
    diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0) - margin
    loss = F.softplus(-diff).mean()
    return loss


def aggregate_multi_image_bbox_loss(
    scores_list: List[torch.Tensor],
    bbox_list: List[Optional[List[float]]],
    grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> Tuple[Optional[torch.Tensor], dict]:
    """
    Compute aggregated bbox supervision loss for multiple images in a batch.

    In viscot dataset, each sample has 2 images:
    - Image 0: Full image (no bbox in path, but conversation has bbox response)
    - Image 1: Cropped image (bbox in path, but we don't use this for supervision)

    We use the normalized bbox from conversation (GPT's response) to supervise
    the token selection on the FIRST image (full image).

    Args:
        scores_list: List of score tensors, one per image
                     Each tensor shape: [num_tokens_for_image_i]
        bbox_list: List of normalized bbox coordinates [x1, y1, x2, y2] or None
                   From conversation_bboxes in batch
        grid_thw: Grid dimensions for all images, shape [num_images, 3]
        spatial_merge_size: Spatial merge factor
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss

    Returns:
        Tuple of (aggregated_loss, metrics_dict)
        - aggregated_loss: Mean loss over all valid bbox-image pairs, or None if no valid pairs
        - metrics_dict: Contains 'bce_loss', 'dice_loss', 'num_valid_pairs', 'foreground_ratio'
    """
    total_bce = 0.0
    total_dice = 0.0
    total_foreground_ratio = 0.0
    num_valid = 0

    # Track which images have valid bbox supervision
    for img_idx, (scores, bbox) in enumerate(zip(scores_list, bbox_list)):
        if bbox is None:
            continue

        # Get grid_thw for this image
        if img_idx >= grid_thw.shape[0]:
            continue
        img_grid_thw = grid_thw[img_idx]

        # Convert bbox to tensor
        bbox_tensor = torch.tensor(bbox, device=scores.device, dtype=scores.dtype)

        # Generate ground truth mask
        gt_mask = bbox_to_token_mask(bbox_tensor, img_grid_thw, spatial_merge_size)

        # Ensure shapes match
        if gt_mask.shape[0] != scores.shape[0]:
            # Shape mismatch - skip this pair
            continue

        # Compute loss
        _, bce_loss, dice_loss = compute_bbox_supervision_loss(
            scores, gt_mask, bce_weight, dice_weight
        )

        total_bce += bce_loss.item()
        total_dice += dice_loss.item()
        total_foreground_ratio += gt_mask.mean().item()
        num_valid += 1

    if num_valid == 0:
        return None, {
            'bce_loss': 0.0,
            'dice_loss': 0.0,
            'num_valid_pairs': 0,
            'foreground_ratio': 0.0,
        }

    # Compute mean losses
    mean_bce = total_bce / num_valid
    mean_dice = total_dice / num_valid
    mean_fg_ratio = total_foreground_ratio / num_valid

    # Re-compute with gradient for backward pass
    total_loss = torch.tensor(0.0, device=scores_list[0].device, requires_grad=True)
    for img_idx, (scores, bbox) in enumerate(zip(scores_list, bbox_list)):
        if bbox is None:
            continue
        if img_idx >= grid_thw.shape[0]:
            continue

        img_grid_thw = grid_thw[img_idx]
        bbox_tensor = torch.tensor(bbox, device=scores.device, dtype=scores.dtype)
        gt_mask = bbox_to_token_mask(bbox_tensor, img_grid_thw, spatial_merge_size)

        if gt_mask.shape[0] != scores.shape[0]:
            continue

        loss, _, _ = compute_bbox_supervision_loss(
            scores, gt_mask, bce_weight, dice_weight
        )
        total_loss = total_loss + loss

    aggregated_loss = total_loss / num_valid

    metrics = {
        'bce_loss': mean_bce,
        'dice_loss': mean_dice,
        'num_valid_pairs': num_valid,
        'foreground_ratio': mean_fg_ratio,
    }

    return aggregated_loss, metrics


class BBoxSupervisionModule(nn.Module):
    """
    Module for computing bbox supervision loss during training.

    This module is attached to the model and computes BCE + Dice loss
    to train the Scorer to distinguish foreground (inside bbox) from
    background (outside bbox) tokens.

    Usage:
        bbox_module = BBoxSupervisionModule(bce_weight=1.0, dice_weight=1.0)
        loss, metrics = bbox_module(scores, conv_bboxes, grid_thw)
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.spatial_merge_size = spatial_merge_size

    def forward(
        self,
        scores: torch.Tensor,
        conversation_bboxes: List[List[Optional[List[float]]]],
        grid_thw: torch.Tensor,
        images_per_sample: int = 2,
    ) -> Tuple[Optional[torch.Tensor], dict]:
        """
        Compute bbox supervision loss.

        Args:
            scores: Token importance scores from Scorer
                    Shape: [total_num_tokens] (flattened across all images)
            conversation_bboxes: Nested list of bbox from conversations
                                 Structure: batch[sample_idx][turn_idx] -> bbox or None
                                 We look for bbox in the second turn (GPT's bbox response)
            grid_thw: Grid dimensions for all images
                     Shape: [num_images, 3]
            images_per_sample: Number of images per sample (default 2 for viscot)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Extract the bbox from conversation (typically in the second turn, index 1)
        # In viscot format: turn 0 = human question, turn 1 = GPT bbox response
        #
        # IMPORTANT: Expand bbox_list to match image count
        # Each sample has `images_per_sample` images, but only the FIRST image gets bbox supervision
        # bbox_list must have same length as scores_list (one entry per image)
        bbox_list = []

        for sample_bboxes in conversation_bboxes:
            if sample_bboxes is None:
                # No bbox for this sample - add None for all its images
                bbox_list.extend([None] * images_per_sample)
                continue

            # Look for bbox in conversation turns
            sample_bbox = None
            for turn_bbox in sample_bboxes:
                if turn_bbox is not None:
                    sample_bbox = turn_bbox
                    break

            # First image gets bbox, rest get None
            bbox_list.append(sample_bbox)
            bbox_list.extend([None] * (images_per_sample - 1))

        # Split scores by image based on grid_thw
        scores_list = self._split_scores_by_image(scores, grid_thw)

        # Compute aggregated loss
        return aggregate_multi_image_bbox_loss(
            scores_list,
            bbox_list,
            grid_thw,
            self.spatial_merge_size,
            self.bce_weight,
            self.dice_weight,
        )

    def _split_scores_by_image(
        self,
        scores: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Split flattened scores tensor by image."""
        scores_list = []
        offset = 0

        for i in range(grid_thw.shape[0]):
            t, h, w = grid_thw[i].long().tolist()
            h_tokens = h // self.spatial_merge_size
            w_tokens = w // self.spatial_merge_size
            num_tokens = t * h_tokens * w_tokens

            if offset + num_tokens <= scores.shape[0]:
                scores_list.append(scores[offset:offset + num_tokens])
                offset += num_tokens
            else:
                # Handle edge case
                scores_list.append(scores[offset:])
                break

        return scores_list


if __name__ == '__main__':
    # Test bbox_to_token_mask
    print("Testing bbox_to_token_mask...")
    bbox = torch.tensor([0.3, 0.2, 0.7, 0.8])  # x1, y1, x2, y2
    grid_thw = torch.tensor([1, 28, 28])  # T, H, W
    mask = bbox_to_token_mask(bbox, grid_thw, spatial_merge_size=2)
    print(f"  Input bbox: {bbox.tolist()}")
    print(f"  Grid THW: {grid_thw.tolist()}")
    print(f"  Output mask shape: {mask.shape}")
    print(f"  Foreground ratio: {mask.mean().item():.4f}")
    print(f"  Foreground tokens: {mask.sum().item():.0f} / {mask.numel()}")

    # Visualize mask
    h_tokens, w_tokens = 14, 14
    mask_2d = mask.reshape(h_tokens, w_tokens)
    print(f"  Mask 2D shape: {mask_2d.shape}")

    # Test compute_bbox_supervision_loss
    print("\nTesting compute_bbox_supervision_loss...")
    scores = torch.randn(196)  # Random scores for 14x14 tokens
    total_loss, bce_loss, dice_loss = compute_bbox_supervision_loss(scores, mask)
    print(f"  Random scores - Total: {total_loss.item():.4f}, BCE: {bce_loss.item():.4f}, Dice: {dice_loss.item():.4f}")

    # Test with perfect prediction
    perfect_scores = mask * 10 - 5  # High scores inside bbox, low outside
    total_loss, bce_loss, dice_loss = compute_bbox_supervision_loss(perfect_scores, mask)
    print(f"  Perfect scores - Total: {total_loss.item():.4f}, BCE: {bce_loss.item():.4f}, Dice: {dice_loss.item():.4f}")

    # Test BBoxSupervisionModule
    print("\nTesting BBoxSupervisionModule...")
    module = BBoxSupervisionModule()
    # 2 samples, each with 2 images = 4 images total, 196 tokens per image = 784 tokens
    scores = torch.randn(784)
    conv_bboxes = [
        [None, [0.3, 0.2, 0.7, 0.8], None, None],  # Sample 0: bbox in turn 1
        [None, [0.1, 0.1, 0.5, 0.5], None, None],  # Sample 1: bbox in turn 1
    ]
    # 4 images: sample0-img0, sample0-img1, sample1-img0, sample1-img1
    grid_thw = torch.tensor([[1, 28, 28], [1, 28, 28], [1, 28, 28], [1, 28, 28]])
    loss, metrics = module(scores, conv_bboxes, grid_thw, images_per_sample=2)
    print(f"  Loss: {loss.item() if loss is not None else 'None':.4f}")
    print(f"  Metrics: {metrics}")
    print(f"  Expected num_valid_pairs: 2 (only first image of each sample)")
    assert metrics['num_valid_pairs'] == 2, f"Expected 2, got {metrics['num_valid_pairs']}"

    print("\n✓ All tests passed!")
