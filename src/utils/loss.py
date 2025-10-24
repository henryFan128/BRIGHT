import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

def loss_bce(scores, edge_label, neg_sampling_ratio: float = 1.0):
    """
    Binary Cross-Entropy Loss with Negative Sampling

    Args:
        scores (Tensor): Predicted scores for positive and negative samples.
        edge_label (Tensor): Ground truth labels (1 for positive, 0 for negative).
        neg_sampling_ratio (float): Ratio of negative to positive samples.

    Returns:
        Tensor: Computed BCE loss.
    """

    pos_weight = torch.tensor([neg_sampling_ratio], device=scores.device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(scores, edge_label.float())


def loss_bpr(scores, edge_label, neg_sampling_ratio: float = 1.0, all_pairs: bool = False):
    """
    Bayesian Personalized Ranking (BPR) Loss

    Args:
        scores (Tensor): Predicted scores for positive and negative samples.
        edge_label (Tensor): Ground truth labels (1 for positive, 0 for negative).
        neg_sampling_ratio (float): Ratio of negative to positive samples.
        all_pairs (bool): If True, compute loss over all positive-negative pairs.

    Returns:
        Tensor: Computed BPR loss.
    """

    pos_mask = (edge_label == 1)
    neg_mask = (edge_label == 0)
    
    if not pos_mask.any() or not neg_mask.any():
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    
    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    
    if all_pairs:
        # Use all positive-negative pairs
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        loss = -F.logsigmoid(diff).mean()
        return loss

    num_pos = pos_scores.size(0)
    num_neg = neg_scores.size(0)
    if num_neg == 0:
        return torch.zeros((), device=scores.device, dtype=scores.dtype)
    
    neg_idx = torch.randint(low=0, high=num_neg, size=(num_pos, max(1, bpr_neg_per_pos)), device=scores.device)
    sampled_neg = neg_scores[neg_idx] 
    pos_expand = pos_scores.unsqueeze(1).expand_as(sampled_neg)
    diff = pos_expand - sampled_neg
    loss = -F.logsigmoid(diff).mean()
    return loss

def combined_loss(
                scores,
                edge_label,
                neg_sampling_ratio: float = 1.0,
                bce_weight: float = 1.0,
                bpr_weight: float = 0.0,
                bpr_neg_per_pos: int = 1,
                bpr_all_pairs: bool = False
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    """
    Combined Loss: Weighted sum of BCE and BPR losses

    Args:
        scores (Tensor): Predicted scores for positive and negative samples.
        edge_label (Tensor): Ground truth labels (1 for positive, 0 for negative).
        neg_sampling_ratio (float): Ratio of negative to positive samples.
        all_pairs (bool): If True, compute BPR loss over all positive-negative pairs.

    Returns:
        Tensor: Computed combined loss.
    """
    total = torch.zeros((), device=scores.device, dtype=scores.dtype)
    components: Dict[str, torch.Tensor] = {}
    
    if bce_weight > 0:
        bce_loss = loss_bce(scores, edge_label, neg_sampling_ratio)
        total = total + bce_weight * bce_loss
        components['bce'] = bce_loss.detach()
        
    if bpr_weight > 0:
        bpr_loss = loss_bpr(
            scores, edge_label,
            bpr_neg_per_pos=bpr_neg_per_pos,
            all_pairs=bpr_all_pairs
        )
        total = total + bpr_weight * bpr_loss
        components['bpr'] = bpr_loss.detach()
        
    return total, components