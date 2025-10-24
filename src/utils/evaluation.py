import torch
import numpy as np
from collections import defaultdict

def monitor_gradients(model):
    """monitor the gradients of the model parameters"""

    total_norm = 0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count > 0:
        total_norm = total_norm ** (1. / 2)
        return total_norm
    else:
        return 0.0

@torch.no_grad()
def compute_ranking_metrics(
                        model,
                        base_data,
                        edge_index_dict,
                        k_list,
                        all_true_tails,
                        edge_type
                    ):
    """
    Compute ranking metrics (MRR, Hits@K, Precision@K, Recall@K, NDCG@K) for link prediction.

    Args:
      - edge_type: A triplet (src, rel, dst), e.g., ('drug', 'treats', 'disease')
    """

    model.eval()
    z_dict = model(base_data.x_dict, edge_index_dict)

    head_emb = z_dict[edge_type[0]]
    tail_emb = z_dict[edge_type[2]]

    if not hasattr(model, "relation_embedding") or edge_type[1] not in model.relation_embedding:
        raise ValueError(f"Model lacks relationship vectors: {edge_type[1]}")
    relation_emb = model.relation_embedding[edge_type[1]]

    edge_label_index = base_data[edge_type].edge_label_index
    edge_label = base_data[edge_type].edge_label
    pos_edges = edge_label_index[:, edge_label == 1]
    true_tails_eval = defaultdict(set)

    for h, t in pos_edges.t().tolist():
        true_tails_eval[h].add(t)

    if not true_tails_eval:
        return {'MRR': 0.0, **{f'Hits@{k}': 0.0 for k in k_list},
                **{f'Precision@{k}': 0.0 for k in k_list},
                **{f'Recall@{k}': 0.0 for k in k_list},
                **{f'NDCG@{k}': 0.0 for k in k_list}}

    ranks = []
    hits_at = {k: 0 for k in k_list}
    precisions = {k: [] for k in k_list}
    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}

    for head, eval_tails in true_tails_eval.items():
        head_embed = head_emb[head:head+1]
        head_rel = head_embed * relation_emb
        scores = torch.matmul(head_rel, tail_emb.t()).squeeze(0).clone()  # [num_disease]

        other_true = all_true_tails.get(head, set()) - eval_tails
        if other_true:
            idx_tensor = torch.tensor(list(other_true), dtype=torch.long, device=scores.device)
            scores[idx_tensor] = -1e9  
        scores_np = scores.detach().cpu().numpy()
        sorted_indices = np.argsort(-scores_np)
        index_position = {idx: pos for pos, idx in enumerate(sorted_indices, start=1)}
        
        for tail in eval_tails:
            ranks.append(index_position[tail])
        for K in k_list:
            hits_k = sum(1 for tail in eval_tails if index_position[tail] <= K)
            if hits_k > 0:
                hits_at[K] += 1
            precisions[K].append(hits_k / K)
            recalls[K].append(hits_k / len(eval_tails))
            # NDCG@K
            dcg = 0.0
            idcg = 0.0
            for tail in eval_tails:
                r = index_position[tail]
                if r <= K:
                    dcg += 1.0 / np.log2(r + 1)
            for i in range(1, min(len(eval_tails), K) + 1):
                idcg += 1.0 / np.log2(i + 1)
            ndcgs[K].append(dcg / idcg if idcg > 0 else 0.0)

    if len(ranks) == 0:
        return {'MRR': 0.0, 
                **{f'Hits@{k}': 0.0 for k in k_list},
                **{f'Precision@{k}': 0.0 for k in k_list},
                **{f'Recall@{k}': 0.0 for k in k_list},
                **{f'NDCG@{k}': 0.0 for k in k_list}}

    ranks_arr = np.array(ranks)
    metrics = {'MRR': float(np.mean(1.0 / ranks_arr))}
    num_heads = len(true_tails_eval)
    for K in k_list:
        metrics[f'Hits@{K}'] = hits_at[K] / num_heads if num_heads > 0 else 0.0
        metrics[f'Precision@{K}'] = float(np.mean(precisions[K])) if precisions[K] else 0.0
        metrics[f'Recall@{K}'] = float(np.mean(recalls[K])) if recalls[K] else 0.0
        metrics[f'NDCG@{K}'] = float(np.mean(ndcgs[K])) if ndcgs[K] else 0.0
    return metrics


