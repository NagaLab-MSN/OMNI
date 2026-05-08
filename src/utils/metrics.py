import torch

def average_precision_at_k(predictions: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    sorted_indices = torch.argsort(predictions, descending=True)
    sorted_labels = labels[sorted_indices]
    top_k_labels = sorted_labels[:k]
    relevant_indices = torch.where(top_k_labels == 1)[0]
    if relevant_indices.numel() == 0:
        return 0.0
    precisions = [
        (i + 1) / (pos.item() + 1)
        for i, pos in enumerate(relevant_indices)
    ]
    return sum(precisions) / relevant_indices.numel() if precisions else 0.0


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    
    return ((probs - labels.float()) ** 2).mean().item()


def expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10
) -> float:
    
    probs = probs.detach().cpu().float()
    labels = labels.detach().cpu().float()
    n = probs.numel()
    if n == 0:
        return 0.0

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
           
            mask = (probs >= lo) & (probs <= hi)
        bin_size = mask.sum().item()
        if bin_size == 0:
            continue
        avg_confidence = probs[mask].mean().item()
        avg_accuracy = labels[mask].mean().item()
        ece += (bin_size / n) * abs(avg_accuracy - avg_confidence)
    return ece


def compute_ranking_metrics(
    pos_scores_list: List[torch.Tensor],
    neg_scores_list: List[torch.Tensor],
    neg_k: int,
) -> Dict[str, float]:
    if not pos_scores_list or not neg_scores_list:
        return {'mrr': 0.0, 'hits@1': 0.0, 'hits@3': 0.0, 'hits@10': 0.0}

    all_pos = torch.cat(pos_scores_list)         
    all_neg = torch.cat(neg_scores_list)          
    n = all_pos.numel()
    if n == 0:
        return {'mrr': 0.0, 'hits@1': 0.0, 'hits@3': 0.0, 'hits@10': 0.0}

    
    neg_matrix = all_neg.view(n, neg_k)

    
    candidates = torch.cat([all_pos.unsqueeze(1), neg_matrix], dim=1)

  
    desc_order = torch.argsort(candidates, dim=1, descending=True)
    ranks_matrix = torch.argsort(desc_order, dim=1)  
    pos_rank = ranks_matrix[:, 0].float() + 1.0     

    mrr = (1.0 / pos_rank).mean().item()
    hits_1 = (pos_rank <= 1).float().mean().item()
    hits_3 = (pos_rank <= 3).float().mean().item()
    hits_10 = (pos_rank <= 10).float().mean().item()

    return {'mrr': mrr, 'hits@1': hits_1, 'hits@3': hits_3, 'hits@10': hits_10}