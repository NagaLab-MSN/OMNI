import torch

def average_precision_at_k(predictions: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """
    Computes Average Precision at K for a ranked list of predictions.
    """
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