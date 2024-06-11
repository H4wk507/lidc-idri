import torch


def recall(mask: torch.Tensor, pred: torch.Tensor, smooth: float = 1e-10) -> torch.Tensor:
    # assuming mask and pred are binary

    mask_flat = mask.view(mask.size(0), -1)
    pred_flat = pred.view(pred.size(0), -1)

    tp = torch.sum((mask_flat == 1) & (pred_flat == 1), dim=1)
    fn = torch.sum((pred_flat == 0) & (mask_flat == 1), dim=1)
    return tp / (tp + fn + smooth)


def precision(mask: torch.Tensor, pred: torch.Tensor, smooth: float = 1e-10) -> torch.Tensor:
    # assuming mask and pred are binary

    mask_flat = mask.view(mask.size(0), -1)
    pred_flat = pred.view(pred.size(0), -1)

    tp = torch.sum((mask_flat == 1) & (pred_flat == 1), dim=1)
    fp = torch.sum((pred_flat == 1) & (mask_flat == 0), dim=1)
    return tp / (tp + fp + smooth)

