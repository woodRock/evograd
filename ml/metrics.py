"""Evaluation metrics on EvoGrad Tensors."""
from __future__ import annotations
import torch
from ..core.tensor import Tensor


def accuracy(pred: Tensor, target: Tensor) -> float:
    return (pred.data == target.data).float().mean().item()


def mse(pred: Tensor, target: Tensor) -> float:
    return ((pred.data - target.data) ** 2).mean().item()


def mae(pred: Tensor, target: Tensor) -> float:
    return (pred.data - target.data).abs().mean().item()


def r2_score(pred: Tensor, target: Tensor) -> float:
    ss_res = ((target.data - pred.data) ** 2).sum()
    ss_tot = ((target.data - target.data.mean()) ** 2).sum()
    return 1.0 - (ss_res / ss_tot).item()


def confusion_matrix(pred: Tensor, target: Tensor,
                     n_classes: int) -> Tensor:
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long,
                     device=pred.device)
    for t, p in zip(target.data.long(), pred.data.long()):
        cm[t, p] += 1
    return Tensor(cm)


def precision_recall_f1(pred: Tensor, target: Tensor,
                         n_classes: int) -> dict:
    cm = confusion_matrix(pred, target, n_classes).data.float()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    precision = (tp / (tp + fp + 1e-8)).mean().item()
    recall = (tp / (tp + fn + 1e-8)).mean().item()
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}
