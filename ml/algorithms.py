"""
Basic ML algorithms implemented on EvoGrad Tensors.

All algorithms follow a fit(X, y) → self, predict(X) → Tensor
interface.  Internally everything uses the immutable Tensor API; we
call into PyTorch only for BLAS/LAPACK routines that have no clean
closed-form expression.
"""
from __future__ import annotations
import math
from typing import Optional, List
import torch

from ..core.tensor import Tensor
from ..nn import functional as F


# ------------------------------------------------------------------ #
# Linear regression (closed-form normal equations)
# ------------------------------------------------------------------ #

class LinearRegression:
    """
    β = (X^T X)^{-1} X^T y   — solved via pseudo-inverse for stability.
    No iterations, no hyperparameters.
    """

    def __init__(self, fit_intercept: bool = True):
        self._fit_intercept = fit_intercept
        self.weights: Optional[Tensor] = None
        self.bias: Optional[float] = None

    def fit(self, X: Tensor, y: Tensor) -> "LinearRegression":
        x = X.data.double()
        t = y.data.double().reshape(-1)
        if self._fit_intercept:
            ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
            x = torch.cat([x, ones], dim=1)
        # β = pinv(X) y
        beta = torch.linalg.lstsq(x, t).solution
        if self._fit_intercept:
            self.weights = Tensor(beta[:-1].float())
            self.bias = beta[-1].item()
        else:
            self.weights = Tensor(beta.float())
            self.bias = 0.0
        return self

    def predict(self, X: Tensor) -> Tensor:
        out = X @ self.weights + self.bias
        return out


# ------------------------------------------------------------------ #
# Logistic regression (gradient descent)
# ------------------------------------------------------------------ #

class LogisticRegression:
    """
    Binary logistic regression via mini-batch gradient descent.
    Multi-class via one-vs-rest.
    """

    def __init__(self, lr: float = 0.1, epochs: int = 100,
                 l2: float = 1e-4):
        self._lr = lr
        self._epochs = epochs
        self._l2 = l2
        self._w: Optional[torch.Tensor] = None
        self._b: Optional[torch.Tensor] = None

    def fit(self, X: Tensor, y: Tensor) -> "LogisticRegression":
        n, d = X.shape
        n_classes = int(y.data.max().item()) + 1
        device = X.device

        w = torch.zeros(d, n_classes, device=device, requires_grad=True)
        b = torch.zeros(n_classes, device=device, requires_grad=True)
        opt = torch.optim.SGD([w, b], lr=self._lr,
                               weight_decay=self._l2)

        for _ in range(self._epochs):
            logits = X.data @ w + b
            loss = torch.nn.functional.cross_entropy(logits, y.data.long())
            opt.zero_grad()
            loss.backward()
            opt.step()

        self._w = w.detach()
        self._b = b.detach()
        return self

    def predict(self, X: Tensor) -> Tensor:
        logits = X.data @ self._w + self._b
        return Tensor(logits.argmax(dim=1))

    def predict_proba(self, X: Tensor) -> Tensor:
        logits = X.data @ self._w + self._b
        return Tensor(torch.softmax(logits, dim=1))


# ------------------------------------------------------------------ #
# k-Nearest Neighbours (exact, GPU-friendly)
# ------------------------------------------------------------------ #

class KNearestNeighbours:
    """
    Exact KNN via pairwise L2 distance, computed as a single matrix
    op: ||a - b||^2 = ||a||^2 + ||b||^2 - 2a·b^T
    All arithmetic stays on-device.
    """

    def __init__(self, k: int = 5, task: str = "classify"):
        self._k = k
        self._task = task
        self._X_train: Optional[Tensor] = None
        self._y_train: Optional[Tensor] = None

    def fit(self, X: Tensor, y: Tensor) -> "KNearestNeighbours":
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X: Tensor) -> Tensor:
        # Squared pairwise distances: (N_test, N_train)
        A = X.data
        B = self._X_train.data
        dist = (A ** 2).sum(1, keepdim=True) \
             + (B ** 2).sum(1) \
             - 2 * (A @ B.T)
        # k nearest indices
        _, idx = dist.topk(self._k, dim=1, largest=False)
        neighbours = self._y_train.data[idx]   # (N_test, k)

        if self._task == "classify":
            # majority vote — works for arbitrary integer labels
            pred = torch.mode(neighbours.long(), dim=1).values
            return Tensor(pred)
        else:
            return Tensor(neighbours.float().mean(dim=1))


# ------------------------------------------------------------------ #
# Gaussian Naive Bayes
# ------------------------------------------------------------------ #

class GaussianNaiveBayes:
    def __init__(self, var_smoothing: float = 1e-9):
        self._eps = var_smoothing
        self._means: Optional[torch.Tensor] = None
        self._vars: Optional[torch.Tensor] = None
        self._log_priors: Optional[torch.Tensor] = None
        self._classes: Optional[torch.Tensor] = None

    def fit(self, X: Tensor, y: Tensor) -> "GaussianNaiveBayes":
        classes = y.data.unique().long()
        means, variances, log_priors = [], [], []
        n = X.shape[0]
        for c in classes:
            mask = y.data.long() == c
            Xc = X.data[mask]
            means.append(Xc.mean(0))
            variances.append(Xc.var(0) + self._eps)
            log_priors.append(torch.tensor(math.log(mask.sum().item() / n)))
        self._classes = classes
        self._means = torch.stack(means)
        self._vars = torch.stack(variances)
        self._log_priors = torch.stack(log_priors).to(X.device)
        return self

    def predict(self, X: Tensor) -> Tensor:
        # log P(x|c) = -0.5 * sum(log(2π var) + (x-μ)²/var)
        # shape: (N, C)
        x = X.data.unsqueeze(1)            # (N, 1, D)
        mu = self._means.unsqueeze(0)      # (1, C, D)
        var = self._vars.unsqueeze(0)      # (1, C, D)
        log_lh = -0.5 * (
            (2 * math.pi * var).log() + (x - mu) ** 2 / var
        ).sum(-1)                          # (N, C)
        log_post = log_lh + self._log_priors
        pred_idx = log_post.argmax(1)
        return Tensor(self._classes[pred_idx])


# ------------------------------------------------------------------ #
# Principal Component Analysis (SVD-based)
# ------------------------------------------------------------------ #

class PCA:
    def __init__(self, n_components: int):
        self._k = n_components
        self._components: Optional[Tensor] = None
        self._mean: Optional[Tensor] = None
        self._explained_variance: Optional[Tensor] = None

    def fit(self, X: Tensor) -> "PCA":
        x = X.data.double()
        mu = x.mean(0, keepdim=True)
        x_centered = x - mu
        _, S, Vt = torch.linalg.svd(x_centered, full_matrices=False)
        self._mean = Tensor(mu.float().squeeze(0))
        self._components = Tensor(Vt[:self._k].float())
        var = (S[:self._k] ** 2 / (x.shape[0] - 1)).float()
        self._explained_variance = Tensor(var)
        return self

    def transform(self, X: Tensor) -> Tensor:
        centered = X - self._mean
        return centered @ self._components.T

    def fit_transform(self, X: Tensor) -> Tensor:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: Tensor) -> Tensor:
        return Z @ self._components + self._mean

    @property
    def explained_variance_ratio(self) -> Tensor:
        total = self._explained_variance.data.sum()
        return Tensor(self._explained_variance.data / total)


# ------------------------------------------------------------------ #
# k-Means clustering (Lloyd's algorithm — GPU-vectorised)
# ------------------------------------------------------------------ #

class KMeans:
    def __init__(self, k: int, max_iter: int = 100, tol: float = 1e-4,
                 seed: int = 42):
        self._k = k
        self._max_iter = max_iter
        self._tol = tol
        self._seed = seed
        self.centroids: Optional[Tensor] = None
        self.inertia: float = float("inf")

    def fit(self, X: Tensor) -> "KMeans":
        torch.manual_seed(self._seed)
        n, d = X.shape
        device = X.device
        idx = torch.randperm(n, device=device)[:self._k]
        C = X.data[idx].clone()

        for _ in range(self._max_iter):
            # assign
            dist = (X.data.unsqueeze(1) - C.unsqueeze(0)).pow(2).sum(-1)
            labels = dist.argmin(1)

            # update
            new_C = torch.zeros_like(C)
            counts = torch.zeros(self._k, device=device)
            for j in range(self._k):
                mask = labels == j
                if mask.any():
                    new_C[j] = X.data[mask].mean(0)
                    counts[j] = mask.sum()
                else:
                    new_C[j] = C[j]

            shift = (new_C - C).norm()
            C = new_C
            if shift < self._tol:
                break

        self.centroids = Tensor(C)
        dist = (X.data.unsqueeze(1) - C.unsqueeze(0)).pow(2).sum(-1)
        self.inertia = dist.min(1).values.sum().item()
        self._labels = labels
        return self

    def predict(self, X: Tensor) -> Tensor:
        dist = (X.data.unsqueeze(1) - self.centroids.data.unsqueeze(0)
                ).pow(2).sum(-1)
        return Tensor(dist.argmin(1))
