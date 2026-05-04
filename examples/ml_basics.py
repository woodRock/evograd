"""
EvoGrad example: basic ML algorithms on synthetic data.

All algorithms run on EvoGrad Tensors; the underlying compute uses
PyTorch so everything can move to CUDA with a single .to("cuda") call.

Run:
  cd /path/to/evograd
  python -m examples.ml_basics
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from evograd.core import Tensor
from evograd.ml import (
    LinearRegression, LogisticRegression,
    KNearestNeighbours, GaussianNaiveBayes,
    PCA, KMeans,
    accuracy, mse, r2_score,
)

torch.manual_seed(42)


def section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print("─" * 50)


# ------------------------------------------------------------------ #
# Linear Regression
# ------------------------------------------------------------------ #
section("Linear Regression (normal equations)")

X_r = Tensor(torch.randn(300, 5))
true_w = Tensor(torch.tensor([1.5, -2.0, 0.5, 3.0, -1.0]))
y_r = Tensor(X_r.data @ true_w.data + torch.randn(300) * 0.2)

lr = LinearRegression(fit_intercept=True).fit(X_r, y_r)
pred_r = lr.predict(X_r)
print(f"  R²:  {r2_score(pred_r, y_r):.4f}  (expect ≈ 1.0)")
print(f"  MSE: {mse(pred_r, y_r):.5f}  (expect ≈ 0.04)")


# ------------------------------------------------------------------ #
# Logistic Regression
# ------------------------------------------------------------------ #
section("Logistic Regression (multi-class SGD)")

n_per_class = 100
X_cls = torch.cat([
    torch.randn(n_per_class, 10) + torch.tensor([3., 0.] + [0.]*8),
    torch.randn(n_per_class, 10) + torch.tensor([-3., 0.] + [0.]*8),
    torch.randn(n_per_class, 10) + torch.tensor([0., 3.] + [0.]*8),
])
y_cls = torch.cat([torch.zeros(n_per_class),
                   torch.ones(n_per_class),
                   torch.full((n_per_class,), 2)])
X_c, y_c = Tensor(X_cls), Tensor(y_cls)

log_reg = LogisticRegression(lr=0.3, epochs=200).fit(X_c, y_c)
pred_c = log_reg.predict(X_c)
print(f"  Accuracy: {accuracy(pred_c, y_c):.3f}  (expect > 0.90)")


# ------------------------------------------------------------------ #
# k-Nearest Neighbours
# ------------------------------------------------------------------ #
section("k-Nearest Neighbours (exact GPU distance)")

knn = KNearestNeighbours(k=5).fit(X_c, y_c)
pred_knn = knn.predict(Tensor(X_cls[:50]))
print(f"  Accuracy (first 50): "
      f"{accuracy(pred_knn, Tensor(y_cls[:50])):.3f}")


# ------------------------------------------------------------------ #
# Gaussian Naive Bayes
# ------------------------------------------------------------------ #
section("Gaussian Naive Bayes")

gnb = GaussianNaiveBayes().fit(X_c, y_c)
pred_gnb = gnb.predict(X_c)
print(f"  Accuracy: {accuracy(pred_gnb, y_c):.3f}")


# ------------------------------------------------------------------ #
# PCA
# ------------------------------------------------------------------ #
section("PCA (SVD-based, closed form)")

X_high = Tensor(torch.randn(200, 50))
pca = PCA(n_components=5)
Z = pca.fit_transform(X_high)
X_rec = pca.inverse_transform(Z)
rec_mse = mse(X_rec, X_high)
evr = pca.explained_variance_ratio
print(f"  Compressed: {X_high.shape} → {Z.shape}")
print(f"  Reconstruction MSE: {rec_mse:.4f}")
print(f"  Explained variance (top-5): "
      f"{evr.data.tolist()}")


# ------------------------------------------------------------------ #
# k-Means
# ------------------------------------------------------------------ #
section("k-Means clustering (Lloyd's, GPU-vectorised)")

# 3-cluster data
X_km = torch.cat([
    torch.randn(150, 2) + torch.tensor([5., 0.]),
    torch.randn(150, 2) + torch.tensor([-5., 0.]),
    torch.randn(150, 2) + torch.tensor([0., 5.]),
])
X_km_t = Tensor(X_km)
km = KMeans(k=3, max_iter=100).fit(X_km_t)
pred_km = km.predict(X_km_t)
print(f"  Inertia: {km.inertia:.2f}")
print(f"  Unique cluster labels: {pred_km.data.unique().tolist()}")
# Check roughly correct separation
sizes = [(pred_km.data == i).sum().item() for i in range(3)]
print(f"  Cluster sizes: {sizes}  (expect ≈ [150, 150, 150])")
