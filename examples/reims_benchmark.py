"""
REIMS Fish Species Classification Benchmark.

This script benchmarks all major algorithms in the EvoGrad library on the
REIMS (Rapid Evaporative Ionization Mass Spectrometry) dataset.

Task: Classify fish samples based on 2080 spectral features.
Validation: 3-fold Stratified Cross-Validation.
Dimensionality Reduction: PCA to 32 components (standard for spectral data).
"""

import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add parent dir to path to find evograd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evograd import Tensor, auto_device
from evograd.ml.algorithms import LogisticRegression, KNearestNeighbours, GaussianNaiveBayes, PCA
from evograd.ml.metrics import accuracy
from evograd.evo import (
    Population, EvoEngine, EvoConfig,
    run_pso, run_es, run_eda,
    NetworkFitness
)
from evograd.nn.layers import Linear as Dense, Sequential, ReLU, Softmax

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

DEVICE = auto_device()
DATA_PATH = "data/REIMS.xlsx"
N_SPLITS = 3
PCA_COMPONENTS = 32
RANDOM_SEED = 42

# EA Config
EPOCHS_GD = 150       # Gradient Descent epochs
GENS_EVO = 100        # Evolutionary generations
POP_SIZE = 64         # Population size

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def load_genome_to_model(model, genome):
    offset = 0
    for param in model.parameters():
        n = param.numel()
        param.data.copy_(genome.data[offset:offset + n].reshape(param.shape))
        offset += n

def load_reims():
    print(f"Loading {DATA_PATH}...")
    df = pd.read_excel(DATA_PATH)
    
    # Filter: ignore QC, HM (mix), and MO (oil)
    mask = ~df.iloc[:, 0].str.contains("QC|HM|MO")
    df = df[mask].copy()
    
    # Map to Species: H* -> Hoki, M* -> Mackerel
    def map_species(label):
        if label.startswith("H"): return "Hoki"
        if label.startswith("M"): return "Mackerel"
        return "Unknown"
    
    df["species"] = df.iloc[:, 0].apply(map_species)
    df = df[df["species"] != "Unknown"]
    
    y_raw = df["species"].values
    X_raw = df.iloc[:, 1:-1].values # Exclude original label and new species col
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes ({', '.join(le.classes_)})")
    return X, y, n_classes, le.classes_

# ------------------------------------------------------------------ #
# Model Factory for NetworkFitness
# ------------------------------------------------------------------ #

def make_model_factory(in_features, out_features, device):
    def factory():
        # Simple linear model for fair comparison with Logistic Regression
        return Dense(in_features, out_features, device=device)
    return factory

# ------------------------------------------------------------------ #
# Benchmark Execution
# ------------------------------------------------------------------ #

def run_benchmark():
    X_all, y_all, n_classes, class_names = load_reims()
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    methods = [
        "Logistic Regression",
        "K-Nearest Neighbours",
        "Gaussian Naive Bayes",
        "Neural Network (GD)",
        "Genetic Algorithm (GA)",
        "Particle Swarm (PSO)",
        "Evolution Strategy (ES)",
        "EDA (UMDA)"
    ]
    
    results = {m: [] for m in methods}
    times = {m: [] for m in methods}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
        
        X_train_np, X_test_np = X_all[train_idx], X_all[test_idx]
        y_train_np, y_test_np = y_all[train_idx], y_all[test_idx]
        
        # 1. Dimensionality Reduction with PCA
        pca = PCA(n_components=PCA_COMPONENTS)
        X_train_t = Tensor(torch.from_numpy(X_train_np).float().to(DEVICE))
        X_test_t = Tensor(torch.from_numpy(X_test_np).float().to(DEVICE))
        
        X_train_pca = pca.fit_transform(X_train_t)
        X_test_pca = pca.transform(X_test_t)
        
        y_train_t = Tensor(torch.from_numpy(y_train_np).to(DEVICE))
        y_test_t = Tensor(torch.from_numpy(y_test_np).to(DEVICE))
        
        # 2. Traditional ML Methods
        # Logistic Regression
        t0 = time.perf_counter()
        lr = LogisticRegression(epochs=EPOCHS_GD)
        lr.fit(X_train_pca, y_train_t)
        pred = lr.predict(X_test_pca)
        acc = accuracy(pred, y_test_t)
        results["Logistic Regression"].append(acc)
        times["Logistic Regression"].append(time.perf_counter() - t0)
        print(f"  Logistic Regression: {acc:.4f}")
        
        # KNN
        t0 = time.perf_counter()
        knn = KNearestNeighbours(k=3)
        knn.fit(X_train_pca, y_train_t)
        pred = knn.predict(X_test_pca)
        acc = accuracy(pred, y_test_t)
        results["K-Nearest Neighbours"].append(acc)
        times["K-Nearest Neighbours"].append(time.perf_counter() - t0)
        print(f"  K-Nearest Neighbours: {acc:.4f}")
        
        # Naive Bayes
        t0 = time.perf_counter()
        nb = GaussianNaiveBayes()
        nb.fit(X_train_pca, y_train_t)
        pred = nb.predict(X_test_pca)
        acc = accuracy(pred, y_test_t)
        results["Gaussian Naive Bayes"].append(acc)
        times["Gaussian Naive Bayes"].append(time.perf_counter() - t0)
        print(f"  Gaussian Naive Bayes: {acc:.4f}")
        
        # 3. Neural Network (Gradient Descent)
        t0 = time.perf_counter()
        model = Dense(PCA_COMPONENTS, n_classes, device=DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(EPOCHS_GD):
            # Use model._mod (the underlying nn.Module) to maintain gradients
            logits = model._mod(X_train_pca.data)
            loss = torch.nn.functional.cross_entropy(logits, y_train_t.data.long())
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits = model._mod(X_test_pca.data)
            pred = Tensor(logits.argmax(1))
            acc = accuracy(pred, y_test_t)
        results["Neural Network (GD)"].append(acc)
        times["Neural Network (GD)"].append(time.perf_counter() - t0)
        print(f"  Neural Network (GD): {acc:.4f}")
        
        # 4. Evolutionary Methods
        factory = make_model_factory(PCA_COMPONENTS, n_classes, DEVICE)
        fit_fn = NetworkFitness(factory, X_train_pca, y_train_t)
        genome_len = (PCA_COMPONENTS + 1) * n_classes # weight + bias
        
        # GA
        t0 = time.perf_counter()
        cfg = EvoConfig(generations=GENS_EVO, verbose=False)
        engine = EvoEngine(cfg)
        pop = Population.random(POP_SIZE, genome_len, device=DEVICE)
        best_pop, _ = engine.run(pop, fit_fn)
        best_genome = best_pop.best()
        model = factory()
        load_genome_to_model(model, best_genome)
        with torch.no_grad():
            logits = model(X_test_pca)
            pred = Tensor(logits.data.argmax(1))
            acc = accuracy(pred, y_test_t)
        results["Genetic Algorithm (GA)"].append(acc)
        times["Genetic Algorithm (GA)"].append(time.perf_counter() - t0)
        print(f"  Genetic Algorithm (GA): {acc:.4f}")
        
        # PSO
        t0 = time.perf_counter()
        from evograd.evo.pso import PSO
        pso = PSO(POP_SIZE, genome_len, device=DEVICE)
        for _ in range(GENS_EVO): pso.step(fit_fn)
        model = factory()
        load_genome_to_model(model, Tensor(pso.gbest_pos))
        with torch.no_grad():
            logits = model(X_test_pca)
            pred = Tensor(logits.data.argmax(1))
            acc = accuracy(pred, y_test_t)
        results["Particle Swarm (PSO)"].append(acc)
        times["Particle Swarm (PSO)"].append(time.perf_counter() - t0)
        print(f"  Particle Swarm (PSO): {acc:.4f}")

        # ES
        t0 = time.perf_counter()
        from evograd.evo.es import ES
        es = ES(mu=POP_SIZE//4, lam=POP_SIZE, genome_length=genome_len, device=DEVICE)
        for _ in range(GENS_EVO): es.step(fit_fn)
        model = factory()
        load_genome_to_model(model, Tensor(es.best_x))
        with torch.no_grad():
            logits = model(X_test_pca)
            pred = Tensor(logits.data.argmax(1))
            acc = accuracy(pred, y_test_t)
        results["Evolution Strategy (ES)"].append(acc)
        times["Evolution Strategy (ES)"].append(time.perf_counter() - t0)
        print(f"  Evolution Strategy (ES): {acc:.4f}")

        # EDA
        t0 = time.perf_counter()
        from evograd.evo.eda import UMDA
        eda = UMDA(pop_size=POP_SIZE, genome_length=genome_len, device=DEVICE)
        for _ in range(GENS_EVO): eda.step(fit_fn)
        model = factory()
        load_genome_to_model(model, Tensor(eda.best_x))
        with torch.no_grad():
            logits = model(X_test_pca)
            pred = Tensor(logits.data.argmax(1))
            acc = accuracy(pred, y_test_t)
        results["EDA (UMDA)"].append(acc)
        times["EDA (UMDA)"].append(time.perf_counter() - t0)
        print(f"  EDA (UMDA): {acc:.4f}")



    # ------------------------------------------------------------------ #
    # Final Report
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"  REIMS Fish Species Classification (3-fold Stratified CV)")
    print(f"  Samples: {X_all.shape[0]} | PCA: {PCA_COMPONENTS} | Device: {DEVICE}")
    print("=" * 60)
    
    COL = (25, 15, 15)
    print(f"{'Method':<{COL[0]}}  {'Mean Accuracy':>{COL[1]}}  {'Mean Time (s)':>{COL[2]}}")
    print("-" * sum(COL))
    
    for m in methods:
        mean_acc = np.mean(results[m])
        std_acc = np.std(results[m])
        mean_time = np.mean(times[m])
        print(f"{m:<{COL[0]}}  {mean_acc:>{COL[1]}.4f} ± {std_acc:.3f}  {mean_time:>{COL[2]}.2f}s")
    
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
