<div align="center">

# EvoGrad

**GPU-accelerated evolutionary computation + functional deep learning — one library, two paradigms.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/version-0.1.0-brightgreen)](https://github.com/woodRock/evograd)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![MPS](https://img.shields.io/badge/Apple%20Silicon-MPS-black?logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)](https://github.com/woodRock/evograd)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

EvoGrad treats every population as a single contiguous tensor. Selection, crossover, mutation, and fitness evaluation are all batched operations — no Python loops over individuals, no host/device round-trips between steps. The result is **5.6× faster** than DEAP on average, peaking at **9.6×** on Evolution Strategies.

The same `Tensor` primitive powers a NumPy-style functional deep learning stack, so you can freely mix gradient descent and evolutionary search in one codebase.

---

## Features

- **Vectorised evolutionary algorithms** — GA, PSO, EDA (UMDA), (μ,λ)-ES, and linear GP, all expressed as batched tensor ops
- **Functional neural networks** — immutable tensors, pure-function layers, no implicit state
- **Hybrid training** — evolutionary warm-start followed by Adam/SGD fine-tuning in a single `HybridTrainer` call
- **Native Data Pipeline** — `TensorDataset` and `DataLoader` for efficient batching of EvoGrad Tensors
- **PyTorch-like Training** — `Adam`/`SGD` optimizers, standard schedulers, and `xavier`/`kaiming` initializers
- **First-class Apple Silicon** — automatic MPS dispatch; ops unsupported by MPS fall back to CPU transparently
- **CUDA support** — optional custom CUDA kernels; pure-PyTorch fallback ships with no build step
- **Rich operator library** — 4 selection strategies, 5 crossover ops, 5 mutation ops, all O(P·G) on-device
- **Classic fitness landscapes** — Rastrigin, Ackley, Rosenbrock, Sphere, Schwefel built in
- **ML algorithms** — Linear/Logistic regression, KNN, Naïve Bayes, PCA, K-Means on EvoGrad Tensors
- **DSL compiler** — define networks and populations in a typed DSL; parse → type-check → codegen pipeline

---

## Installation

### CPU / Apple Silicon (MPS)

```bash
pip install evograd
```

Or install directly from source:

```bash
git clone https://github.com/woodRock/evograd.git
cd evograd
pip install -e .
```

### With CUDA kernels

```bash
pip install -e ".[cuda]"
# or, to build the extension in-place:
python setup.py build_ext --inplace
```

> **Requirements:** `nvcc` and CUDA ≥ 11.8. The pure-PyTorch fallback is used automatically when the extension is absent.

### With benchmark dependencies

```bash
pip install -e ".[benchmark]"   # adds deap
pip install -e ".[dev]"         # adds pytest, pytest-cov
pip install -e ".[all]"         # everything
```

---

## Quick Start

### Evolutionary Computation

```python
from evograd import auto_device, Population, EvoEngine, EvoConfig
from evograd.evo.fitness import rastrigin

device = auto_device()   # "mps" | "cuda" | "cpu"

# Entire population stored as one (512, 20) tensor — zero Python loops
pop = Population.random(512, 20, low=-5.12, high=5.12, device=device)

cfg = EvoConfig(
    generations=200,
    selection="tournament", tournament_k=7,
    crossover="sbx",        sbx_eta=20.0,
    mutation="gaussian",    mutation_sigma=0.05,
    verbose=True,
)
final_pop, logs = EvoEngine(cfg).run(pop, rastrigin)
print(f"Best fitness: {final_pop.best_fitness():.4f}")
```

### PSO / ES / EDA / GP

```python
from evograd.evo import run_pso, run_es, run_eda, run_gp
from evograd.evo.fitness import ackley
import torch

device = "mps"

# Particle Swarm
best, log = run_pso(ackley, pop_size=256, genome_length=20,
                    generations=100, device=device)

# Self-adaptive (μ,λ)-ES
best, log = run_es(ackley, mu=50, lam=200,
                   genome_length=20, generations=100, device=device)

# UMDA / EDA
best, log = run_eda(ackley, pop_size=256, genome_length=20,
                    generations=100, device=device)

# Linear GP — symbolic regression
X = torch.linspace(-1, 1, 50, device=device)
y = torch.sin(2 * 3.14159 * X) + 0.1 * torch.randn(50, device=device)
best, log = run_gp(X, y, pop_size=256, prog_len=16,
                   generations=80, device=device)
```

### Neural Networks

```python
from evograd.core import Tensor
from evograd.nn import Sequential, Linear, ReLU, Softmax
import torch

model = Sequential(
    Linear(784, 256), ReLU(),
    Linear(256, 128), ReLU(),
    Linear(128, 10),  Softmax(),
)

X = Tensor(torch.randn(32, 784))
logits = model(X)          # shape (32, 10) — fully immutable
```

### Hybrid Training (Evo → Gradient)

```python
from evograd.dsl import HybridTrainer, HybridConfig
from evograd.nn import Sequential, Linear, ReLU, Softmax
from evograd.core import Tensor
import torch

model = Sequential(Linear(10, 32), ReLU(), Linear(32, 2), Softmax())

X = Tensor(torch.randn(200, 10))
y = Tensor(torch.randint(0, 2, (200,)))

cfg = HybridConfig(
    evo_generations=50, pop_size=64,      # Phase 1: evolutionary warm-start
    gradient_optimizer="adam", gradient_epochs=20,  # Phase 2: Adam fine-tune
    device="mps",
)
trainer = HybridTrainer(lambda: model, cfg)
trainer.fit(X, y)
```

### DSL (Domain-Specific Language)

```python
from evograd.compiler import parse, type_check, compile_program

src = """
network MLP {
  type: Float[batch, 784] -> Float[batch, 10]

  layers {
    fc1 = Dense(784, 256) |> ReLU
    fc2 = Dense(256, 128) |> ReLU
    out = Dense(128, 10)  |> Softmax
  }

  forward = fc1 >> fc2 >> out
}
"""

ast     = parse(src)
checked = type_check(ast)
program = compile_program(checked)
model   = program.networks["MLP"]
```

---

## Benchmark vs DEAP

All timings are median wall-clock time over 5 trials, measured on **Apple M-series (MPS)**.  
Population size 512, 150 generations (80/256 for GP).  Problem: Rastrigin-20d (GA/PSO/EDA/ES) and symbolic regression sin(2πx) (GP).

| Algorithm | Problem | Pop | Gens | DEAP | EvoGrad | Speedup |
|-----------|---------|----:|-----:|-----:|--------:|:-------:|
| GA | Rastrigin-20d | 512 | 150 | 1382 ± 12 ms | 169 ± 31 ms | **8.2×** |
| PSO | Rastrigin-20d | 512 | 150 | 1049 ± 13 ms | 118 ± 5 ms | **8.9×** |
| EDA (UMDA) | Rastrigin-20d | 512 | 150 | 838 ± 11 ms | 134 ± 9 ms | **6.3×** |
| ES (μ,λ) | Rastrigin-20d | 512 | 150 | 1504 ± 10 ms | 157 ± 6 ms | **9.6×** |
| GP (linear) | SymReg sin(2πx) | 256 | 80 | 6111 ± 7813 ms | 4713 ± 201 ms | **1.3×** |
| **Geometric mean** | | | | | | **5.6×** |

> Reproduce: `python3 examples/benchmark.py`  
> DEAP GP variance is high because tree bloat causes occasional `SyntaxError`; EvoGrad linear GP has bounded program length.

---

## Module Overview

| Package | Contents |
|---------|----------|
| `evograd.core` | `Tensor`, `TensorType`, `DType`, `IRGraph` — the immutable tensor primitive and IR |
| `evograd.nn` | `Linear`, `Conv2d`, `Embedding`, `ReLU/GELU/SiLU`, `LayerNorm`, `Dropout`, `Residual`, … |
| `evograd.evo` | `Population`, `EvoEngine`, `EvoConfig`, `run_pso`, `run_es`, `run_eda`, `run_gp` |
| `evograd.evo.operators` | Tournament/roulette/rank/SUS selection; uniform/one-point/two-point/SBX/blend crossover; Gaussian/uniform/polynomial/bitflip/creep mutation |
| `evograd.evo.fitness` | `rastrigin`, `ackley`, `rosenbrock`, `sphere`, `schwefel`, `NetworkFitness`, `EnsembleFitness` |
| `evograd.dsl` | `HybridTrainer`, `HybridConfig` — combined evo + gradient training |
| `evograd.compiler` | `parse`, `type_check`, `compile_program` — DSL → `CompiledProgram` |
| `evograd.ml` | `LinearRegression`, `LogisticRegression`, `KNN`, `GaussianNaiveBayes`, `PCA`, `KMeans`, `accuracy`, `mse`, `r2_score`, … |
| `evograd.mps` | `auto_device`, `warmup_mps`, `mps_sync` — Apple Silicon utilities |

---

## Examples

| Script | Description |
|--------|-------------|
| `examples/benchmark.py` | Wall-clock comparison of GA / PSO / EDA / ES / GP against DEAP |
| `examples/evo_benchmarks.py` | Rastrigin, Ackley, Rosenbrock on multiple selection strategies |
| `examples/ml_basics.py` | Linear regression, logistic regression, KNN, PCA, K-Means |
| `examples/mnist_hybrid.py` | Hybrid evo+gradient training on a synthetic MNIST-like dataset |
| `examples/m2_demo.py` | MPS-specific performance demo for Apple Silicon |

---

## License

MIT © [woodRock](https://github.com/woodRock)
