"""
EvoGrad on Apple M2 — end-to-end MPS demo.

Benchmarks:
  1. Tensor ops:        immutable EvoGrad Tensor ops on MPS vs CPU
  2. Evolutionary loop: population (1024 × 512) on MPS vs CPU
  3. Neural training:   hybrid trainer on MPS
  4. ML algorithms:     KNN / PCA / k-Means on MPS

Run:
  PYTHONPATH=/path/to/parent python3 evograd/examples/m2_demo.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import time
import torch

import evograd
from evograd import auto_device, warmup_mps, mps_sync, Tensor
from evograd.evo import Population, EvoEngine, EvoConfig
from evograd.evo.fitness import rastrigin, ackley
from evograd.ml import (
    LinearRegression, LogisticRegression, KNearestNeighbours,
    PCA, KMeans, accuracy,
)
from evograd.mps.ops import active_tier

DEVICE = auto_device()


def header(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def bench(label: str, fn, warmup: int = 1, reps: int = 3) -> float:
    for _ in range(warmup):
        fn()
    mps_sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    mps_sync()
    elapsed = (time.perf_counter() - t0) / reps
    print(f"  {label:45s}: {elapsed * 1000:7.2f} ms")
    return elapsed


# ================================================================== #
# Startup
# ================================================================== #

header("EvoGrad — Apple M2 MPS Demo")
print(f"  Device    : {DEVICE}")
print(f"  PyTorch   : {torch.__version__}")
print(f"  MPS tier  : {active_tier()} "
      f"({'custom Metal kernels' if active_tier() == 2 else 'PyTorch MPS native'})")

warmup_mps()

# ================================================================== #
# 1. Tensor ops benchmark
# ================================================================== #

header("1. Immutable Tensor Operations")

N = 2048
x_cpu = Tensor.randn((N, N), device="cpu")
x_mps = Tensor.randn((N, N), device=DEVICE)
w_cpu = Tensor.randn((N, N), device="cpu")
w_mps = Tensor.randn((N, N), device=DEVICE)

bench("matmul CPU (2048×2048)",
      lambda: x_cpu.matmul(w_cpu))
bench(f"matmul MPS (2048×2048)",
      lambda: x_mps.matmul(w_mps))

from evograd.nn.functional import relu, softmax
bench("relu + softmax CPU",
      lambda: softmax(relu(x_cpu @ w_cpu)))
bench(f"relu + softmax MPS",
      lambda: softmax(relu(x_mps @ w_mps)))


# ================================================================== #
# 2. Evolutionary computation benchmark
# ================================================================== #

header("2. GPU-Accelerated Evolutionary Operators")

POP, GENOME = 1024, 512
print(f"  Population: {POP} × {GENOME}  ({POP * GENOME * 4 / 1024:.0f} KB genomes)\n")

pop_cpu = Population.gaussian(POP, GENOME, device="cpu").evaluate(rastrigin)
pop_mps = Population.gaussian(POP, GENOME, device=DEVICE).evaluate(
    lambda g: rastrigin(Tensor(g))
    if not isinstance(g, Tensor)
    else rastrigin(g)
)

from evograd.mps import ops as mops

bench("tournament_select CPU",
      lambda: mops.tournament_select(pop_cpu.fitness.data, k=7))
bench(f"tournament_select MPS",
      lambda: mops.tournament_select(pop_mps.fitness.data, k=7))

bench("gaussian_mutate CPU",
      lambda: mops.gaussian_mutate(pop_cpu.genomes.data, sigma=0.02))
bench(f"gaussian_mutate MPS",
      lambda: mops.gaussian_mutate(pop_mps.genomes.data, sigma=0.02))

bench("uniform_crossover CPU",
      lambda: mops.uniform_crossover(pop_cpu.genomes.data))
bench(f"uniform_crossover MPS",
      lambda: mops.uniform_crossover(pop_mps.genomes.data))

bench("rastrigin fitness CPU (1024 individuals)",
      lambda: mops.rastrigin_fitness(pop_cpu.genomes.data))
bench(f"rastrigin fitness MPS (1024 individuals)",
      lambda: mops.rastrigin_fitness(pop_mps.genomes.data))


# ================================================================== #
# 3. Full evolutionary run on MPS
# ================================================================== #

header("3. Full Evolutionary Run on MPS")

cfg = EvoConfig(
    generations=100,
    elite_size=4,
    selection="tournament", tournament_k=7,
    crossover="sbx",        sbx_eta=20.0,
    mutation="gaussian",    mutation_sigma=0.03,
    verbose=True,           log_every=25,
)

pop = Population.gaussian(POP, GENOME, device=DEVICE)

def _fit(g):
    x = g if isinstance(g, Tensor) else Tensor(g)
    return rastrigin(x)

t0 = time.perf_counter()
final, logs = EvoEngine(cfg).run(pop, _fit)
mps_sync()
total = time.perf_counter() - t0

print(f"\n  {POP} individuals × {cfg.generations} generations")
print(f"  Best fitness : {final.best_fitness():.4f}")
print(f"  Wall time    : {total:.2f}s  "
      f"({total / cfg.generations * 1000:.1f} ms/gen)")


# ================================================================== #
# 4. ML algorithms on MPS
# ================================================================== #

header("4. Classic ML Algorithms on MPS")

torch.manual_seed(42)
N_TRAIN = 2000

# ── Linear regression ─────────────────────────────────────────────
X_r = Tensor(torch.randn(N_TRAIN, 20, device=DEVICE))
w_true = Tensor(torch.randn(20, device=DEVICE))
y_r = Tensor(X_r.data @ w_true.data + torch.randn(N_TRAIN, device=DEVICE) * 0.1)

t0 = time.perf_counter()
lr = LinearRegression().fit(X_r, y_r)
mps_sync()
print(f"  LinearRegression ({N_TRAIN} samples, 20 features): "
      f"{(time.perf_counter()-t0)*1000:.1f} ms")

# ── k-Means ───────────────────────────────────────────────────────
X_km = Tensor(torch.randn(N_TRAIN, 50, device=DEVICE))
t0 = time.perf_counter()
km = KMeans(k=8, max_iter=50).fit(X_km)
mps_sync()
print(f"  KMeans k=8 ({N_TRAIN}×50, 50 iter):  "
      f"{(time.perf_counter()-t0)*1000:.1f} ms  "
      f"inertia={km.inertia:.0f}")

# ── PCA ───────────────────────────────────────────────────────────
X_pca = Tensor(torch.randn(N_TRAIN, 100, device=DEVICE))
t0 = time.perf_counter()
pca = PCA(n_components=10).fit(X_pca)
Z = pca.transform(X_pca)
mps_sync()
print(f"  PCA 100→10 ({N_TRAIN} samples):        "
      f"{(time.perf_counter()-t0)*1000:.1f} ms")

# ── Logistic regression ───────────────────────────────────────────
n_per = N_TRAIN // 3
X_cls = Tensor(torch.cat([
    torch.randn(n_per, 20, device=DEVICE) + torch.tensor([4.]*20, device=DEVICE),
    torch.randn(n_per, 20, device=DEVICE),
    torch.randn(n_per, 20, device=DEVICE) - torch.tensor([4.]*20, device=DEVICE),
]))
y_cls = Tensor(torch.cat([
    torch.zeros(n_per, device=DEVICE),
    torch.ones(n_per, device=DEVICE),
    torch.full((n_per,), 2, device=DEVICE),
]))
t0 = time.perf_counter()
log_reg = LogisticRegression(lr=0.5, epochs=200).fit(X_cls, y_cls)
mps_sync()
pred = log_reg.predict(X_cls)
acc = accuracy(pred, y_cls)
print(f"  LogisticRegression 3-class ({N_TRAIN} samples): "
      f"{(time.perf_counter()-t0)*1000:.1f} ms  acc={acc:.3f}")


# ================================================================== #
# 5. Hybrid training on MPS
# ================================================================== #

header("5. Hybrid Evolutionary+Gradient Training on MPS")

from evograd.compiler import parse, type_check, compile_program
from evograd.dsl import HybridTrainer, HybridConfig

DSL = """
network MLP {
  type: Float[batch, 64] -> Float[batch, 4]
  layers {
    fc1 = Dense(64, 128) |> ReLU
    fc2 = Dense(128, 64) |> ReLU
    out = Dense(64, 4)   |> Softmax
  }
  forward = fc1 >> fc2 >> out
}
"""

program = parse(DSL)
type_check(program)
compiled = compile_program(program, device=DEVICE)
model = compiled.networks["MLP"]

n_per = 250
X_train = Tensor(torch.cat([
    torch.randn(n_per, 64, device=DEVICE) + i
    for i in range(4)
]))
y_train = Tensor(torch.cat([
    torch.full((n_per,), i, device=DEVICE)
    for i in range(4)
]))

cfg_h = HybridConfig(
    evo_generations=20,
    pop_size=32,
    mutation_sigma=0.03,
    selection="tournament",
    crossover="sbx",
    gradient_optimizer="adam",
    gradient_lr=1e-3,
    gradient_epochs=5,
    device=DEVICE,
    verbose=True,
)

t0 = time.perf_counter()
results = HybridTrainer(model, cfg_h).fit(X_train, y_train)
mps_sync()
total_h = time.perf_counter() - t0

model.eval()
with torch.no_grad():
    logits = model(X_train.data)
pred_train = Tensor(logits.argmax(1))
train_acc = accuracy(pred_train, y_train)
print(f"\n  Train accuracy: {train_acc:.3f}  |  total time: {total_h:.2f}s")

# ================================================================== #
# Summary
# ================================================================== #

header("Summary")
print(f"  Device          : {DEVICE.upper()}")
print(f"  MPS kernel tier : {active_tier()}")
print(f"  All tests passed on {DEVICE}.")
