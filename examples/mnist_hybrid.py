"""
EvoGrad example: hybrid evolutionary + gradient training on a
synthetic dataset that mirrors MNIST in structure.

Demonstrates:
  1. Writing a network in the EvoGrad DSL.
  2. The compiler pipeline: parse → type-check → codegen.
  3. Hybrid training: evolutionary warm-start then Adam fine-tuning.
  4. Evaluating with EvoGrad ML metrics.

Run:
  cd /path/to/evograd
  python -m examples.mnist_hybrid
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from evograd.core import Tensor
from evograd.compiler import parse, type_check, compile_program
from evograd.dsl import HybridTrainer, HybridConfig
from evograd.ml import accuracy


# ------------------------------------------------------------------ #
# 1. Dataset (synthetic 784-dim classification, 10 classes)
# ------------------------------------------------------------------ #

torch.manual_seed(0)
N_TRAIN, N_TEST = 1000, 200
X_train_t = torch.randn(N_TRAIN, 784)
y_train_t = torch.randint(0, 10, (N_TRAIN,))
X_test_t  = torch.randn(N_TEST,  784)
y_test_t  = torch.randint(0, 10, (N_TEST,))

X_train = Tensor(X_train_t)
y_train = Tensor(y_train_t)
X_test  = Tensor(X_test_t)
y_test  = Tensor(y_test_t)


# ------------------------------------------------------------------ #
# 2. Define the network in EvoGrad DSL
# ------------------------------------------------------------------ #

DSL_SOURCE = """
network MLP {
  type: Float[batch, 784] -> Float[batch, 10]

  layers {
    fc1 = Dense(784, 256) |> ReLU
    fc2 = Dense(256, 128) |> ReLU
    out = Dense(128, 10)  |> Softmax
  }

  forward = fc1 >> fc2 >> out
}

population WeightSearcher {
  genome_type: Float[64, 256]
  fitness: CrossEntropy(MLP)

  evolve {
    select    = Tournament(k=7)
    crossover = Uniform(rate=0.5)
    mutate    = Gaussian(sigma=0.02)
  }
}

train {
  mode = Hybrid(evolutionary, gradient)
}
"""

print("=" * 60)
print("EvoGrad Compiler Pipeline")
print("=" * 60)

# 2a. Parse
program = parse(DSL_SOURCE)
print(f"\n[1] Parsed: {len(program.networks)} network(s), "
      f"{len(program.populations)} population(s)")
for net in program.networks:
    print(f"    network '{net.name}': {len(net.layers)} layer(s), "
          f"forward={net.forward}")

# 2b. Type-check (catches shape mismatches at compile time)
try:
    ir_graphs = type_check(program)
    print(f"\n[2] Type check passed. IR graphs: {list(ir_graphs.keys())}")
    for name, g in ir_graphs.items():
        print(f"\n    --- {name} ---")
        for line in g.summary().split("\n")[1:]:
            print(f"    {line}")
except Exception as e:
    print(f"[2] Shape error caught at compile time: {e}")
    sys.exit(1)

# 2c. Code generation
compiled = compile_program(program)
print(f"\n[3] Compiled: networks={list(compiled.networks.keys())}, "
      f"mode={compiled.train_mode}")
model = compiled.networks["MLP"]
print(f"    PyTorch module: {model}")


# ------------------------------------------------------------------ #
# 3. Hybrid training
# ------------------------------------------------------------------ #

cfg = HybridConfig(
    evo_generations=30,
    pop_size=32,
    mutation_sigma=0.02,
    selection="tournament",
    crossover="sbx",
    gradient_optimizer="adam",
    gradient_lr=1e-3,
    gradient_epochs=5,
    device="cpu",
    verbose=True,
)

print("\n" + "=" * 60)
print("Hybrid Training")
print("=" * 60)

trainer = HybridTrainer(model, cfg)
results = trainer.fit(X_train, y_train)

# ------------------------------------------------------------------ #
# 4. Evaluate
# ------------------------------------------------------------------ #

print("\n" + "=" * 60)
print("Evaluation")
print("=" * 60)

model.eval()
with torch.no_grad():
    logits_train = model(X_train_t)
    logits_test  = model(X_test_t)

pred_train = Tensor(logits_train.argmax(1))
pred_test  = Tensor(logits_test.argmax(1))

train_acc = accuracy(pred_train, y_train)
test_acc  = accuracy(pred_test,  y_test)
print(f"  Train accuracy: {train_acc:.3f}")
print(f"  Test  accuracy: {test_acc:.3f}")

if results["grad_losses"]:
    print(f"  Final gradient loss: {results['grad_losses'][-1]:.5f}")

print("\nEvoGrad hybrid training complete.")
