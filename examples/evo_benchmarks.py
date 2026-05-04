"""
EvoGrad example: GPU-accelerated evolutionary computation on
classic benchmark fitness landscapes.

Demonstrates:
  - Population treated as a single (P, G) tensor.
  - All evolutionary ops (select, crossover, mutate) are batched.
  - Multiple selection strategies compared on the same problem.
  - Zero Python loops over individuals in the inner loop.

Run:
  cd /path/to/evograd
  python -m examples.evo_benchmarks
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from evograd.evo import Population, EvoEngine, EvoConfig
from evograd.evo.fitness import rastrigin, ackley, rosenbrock


def run_experiment(name: str, fitness_fn, config: EvoConfig,
                   pop_size: int = 128, genome_len: int = 20,
                   device: str = "cpu"):
    print(f"\n{'─'*55}")
    print(f"  {name}  |  pop={pop_size}  genome={genome_len}  "
          f"gens={config.generations}")
    print(f"  selection={config.selection}  "
          f"crossover={config.crossover}  "
          f"mutation={config.mutation}(σ={config.mutation_sigma})")
    print(f"{'─'*55}")

    pop = Population.gaussian(pop_size, genome_len,
                               mean=0.0, std=1.0, device=device)
    engine = EvoEngine(config)
    final_pop, logs = engine.run(pop, fitness_fn)

    print(f"\n  Best fitness: {final_pop.best_fitness():.5f}")
    print(f"  Best genome (first 5 dims): "
          f"{final_pop.best().data[:5].tolist()}")
    return final_pop, logs


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"EvoGrad Evolutionary Benchmarks  (device={DEVICE})")
    print("=" * 55)

    # ── Rastrigin: highly multi-modal ────────────────────────────
    cfg = EvoConfig(
        generations=200, elite_size=2,
        selection="tournament", tournament_k=7,
        crossover="sbx", sbx_eta=20.0,
        mutation="gaussian", mutation_sigma=0.05,
        lower_bound=-5.12, upper_bound=5.12,
        verbose=True, log_every=50,
    )
    run_experiment("Rastrigin (20-dim)", rastrigin, cfg,
                   pop_size=200, genome_len=20, device=DEVICE)

    # ── Ackley: complex multi-modal ───────────────────────────────
    cfg2 = EvoConfig(
        generations=150, elite_size=2,
        selection="rank",
        crossover="uniform", crossover_rate=0.7,
        mutation="polynomial", mutation_rate=0.1,
        lower_bound=-32.0, upper_bound=32.0,
        verbose=True, log_every=50,
    )
    run_experiment("Ackley (20-dim)", ackley, cfg2,
                   pop_size=150, genome_len=20, device=DEVICE)

    # ── Rosenbrock: narrow curved valley ─────────────────────────
    cfg3 = EvoConfig(
        generations=300, elite_size=4,
        selection="sus",
        crossover="blend", crossover_rate=0.5,
        mutation="gaussian", mutation_sigma=0.02,
        verbose=True, log_every=75,
    )
    run_experiment("Rosenbrock (10-dim)", rosenbrock, cfg3,
                   pop_size=128, genome_len=10, device=DEVICE)

    # ── Selection strategy comparison on Rastrigin ───────────────
    print("\n\n" + "=" * 55)
    print("Selection Strategy Comparison (Rastrigin, 50-dim)")
    print("=" * 55)
    strategies = [
        ("tournament", dict(tournament_k=7)),
        ("roulette",   {}),
        ("rank",       {}),
        ("sus",        {}),
    ]
    base_cfg = dict(
        generations=100, elite_size=2,
        crossover="sbx", mutation="gaussian",
        mutation_sigma=0.05, verbose=False,
    )
    results = {}
    for sel, extra in strategies:
        pop = Population.gaussian(128, 50, device=DEVICE)
        cfg = EvoConfig(**{**base_cfg, "selection": sel, **extra})
        engine = EvoEngine(cfg)
        final, _ = engine.run(pop, rastrigin)
        results[sel] = final.best_fitness()
        print(f"  {sel:12s}: best={final.best_fitness():.4f}")

    best_sel = max(results, key=results.get)
    print(f"\n  → Best strategy on this run: {best_sel}")
