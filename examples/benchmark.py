"""
EvoGrad vs DEAP — wall-clock benchmark across five metaheuristics.

Algorithms
----------
  GA   — Genetic Algorithm          (real-valued, SBX + Gaussian mutation)
  PSO  — Particle Swarm Optimisation
  EDA  — Estimation of Distribution (UMDA, Gaussian marginals)
  ES   — Evolution Strategy         (μ,λ)-ES with self-adaptive σ
  GP   — Genetic Programming        (tree GP in DEAP, linear GP in EvoGrad)

Problem
-------
  GA / PSO / EDA / ES :  Rastrigin minimisation, 20 dimensions
  GP                  :  Symbolic regression  y = sin(2πx) + 0.1·noise

Metric
------
  Median wall-clock time for N_GENS generations, across TRIALS repeats.
  Fitness evaluations per trial = pop_size × N_GENS (same for both).
  Speedup = DEAP_median / EvoGrad_median.

Run
---
  PYTHONPATH=/path/to/parent python3 evograd/examples/benchmark.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import math, time, statistics
from typing import Callable

import torch
import numpy as np

# ── EvoGrad ─────────────────────────────────────────────────────────────
import evograd
from evograd import auto_device, warmup_mps, mps_sync, Tensor
from evograd.evo import (Population, EvoEngine, EvoConfig,
                          run_pso, run_es, run_eda, run_gp)
from evograd.evo.fitness import rastrigin as _eg_rastrigin_fn

# ── DEAP ────────────────────────────────────────────────────────────────
import random
from deap import base, creator, tools, algorithms, gp as deap_gp
import operator

# ── Config ───────────────────────────────────────────────────────────────
DEVICE   = auto_device()
N_DIM    = 20
POP      = 512
N_GENS   = 150
TRIALS   = 5
GP_GENS  = 80    # GP is slower; use fewer gens to keep total time reasonable
GP_POP   = 256

# GP regression dataset
torch.manual_seed(0)
GP_X_T   = torch.linspace(-1, 1, 50)
GP_Y_T   = torch.sin(2 * math.pi * GP_X_T) + 0.1 * torch.randn(50)
GP_X_NP  = GP_X_T.numpy()
GP_Y_NP  = GP_Y_T.numpy()

warmup_mps()


# ═══════════════════════════════════════════════════════════════════════ #
#  Timing helpers
# ═══════════════════════════════════════════════════════════════════════ #

def timed(fn: Callable, trials: int = TRIALS) -> tuple[float, float]:
    """Return (median_s, stdev_s) over `trials` calls."""
    times = []
    for _ in range(trials):
        mps_sync()
        t0 = time.perf_counter()
        fn()
        mps_sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), statistics.stdev(times) if trials > 1 else 0.0


def ms(s: float) -> str:
    return f"{s * 1000:.1f}"


# ═══════════════════════════════════════════════════════════════════════ #
#  EvoGrad fitness wrappers
# ═══════════════════════════════════════════════════════════════════════ #

def eg_rastrigin(g: Tensor) -> Tensor:
    return _eg_rastrigin_fn(g)


# ═══════════════════════════════════════════════════════════════════════ #
#  ── GA ──────────────────────────────────────────────────────────────── #
# ═══════════════════════════════════════════════════════════════════════ #

def bench_ga_deap():
    # One-time DEAP creator setup (idempotent)
    if not hasattr(creator, "FitnessMaxGA"):
        creator.create("FitnessMaxGA", base.Fitness, weights=(1.0,))
        creator.create("IndGA", list, fitness=creator.FitnessMaxGA)

    tb = base.Toolbox()
    tb.register("attr", random.uniform, -5.12, 5.12)
    tb.register("individual", tools.initRepeat,
                creator.IndGA, tb.attr, n=N_DIM)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate",
                lambda ind: (-sum(10 + x**2 - 10*math.cos(2*math.pi*x)
                                  for x in ind),))
    tb.register("select",  tools.selTournament, tournsize=7)
    tb.register("mate",    tools.cxSimulatedBinaryBounded,
                eta=20, low=-5.12, up=5.12)
    tb.register("mutate",  tools.mutPolynomialBounded,
                eta=20, low=-5.12, up=5.12, indpb=1/N_DIM)

    def run():
        pop = tb.population(n=POP)
        for ind in pop:
            ind.fitness.values = tb.evaluate(ind)
        for _ in range(N_GENS):
            offspring = tb.select(pop, len(pop))
            offspring = list(map(tb.clone, offspring))
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.9:
                    tb.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values
            for mut in offspring:
                if random.random() < 0.1:
                    tb.mutate(mut)
                    del mut.fitness.values
            invalid = [i for i in offspring if not i.fitness.valid]
            for ind in invalid:
                ind.fitness.values = tb.evaluate(ind)
            pop[:] = offspring

    return run


def bench_ga_evograd():
    cfg = EvoConfig(generations=N_GENS, elite_size=2,
                    selection="tournament", tournament_k=7,
                    crossover="sbx", sbx_eta=20.0,
                    mutation="gaussian", mutation_sigma=0.05,
                    verbose=False)
    engine = EvoEngine(cfg)

    def run():
        pop = Population.gaussian(POP, N_DIM,
                                   low=-5.12, high=5.12, device=DEVICE)
        engine.run(pop, eg_rastrigin)

    return run


# ═══════════════════════════════════════════════════════════════════════ #
#  ── PSO ─────────────────────────────────────────────────────────────── #
# ═══════════════════════════════════════════════════════════════════════ #

def bench_pso_deap():
    if not hasattr(creator, "FitnessMaxPSO"):
        creator.create("FitnessMaxPSO", base.Fitness, weights=(1.0,))
        creator.create("ParticlePSO", list, fitness=creator.FitnessMaxPSO,
                       speed=list, best=None, smin=-2.0, smax=2.0)

    def generate(size, pmin, pmax, smin, smax):
        part = creator.ParticlePSO(
            random.uniform(pmin, pmax) for _ in range(size))
        part.speed = [random.uniform(smin, smax) for _ in range(size)]
        return part

    def update(part, best, phi1=1.494, phi2=1.494, w=0.729):
        u1 = [random.random() for _ in range(len(part))]
        u2 = [random.random() for _ in range(len(part))]
        v_u1 = list(map(operator.mul, [phi1 * x for x in u1],
                        list(map(operator.sub, part.best, part))))
        v_u2 = list(map(operator.mul, [phi2 * x for x in u2],
                        list(map(operator.sub, best, part))))
        part.speed = list(map(operator.add,
                              [w * s for s in part.speed],
                              list(map(operator.add, v_u1, v_u2))))
        for i, speed in enumerate(part.speed):
            part.speed[i] = max(min(speed, part.smax), part.smin)
        part[:] = list(map(operator.add, part, part.speed))

    def evaluate(ind):
        return (-sum(10 + x**2 - 10*math.cos(2*math.pi*x)
                     for x in ind),)

    def run():
        tb = base.Toolbox()
        tb.register("particle", generate,
                    size=N_DIM, pmin=-5.12, pmax=5.12,
                    smin=-2.0, smax=2.0)
        tb.register("population", tools.initRepeat, list, tb.particle)
        tb.register("evaluate", evaluate)
        tb.register("update", update)

        pop = tb.population(n=POP)
        best = None
        for part in pop:
            part.fitness.values = tb.evaluate(part)
            part.best = creator.ParticlePSO(part)
            part.best.fitness.values = part.fitness.values
            if best is None or best.fitness < part.fitness:
                best = creator.ParticlePSO(part)
                best.fitness.values = part.fitness.values

        for _ in range(N_GENS):
            for part in pop:
                tb.update(part, best)
                part.fitness.values = tb.evaluate(part)
                if part.fitness > part.best.fitness:
                    part.best = creator.ParticlePSO(part)
                    part.best.fitness.values = part.fitness.values
                if best is None or best.fitness < part.best.fitness:
                    best = creator.ParticlePSO(part)
                    best.fitness.values = part.best.fitness.values

    return run


def bench_pso_evograd():
    def run():
        run_pso(eg_rastrigin, pop_size=POP,
                genome_length=N_DIM, generations=N_GENS, device=DEVICE)
    return run


# ═══════════════════════════════════════════════════════════════════════ #
#  ── EDA (UMDA) ──────────────────────────────────────────────────────── #
# ═══════════════════════════════════════════════════════════════════════ #

def bench_eda_deap():
    """UMDA in pure Python — DEAP has no built-in EDA so we implement it
    using the same DEAP individual/fitness infrastructure."""
    if not hasattr(creator, "FitnessMaxEDA"):
        creator.create("FitnessMaxEDA", base.Fitness, weights=(1.0,))
        creator.create("IndEDA", list, fitness=creator.FitnessMaxEDA)

    def evaluate(ind):
        return (-sum(10 + x**2 - 10*math.cos(2*math.pi*x)
                     for x in ind),)

    n_select = POP // 2

    def run():
        # initialise
        pop = [creator.IndEDA(
            [random.gauss(0, 1) for _ in range(N_DIM)]
        ) for _ in range(POP)]
        for ind in pop:
            ind.fitness.values = evaluate(ind)

        # UMDA loop
        mu_est  = [0.0] * N_DIM
        sig_est = [1.0] * N_DIM

        for _ in range(N_GENS):
            # evaluate
            for ind in pop:
                ind.fitness.values = evaluate(ind)

            # select top half
            pop.sort(key=lambda i: i.fitness.values[0], reverse=True)
            selected = pop[:n_select]

            # estimate marginals
            for d in range(N_DIM):
                vals = [ind[d] for ind in selected]
                mu_est[d]  = sum(vals) / n_select
                var = sum((v - mu_est[d])**2 for v in vals) / n_select
                sig_est[d] = math.sqrt(max(var, 1e-6))

            # sample new population
            pop = [creator.IndEDA(
                [random.gauss(mu_est[d], sig_est[d]) for d in range(N_DIM)]
            ) for _ in range(POP)]

    return run


def bench_eda_evograd():
    def run():
        run_eda(eg_rastrigin, pop_size=POP,
                genome_length=N_DIM, generations=N_GENS, device=DEVICE)
    return run


# ═══════════════════════════════════════════════════════════════════════ #
#  ── ES (μ,λ)-ES ─────────────────────────────────────────────────────── #
# ═══════════════════════════════════════════════════════════════════════ #

def bench_es_deap():
    MU, LAM = POP // 5, POP
    N = N_DIM
    tau       = 1.0 / math.sqrt(2 * N)
    tau_prime = 1.0 / math.sqrt(2 * math.sqrt(N))

    if not hasattr(creator, "FitnessMaxES"):
        creator.create("FitnessMaxES", base.Fitness, weights=(1.0,))
        creator.create("IndES", list, fitness=creator.FitnessMaxES,
                       strategy=None)

    def evaluate(ind):
        return (-sum(10 + x**2 - 10*math.cos(2*math.pi*x)
                     for x in ind),)

    def mutate_es(ind):
        global_noise = random.gauss(0, 1)
        for i in range(len(ind)):
            ind.strategy[i] *= math.exp(
                tau_prime * global_noise + tau * random.gauss(0, 1)
            )
            ind.strategy[i] = max(ind.strategy[i], 1e-5)
            ind[i] += ind.strategy[i] * random.gauss(0, 1)
        return (ind,)

    def run():
        pop = []
        for _ in range(MU):
            ind = creator.IndES([random.gauss(0, 1) for _ in range(N)])
            ind.strategy = [0.3] * N
            ind.fitness.values = evaluate(ind)
            pop.append(ind)

        for _ in range(N_GENS):
            # generate λ offspring (cycle through parents)
            offspring = []
            for i in range(LAM):
                parent = pop[i % MU]
                child = creator.IndES(parent)
                child.strategy = list(parent.strategy)
                child.fitness.values = (0.0,)
                offspring.append(child)

            for child in offspring:
                mutate_es(child)
                child.fitness.values = evaluate(child)

            # (μ, λ) selection
            offspring.sort(key=lambda i: i.fitness.values[0], reverse=True)
            pop = offspring[:MU]

    return run


def bench_es_evograd():
    MU, LAM = POP // 5, POP

    def run():
        run_es(eg_rastrigin, mu=MU, lam=LAM,
               genome_length=N_DIM, generations=N_GENS, device=DEVICE)
    return run


# ═══════════════════════════════════════════════════════════════════════ #
#  ── GP ──────────────────────────────────────────────────────────────── #
# ═══════════════════════════════════════════════════════════════════════ #

def bench_gp_deap():
    """Tree GP via DEAP for symbolic regression of sin(2πx)."""
    pset = deap_gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(lambda a, b: a / (abs(b) + 1e-6), 2, name="pdiv")
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addEphemeralConstant("rand", lambda: random.uniform(-2, 2))
    pset.renameArguments(ARG0="x")

    if not hasattr(creator, "FitnessMinGP"):
        creator.create("FitnessMinGP", base.Fitness, weights=(-1.0,))
        creator.create("IndGP", deap_gp.PrimitiveTree,
                       fitness=creator.FitnessMinGP)

    tb = base.Toolbox()
    tb.register("expr",       deap_gp.genHalfAndHalf,
                pset=pset, min_=2, max_=4)
    tb.register("individual", tools.initIterate,
                creator.IndGP, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("compile",    deap_gp.compile, pset=pset)
    tb.register("select",     tools.selTournament, tournsize=5)
    tb.register("mate",       deap_gp.cxOnePoint)
    tb.register("mutate",     deap_gp.mutUniform, expr=tb.expr, pset=pset)

    def evaluate(ind):
        fn = tb.compile(expr=ind)
        try:
            mse = sum((fn(x) - y)**2
                      for x, y in zip(GP_X_NP, GP_Y_NP)) / len(GP_X_NP)
            return (float(mse),)
        except Exception:
            return (1e6,)

    tb.register("evaluate", evaluate)

    def run():
        pop = tb.population(n=GP_POP)
        for ind in pop:
            ind.fitness.values = tb.evaluate(ind)
        for _ in range(GP_GENS):
            offspring = tb.select(pop, len(pop))
            offspring = list(map(tb.clone, offspring))
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    tb.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values
            for mut in offspring:
                if random.random() < 0.1:
                    tb.mutate(mut)
                    del mut.fitness.values
            invalid = [i for i in offspring if not i.fitness.valid]
            for ind in invalid:
                ind.fitness.values = tb.evaluate(ind)
            pop[:] = offspring

    return run


def bench_gp_evograd():
    def run():
        run_gp(GP_X_T, GP_Y_T, pop_size=GP_POP,
               prog_len=16, generations=GP_GENS, device=DEVICE)
    return run


# ═══════════════════════════════════════════════════════════════════════ #
#  Run everything and print the table
# ═══════════════════════════════════════════════════════════════════════ #

def run_benchmark():
    print("\n" + "═" * 78)
    print("  EvoGrad vs DEAP — Benchmark")
    print(f"  Device: {DEVICE.upper()}  |  "
          f"Dims: {N_DIM}  |  Population: {POP}  |  "
          f"Trials: {TRIALS}")
    print("═" * 78)

    configs = [
        ("GA",  POP,    N_GENS, bench_ga_deap,  bench_ga_evograd),
        ("PSO", POP,    N_GENS, bench_pso_deap, bench_pso_evograd),
        ("EDA", POP,    N_GENS, bench_eda_deap, bench_eda_evograd),
        ("ES",  POP,    N_GENS, bench_es_deap,  bench_es_evograd),
        ("GP",  GP_POP, GP_GENS,bench_gp_deap,  bench_gp_evograd),
    ]

    rows = []
    for name, pop, gens, deap_fn, eg_fn in configs:
        print(f"\n  [{name}]  pop={pop}  gens={gens}")

        # warm up (exclude first-call compilation)
        print(f"    warming up DEAP ...", end="", flush=True)
        deap_run = deap_fn()
        deap_run()           # discard
        print(" done")

        print(f"    warming up EvoGrad ...", end="", flush=True)
        eg_run = eg_fn()
        eg_run()             # discard
        print(" done")

        print(f"    timing DEAP ({TRIALS} trials)...", end="", flush=True)
        d_med, d_std = timed(deap_fn())
        print(f"  {d_med*1000:.1f} ms ± {d_std*1000:.1f}")

        print(f"    timing EvoGrad ({TRIALS} trials)...", end="", flush=True)
        e_med, e_std = timed(eg_fn())
        print(f"  {e_med*1000:.1f} ms ± {e_std*1000:.1f}")

        speedup = d_med / e_med
        ms_per_gen_deap = d_med / gens * 1000
        ms_per_gen_eg   = e_med / gens * 1000
        rows.append((name, pop, gens, d_med, d_std, e_med, e_std, speedup,
                     ms_per_gen_deap, ms_per_gen_eg))

    # ── Print table ────────────────────────────────────────────────────
    COL = (10, 5, 5, 16, 16, 13, 13, 10)
    SEP = "─" * sum(COL) + "─" * (len(COL) - 1) * 3

    print("\n\n" + "═" * len(SEP))
    print(f"  EvoGrad vs DEAP  |  device={DEVICE.upper()}  "
          f"prob=Rastrigin-{N_DIM}d (GA/PSO/EDA/ES)  "
          f"SymReg (GP)")
    print("═" * len(SEP))

    hdr = (f"{'Algo':<{COL[0]}}  {'Pop':>{COL[1]}}  {'Gens':>{COL[2]}}  "
           f"{'DEAP total':>{COL[3]}}  {'EvoGrad total':>{COL[4]}}  "
           f"{'DEAP /gen':>{COL[5]}}  {'EG /gen':>{COL[6]}}  "
           f"{'Speedup':>{COL[7]}}")
    print(hdr)
    print(SEP)

    for name, pop, gens, d_med, d_std, e_med, e_std, speedup, mpg_d, mpg_e in rows:
        d_str = f"{d_med*1000:.0f} ± {d_std*1000:.0f} ms"
        e_str = f"{e_med*1000:.0f} ± {e_std*1000:.0f} ms"
        print(
            f"{name:<{COL[0]}}  {pop:>{COL[1]}}  {gens:>{COL[2]}}  "
            f"{d_str:>{COL[3]}}  {e_str:>{COL[4]}}  "
            f"{mpg_d:>{COL[5]-3}.2f} ms  "
            f"{mpg_e:>{COL[6]-3}.3f} ms  "
            f"{speedup:>{COL[7]-1}.1f}×"
        )

    print(SEP)

    # ── Summary line ──────────────────────────────────────────────────
    speedups = [r[7] for r in rows]
    print(f"\n  Geometric mean speedup: "
          f"{math.exp(sum(math.log(s) for s in speedups)/len(speedups)):.1f}×")
    print(f"  Best speedup:  {max(speedups):.1f}×  ({rows[speedups.index(max(speedups))][0]})")
    print(f"  Lowest speedup:{min(speedups):.1f}×  ({rows[speedups.index(min(speedups))][0]})")
    print()


if __name__ == "__main__":
    run_benchmark()
