"""
HybridTrainer — combines evolutionary search with gradient descent.

Phase 1 (evolutionary):  EvoEngine explores the weight landscape,
                          ignoring non-differentiable choices.
Phase 2 (gradient):      Adam/SGD refines the best individual found
                          by Phase 1 using backpropagation.

This is especially useful for:
  - Neural Architecture Search (NAS): evolve architecture decisions
    that are non-differentiable, then gradient-train the weights.
  - Ensemble search: evolve ensemble compositions, then fine-tune.
  - Saddle-point escape: evolutionary warm-start avoids local minima
    that gradient descent falls into from random init.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple
import torch
import torch.nn as nn

from ..core.tensor import Tensor
from ..evo.population import Population
from ..evo.engine import EvoEngine, EvoConfig
from ..evo.fitness import NetworkFitness


@dataclass
class HybridConfig:
    # evolutionary phase
    evo_generations: int = 50
    pop_size: int = 64
    mutation_sigma: float = 0.02
    selection: str = "tournament"
    crossover: str = "sbx"

    # gradient phase
    gradient_optimizer: str = "adam"    # adam | sgd | rmsprop
    gradient_lr: float = 1e-3
    gradient_epochs: int = 10
    weight_decay: float = 1e-4

    # misc
    device: str = "cpu"
    verbose: bool = True


class HybridTrainer:
    """
    Run evolutionary search to warm-start a neural network, then
    fine-tune with gradient descent.

    Parameters
    ----------
    model       : nn.Module (built by codegen or user-supplied)
    config      : HybridConfig
    loss_fn     : callable(output_tensor, target_tensor) → scalar Tensor
    """

    def __init__(self, model: nn.Module, config: HybridConfig,
                 loss_fn: Optional[Callable] = None):
        self._model = model.to(config.device)
        self._cfg = config
        self._loss_fn = loss_fn or (
            lambda out, y: Tensor(
                torch.nn.functional.cross_entropy(out, y.data.long())
            )
        )

    # ------------------------------------------------------------------ #
    # Phase 1: evolutionary warm-start
    # ------------------------------------------------------------------ #

    def evolutionary_phase(
        self,
        X: Tensor,
        y: Tensor,
    ) -> Tuple[Population, List]:
        cfg = self._cfg
        params = list(self._model.parameters())
        genome_length = sum(p.numel() for p in params)

        pop = Population.from_network_weights(
            params, pop_size=cfg.pop_size,
            noise_std=cfg.mutation_sigma * 5,
            device=cfg.device,
        )

        fitness_fn = NetworkFitness(
            model_factory=lambda: _ModelWrapper(self._model),
            X=X, y=y, metric="accuracy",
        )

        evo_cfg = EvoConfig(
            generations=cfg.evo_generations,
            selection=cfg.selection,
            crossover=cfg.crossover,
            mutation="gaussian",
            mutation_sigma=cfg.mutation_sigma,
            elite_size=2,
            verbose=cfg.verbose,
            log_every=max(1, cfg.evo_generations // 10),
        )
        engine = EvoEngine(evo_cfg)
        if cfg.verbose:
            print(f"\n[EvoGrad] Evolutionary phase: "
                  f"{cfg.pop_size} individuals × {cfg.evo_generations} gen")
        pop, logs = engine.run(pop, fitness_fn)

        # Load best genome back into model
        best = pop.best().data
        offset = 0
        with torch.no_grad():
            for param in params:
                n = param.numel()
                param.data.copy_(best[offset:offset + n].reshape(param.shape))
                offset += n

        if cfg.verbose:
            print(f"[EvoGrad] Best evolutionary fitness: "
                  f"{pop.best_fitness():.4f}")
        return pop, logs

    # ------------------------------------------------------------------ #
    # Phase 2: gradient refinement
    # ------------------------------------------------------------------ #

    def gradient_phase(
        self,
        dataloader: Iterator[Tuple[Tensor, Tensor]],
    ) -> List[float]:
        cfg = self._cfg
        params = list(self._model.parameters())

        if cfg.gradient_optimizer == "adam":
            opt = torch.optim.Adam(params, lr=cfg.gradient_lr,
                                    weight_decay=cfg.weight_decay)
        elif cfg.gradient_optimizer == "sgd":
            opt = torch.optim.SGD(params, lr=cfg.gradient_lr,
                                   momentum=0.9,
                                   weight_decay=cfg.weight_decay)
        else:
            opt = torch.optim.RMSprop(params, lr=cfg.gradient_lr,
                                       weight_decay=cfg.weight_decay)

        if cfg.verbose:
            print(f"\n[EvoGrad] Gradient phase: "
                  f"{cfg.gradient_epochs} epochs, {cfg.gradient_optimizer}")

        losses = []
        for epoch in range(cfg.gradient_epochs):
            epoch_loss = 0.0
            batches = 0
            for X_batch, y_batch in dataloader:
                X_b = X_batch.to(cfg.device)
                y_b = y_batch.to(cfg.device)
                # enable gradient tracking for this pass
                x_t = X_b.data.requires_grad_(False)
                out_t = self._model(x_t)
                loss = torch.nn.functional.cross_entropy(
                    out_t, y_b.data.long()
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                batches += 1
            avg = epoch_loss / max(batches, 1)
            losses.append(avg)
            if cfg.verbose:
                print(f"  epoch {epoch + 1:3d}/{cfg.gradient_epochs} "
                      f"| loss={avg:.5f}")
        return losses

    # ------------------------------------------------------------------ #
    # Convenience: run both phases
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: Tensor,
        y_train: Tensor,
        dataloader: Optional[Iterator] = None,
    ) -> dict:
        evo_pop, evo_logs = self.evolutionary_phase(X_train, y_train)

        grad_losses: List[float] = []
        if dataloader is not None:
            grad_losses = self.gradient_phase(dataloader)
        elif self._cfg.gradient_epochs > 0:
            # Wrap (X_train, y_train) into a single-batch dataloader
            single_batch = [(X_train, y_train)]
            grad_losses = self.gradient_phase(iter(single_batch * self._cfg.gradient_epochs))

        return {
            "evo_logs": evo_logs,
            "grad_losses": grad_losses,
            "final_model": self._model,
        }

    @property
    def model(self) -> nn.Module:
        return self._model


class _ModelWrapper:
    """Thin wrapper so NetworkFitness can call model(X) → Tensor."""
    def __init__(self, model: nn.Module):
        self._model = model

    def __call__(self, X: Tensor) -> Tensor:
        with torch.no_grad():
            return Tensor(self._model(X.data))

    def parameters(self) -> list:
        return list(self._model.parameters())
