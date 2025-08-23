from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


class PSDOptimizer(Optimizer):
    """Perturbed Saddle-point Descent (PSD) optimizer.

    This is a simple PyTorch implementation of the PSD algorithm described in
    the accompanying paper.  The optimizer behaves like gradient descent when
    the gradient norm is large.  When the norm falls below ``epsilon`` it
    injects random noise of radius ``r`` and performs ``T`` additional gradient
    steps in an attempt to escape saddle points.

    The implementation requires a ``closure`` function in ``step`` similar to
    :class:`torch.optim.LBFGS` so that gradients can be recomputed multiple
    times during the escape episode.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        epsilon: float = 1e-3,
        r: float = 1e-3,
        T: int = 10,
        max_grad_norm: float | None = 1.0,
        **kwargs: object,
    ) -> None:
        if lr <= 0:
            raise ValueError("Invalid learning rate")
        defaults = dict(lr=lr, epsilon=epsilon, r=r, T=T, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults, **kwargs)

    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor:  # type: ignore[override]
        """Perform a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that re-evaluates the model and returns the loss.  This is
            required because the PSD escape episode may need to re-compute the
            gradients multiple times.

        Returns
        -------
        float
            The loss value evaluated by the provided ``closure``.
        """
        if closure is None:
            raise RuntimeError("PSDOptimizer requires a closure to evaluate the model")

        # Compute loss and gradients via the user-provided closure
        loss = closure()
        params = [p for group in self.param_groups for p in group["params"] if p.grad is not None]
        if not params:
            return loss

        group = self.param_groups[0]
        max_grad_norm = group.get("max_grad_norm")
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

        prev_params = [p.detach().clone() for p in params]

        if any(not torch.isfinite(p.grad).all() for p in params):
            for p, prev in zip(params, prev_params, strict=False):
                p.data.copy_(prev)
            group["lr"] *= 0.5
            logger.warning(
                "Non-finite gradients detected. Reducing learning rate to %s and skipping update.",
                group["lr"],
            )
            return loss

        grad_norm_sq = sum(float(torch.sum(p.grad.detach() ** 2)) for p in params)
        grad_norm = math.sqrt(grad_norm_sq)
        eps = group["epsilon"]
        lr = group["lr"]

        if grad_norm > eps:
            with torch.no_grad():
                for p in params:
                    p.add_(p.grad, alpha=-lr)
            for p, prev in zip(params, prev_params, strict=False):
                if not torch.isfinite(p).all() or not torch.isfinite(p.grad).all():
                    p.data.copy_(prev)
                    group["lr"] *= 0.5
                    logger.warning(
                        "Non-finite values detected. Reducing learning rate to %s and skipping update.",
                        group["lr"],
                    )
                    break
            return loss

        # Escape episode
        r = group["r"]
        T = group["T"]
        with torch.no_grad():
            for p in params:
                noise = torch.randn_like(p) * r
                p.add_(noise)

        for _ in range(T):
            loss = closure()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        prev = p.detach().clone()
                        p.add_(p.grad, alpha=-lr)
                        if not torch.isfinite(p).all() or not torch.isfinite(p.grad).all():
                            p.data.copy_(prev)
                            group["lr"] *= 0.5
                            logger.warning(
                                (
                                    "Non-finite values detected during escape step. "
                                    "Reducing learning rate to %s and skipping update."
                                ),
                                group["lr"],
                            )
                            return loss
        return loss


__all__ = ["PSDOptimizer"]
