import math
from typing import Iterable, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer


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
    ) -> None:
        if lr <= 0:
            raise ValueError("Invalid learning rate")
        defaults = dict(lr=lr, epsilon=epsilon, r=r, T=T)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        """Perform a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that re-evaluates the model and returns the loss.  This is
            required because the PSD escape episode may need to re-compute the
            gradients multiple times.
        """
        if closure is None:
            raise RuntimeError("PSDOptimizer requires a closure to evaluate the model")

        loss = closure()
        params = [p for group in self.param_groups for p in group['params'] if p.grad is not None]
        if not params:
            return loss

        grad_norm = torch.sqrt(sum(torch.sum(p.grad.detach() ** 2) for p in params))
        eps = self.param_groups[0]['epsilon']
        lr = self.param_groups[0]['lr']

        if grad_norm > eps:
            for p in params:
                p.add_(p.grad, alpha=-lr)
            return loss

        # Escape episode
        r = self.param_groups[0]['r']
        T = self.param_groups[0]['T']
        for p in params:
            noise = torch.randn_like(p) * r
            p.add_(noise)

        for _ in range(T):
            loss = closure()
            for p in params:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)
        return loss
