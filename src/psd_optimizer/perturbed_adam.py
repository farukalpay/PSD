from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class PerturbedAdam(Optimizer):
    r"""Adam optimizer with gradient-based perturbations to escape saddle points.

    This optimizer merges the adaptive moment estimation of :class:`~torch.optim.Adam`
    with the perturbation mechanism of Perturbed Gradient Descent (PGD).  The
    standard Adam update is computed using first- and second-moment estimates of
    the gradients.  When the norm of the *effective* Adam step falls below
    ``g_thres`` for ``t_thres`` consecutive steps, an isotropic perturbation of
    radius ``r`` is injected directly into the parameters.  After a perturbation
    the moment estimates are reset to zero so that the large gradient following
    the jump does not contaminate the momentum or adaptive learning rates.

    Parameters
    ----------
    params : iterable of :class:`torch.nn.Parameter`
        Parameters to optimize.
    lr : float, optional
        Learning rate (default: ``1e-3``).
    betas : Tuple[float, float], optional
        Coefficients used for computing running averages of gradient and its
        square (default: ``(0.9, 0.999)``).
    eps : float, optional
        Term added to the denominator to improve numerical stability
        (default: ``1e-8``).
    g_thres : float, optional
        Threshold on the norm of the Adam update below which the optimizer
        considers that it may be near a saddle point (default: ``1e-3``).
    t_thres : int, optional
        Number of consecutive small-update steps required before a perturbation
        is triggered (default: ``10``).
    r : float, optional
        Radius of the isotropic Gaussian noise injected during a perturbation
        (default: ``1e-3``).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        g_thres: float = 1e-3,
        t_thres: int = 10,
        r: float = 1e-3,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if g_thres <= 0:
            raise ValueError("g_thres must be positive")
        if t_thres <= 0:
            raise ValueError("t_thres must be positive")
        if r < 0:
            raise ValueError("r must be non-negative")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            g_thres=g_thres,
            t_thres=t_thres,
            r=r,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:  # type: ignore[override]
        """Perform a single optimization step.

        The method follows the usual Adam update rule.  Additionally, it monitors
        the norm of the bias-corrected Adam step.  If this norm remains below
        ``g_thres`` for more than ``t_thres`` steps since the last perturbation or
        large gradient, the parameters are perturbed with isotropic Gaussian
        noise and the internal moment estimates are reset to zero.
        """

        loss: Tensor | None = None
        if closure is not None:
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            g_thres = group["g_thres"]
            t_thres = group["t_thres"]
            r = group["r"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["t"] = 0
                    state["t_noise"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                state["t"] += 1

                # Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                denom = v_hat.sqrt().add_(eps)
                adam_step = m_hat / denom
                step_norm = adam_step.norm().item()

                # Perturbation condition based on effective Adam update
                if step_norm <= g_thres and state["t"] - state["t_noise"] >= t_thres:
                    noise = torch.randn_like(p) * r
                    p.add_(noise)
                    exp_avg.zero_()
                    exp_avg_sq.zero_()
                    state["t_noise"] = state["t"]
                else:
                    p.add_(adam_step, alpha=-lr)
                    if step_norm > g_thres:
                        # reset t_noise when far from saddle point
                        state["t_noise"] = state["t"]

        return loss


__all__ = ["PerturbedAdam"]
