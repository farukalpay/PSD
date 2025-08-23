"""Framework-specific implementations of the Perturbed Saddle-point Descent (PSD) optimizer.

This module provides :class:`PSDTorch` and :class:`PSDTensorFlow` which implement the
PSD algorithm for PyTorch and TensorFlow respectively.  The optimizers follow the
native APIs of the underlying frameworks and are designed to be drop-in
replacements for standard optimizers.

Both implementations share the same high level behaviour:

* When the global gradient norm is larger than ``g_thres`` the optimizers perform
  a standard gradient descent step.
* If the norm falls below ``g_thres`` and it has been more than ``t_thres`` steps
  since the last perturbation, a random perturbation of radius ``r`` is added to
  the parameters before taking a gradient step.

The classes are intentionally lightweight and avoid any framework specific helper
utilities so they can be easily audited or extended.

Example
-------
PyTorch::

    import torch
    from psd.framework_optimizers import PSDTorch

    model = torch.nn.Linear(10, 1)
    opt = PSDTorch(model.parameters(), lr=1e-2, g_thres=1e-3, t_thres=10, r=1e-3)

    for input, target in data_loader:
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(input), target)
        loss.backward()
        opt.step()

TensorFlow::

    import tensorflow as tf
    from psd.framework_optimizers import PSDTensorFlow

    model = tf.keras.Sequential([...])
    opt = PSDTensorFlow(learning_rate=1e-2, g_thres=1e-3, t_thres=10, r=1e-3)
    model.compile(optimizer=opt, loss="mse")
    model.fit(x, y)
"""

from __future__ import annotations

# mypy: ignore-errors
import logging
import math
from collections.abc import Callable, Iterable

try:  # Optional import for environments without PyTorch
    import torch
except Exception:  # pragma: no cover - fallback when torch is unavailable
    torch = None  # type: ignore[assignment]

try:  # Optional import for environments without TensorFlow
    import tensorflow as tf
except Exception:  # pragma: no cover - fallback when tensorflow is unavailable
    tf = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class PSDTorch(torch.optim.Optimizer):  # type: ignore[misc]
    """Perturbed Saddle-point Descent optimizer for PyTorch.

    Parameters
    ----------
    params:
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr:
        Base learning rate for parameter updates.
    g_thres:
        Gradient-norm threshold that triggers a perturbation episode.
    t_thres:
        Minimum number of steps between two perturbation episodes.
    r:
        Standard deviation of the isotropic Gaussian noise used for
        perturbations.

    Notes
    -----
    * Each parameter group may override ``lr``, ``g_thres``, ``t_thres`` and ``r``.
    * Per-parameter state ``t`` and ``t_noise`` is stored in ``optimizer.state``
      so the optimizer can be checkpointed with :func:`torch.save` and restored
      with :func:`torch.load`.
    * The global gradient norm for each parameter group is computed in a
      memory-efficient way without materialising large intermediate tensors.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        g_thres: float = 1e-3,
        t_thres: int = 10,
        r: float = 1e-3,
        max_grad_norm: float | None = 1.0,
        **kwargs: object,
    ) -> None:
        if torch is None:  # pragma: no cover - ensures clear error if torch missing
            raise ImportError("PyTorch is required to use PSDTorch")
        if lr <= 0:
            raise ValueError("Invalid learning rate")
        if g_thres < 0 or r < 0:
            raise ValueError("g_thres and r must be non-negative")
        if t_thres < 0:
            raise ValueError("t_thres must be non-negative")

        defaults = dict(lr=lr, g_thres=g_thres, t_thres=t_thres, r=r, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults, **kwargs)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that re-evaluates the model and returns the loss.
            This is provided for API compatibility with other PyTorch
            optimizers and is only executed if supplied.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: list[torch.nn.Parameter] = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            max_grad_norm = group.get("max_grad_norm")
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

            prev_params = [p.clone() for p in params]

            # Ensure state dictionaries exist for all parameters in the group.
            states = []
            for p in params:
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["t_noise"] = -math.inf
                states.append(state)

            # Efficient global gradient norm: sum of squared norms.
            grad_norm_sq = 0.0
            for p in params:
                grad = p.grad.detach()
                grad_norm_sq += float(torch.sum(grad * grad))
            grad_norm = math.sqrt(grad_norm_sq)

            g_thres = group["g_thres"]
            t_thres = group["t_thres"]
            r = group["r"]
            lr = group["lr"]

            # Determine whether to perturb.
            need_perturb = grad_norm <= g_thres and all(state["t"] - state["t_noise"] > t_thres for state in states)

            if need_perturb:
                for p, state in zip(params, states, strict=False):
                    noise = torch.randn_like(p) * r
                    p.add_(noise)
                    state["t_noise"] = state["t"]

            # Gradient descent update and step count.
            for p, state in zip(params, states, strict=False):
                p.add_(p.grad, alpha=-lr)
                state["t"] += 1

            # Numerical stability check
            for p, prev in zip(params, prev_params, strict=False):
                if not torch.isfinite(p).all() or not torch.isfinite(p.grad).all():
                    p.copy_(prev)
                    group["lr"] *= 0.5
                    logger.warning(
                        "Non-finite values detected. Reducing learning rate to %s and skipping update.",
                        group["lr"],
                    )
                    break

        return loss


class PSDTensorFlow(tf.keras.optimizers.Optimizer):  # type: ignore[misc]
    """Perturbed Saddle-point Descent optimizer for TensorFlow.

    This class follows the Keras optimizer API and supports serialization
    through :meth:`get_config`.

    Parameters
    ----------
    learning_rate:
        Step size used for the gradient descent update.
    g_thres:
        Gradient-norm threshold that triggers a perturbation episode.
    t_thres:
        Minimum number of steps between two perturbation episodes.
    r:
        Standard deviation of the isotropic Gaussian noise used for
        perturbations.
    name:
        Optional name for the optimizer.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        g_thres: float = 1e-3,
        t_thres: int = 10,
        r: float = 1e-3,
        name: str = "PSDTensorFlow",
        **kwargs: object,
    ) -> None:
        if tf is None:  # pragma: no cover - ensures clear error if TF missing
            raise ImportError("TensorFlow is required to use PSDTensorFlow")
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("g_thres", g_thres)
        self._set_hyper("t_thres", float(t_thres))
        self._set_hyper("r", r)

    def _create_slots(self, var_list: list[tf.Variable]) -> None:  # pragma: no cover - TF specific
        for var in var_list:
            # ``t_noise`` stores the iteration of the last perturbation.  It is
            # initialised to a very negative value so that a perturbation is
            # immediately allowed if the gradient norm condition is met.
            self.add_slot(var, "t_noise", initializer=tf.constant_initializer(-1.0e9))

    @tf.function
    def apply_gradients(
        self,
        grads_and_vars: Iterable[tuple[tf.Tensor, tf.Variable]],
        name: str | None = None,
        **kwargs: object,
    ) -> None:  # pragma: no cover - TF specific
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        if not grads_and_vars:
            return None

        grads, vars = zip(*grads_and_vars, strict=False)
        global_norm = tf.linalg.global_norm(grads)

        # Cached tensor versions of hyper-parameters for efficiency.
        g_thres = self._get_hyper("g_thres")
        t_thres = self._get_hyper("t_thres")
        r = self._get_hyper("r")

        for grad, var in grads_and_vars:
            var_dtype = var.dtype.base_dtype
            lr_t = tf.cast(self._decayed_lr(var_dtype), var_dtype)
            g_thres_t = tf.cast(g_thres, var_dtype)
            t_thres_t = tf.cast(t_thres, var_dtype)
            r_t = tf.cast(r, var_dtype)
            t = tf.cast(self.iterations, var_dtype)

            t_noise = self.get_slot(var, "t_noise")
            should_perturb = tf.logical_and(global_norm <= g_thres_t, (t - t_noise) > t_thres_t)

            noise = tf.random.normal(tf.shape(var), stddev=r_t, dtype=var_dtype)
            perturb = tf.where(should_perturb, noise, tf.zeros_like(var))
            var.assign_add(perturb)
            t_noise.assign(tf.where(should_perturb, t, t_noise))
            var.assign_sub(lr_t * grad)

        self.iterations.assign_add(1)

    def get_config(self) -> dict[str, object]:  # pragma: no cover - TF specific
        config: dict[str, object] = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "g_thres": self._serialize_hyperparameter("g_thres"),
                "t_thres": self._serialize_hyperparameter("t_thres"),
                "r": self._serialize_hyperparameter("r"),
            }
        )
        return config


__all__ = ["PSDTorch", "PSDTensorFlow"]
