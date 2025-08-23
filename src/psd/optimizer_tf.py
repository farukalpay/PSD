"""TensorFlow implementation of the Perturbed Saddle-escape Descent (PSD) optimizer.

This module provides :class:`PSDTensorFlow`, a drop-in replacement for standard
Keras optimizers that mirrors the behaviour of the reference PyTorch
implementation.  The optimizer performs gradient descent steps and injects
random perturbations when the global gradient norm becomes small, helping
escape saddle points.
"""

from __future__ import annotations

# mypy: ignore-errors
from collections.abc import Iterable

import tensorflow as tf


class PSDTensorFlow(tf.keras.optimizers.Optimizer):  # type: ignore[misc]
    """Perturbed Saddle-escape Descent optimizer for TensorFlow/Keras.

    Parameters
    ----------
    learning_rate: float, default 1e-2
        Step size used for the gradient descent update.
    g_thres: float, default 1e-2
        Gradient-norm threshold that triggers a perturbation episode.
    t_thres: int, default 50
        Minimum number of steps between two perturbation episodes.
    r: float, default 1e-2
        Radius of the isotropic perturbation injected when escaping
        saddle points.
    name: str, default "PSDTensorFlow"
        Optional name for the optimizer.
    """

    def __init__(
        self,
        learning_rate: float = 1e-2,
        g_thres: float = 1e-2,
        t_thres: int = 50,
        r: float = 1e-2,
        name: str = "PSDTensorFlow",
        **kwargs: object,
    ) -> None:
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("g_thres", g_thres)
        self._set_hyper("t_thres", float(t_thres))
        self._set_hyper("r", r)
        # ``t_noise`` stores the iteration index of the last perturbation.
        self.t_noise = self.add_weight(
            name="t_noise",
            dtype=tf.int64,
            trainable=False,
            initializer=tf.constant_initializer(-t_thres - 1),
        )

    def _resource_apply_dense(
        self,
        grad: tf.Tensor,
        var: tf.Variable,
        apply_state: dict[str, object] | None = None,
    ) -> None:
        lr_t = tf.cast(self._decayed_lr(var.dtype), var.dtype)
        var.assign_sub(grad * lr_t)

    def _resource_apply_sparse(
        self,
        grad: tf.Tensor,
        var: tf.Variable,
        indices: tf.Tensor,
        apply_state: dict[str, object] | None = None,
    ) -> None:
        lr_t = tf.cast(self._decayed_lr(var.dtype), var.dtype)
        var.scatter_sub(tf.IndexedSlices(grad * lr_t, indices))

    def apply_gradients(
        self,
        grads_and_vars: Iterable[tuple[tf.Tensor, tf.Variable]],
        name: str | None = None,
        **kwargs: object,
    ) -> None:
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        if not grads_and_vars:
            super().apply_gradients(grads_and_vars, name, **kwargs)
            return None

        grads, vars = zip(*grads_and_vars, strict=False)

        # Compute global gradient norm across all tensors.
        sq_norms: list[tf.Tensor] = []
        for g in grads:
            if isinstance(g, tf.IndexedSlices):
                sq_norms.append(tf.reduce_sum(tf.square(g.values)))
            else:
                sq_norms.append(tf.reduce_sum(tf.square(g)))
        global_norm = tf.sqrt(tf.add_n(sq_norms))

        g_thres = self._get_hyper("g_thres")
        t_thres = tf.cast(self._get_hyper("t_thres"), tf.int64)
        r = self._get_hyper("r")

        step = tf.identity(self.iterations)
        need_perturb = tf.logical_and(global_norm <= g_thres, (step - self.t_noise) > t_thres)

        def perturb_vars() -> None:
            for v in vars:
                v_dtype = v.dtype.base_dtype
                noise = tf.random.normal(tf.shape(v), dtype=v_dtype)
                noise /= tf.norm(noise) + tf.constant(1e-8, dtype=v_dtype)
                u = tf.random.uniform([], dtype=v_dtype)
                radius = tf.pow(u, 1.0 / tf.cast(tf.size(v), v_dtype))
                v.assign_add(noise * tf.cast(r, v_dtype) * radius)
            self.t_noise.assign(step)

        tf.cond(need_perturb, perturb_vars, lambda: None)

        super().apply_gradients(grads_and_vars, name, **kwargs)
        return None

    def get_config(self) -> dict[str, object]:
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


__all__ = ["PSDTensorFlow"]
