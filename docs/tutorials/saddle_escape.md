# Saddle Escape with PSD

This tutorial compares basic gradient descent with the `psd` algorithm when starting at a saddle point of the separable quartic function.

```python
import numpy as np
from psd import algorithms, functions

x0 = np.zeros(2)

# Gradient descent becomes stuck at the saddle.
gd_x, _ = algorithms.gradient_descent(x0, functions.SEPARABLE_QUARTIC.grad, step_size=0.1, max_iter=10)
print(f"gradient descent result: {gd_x}")

# PSD adds noise and escapes to a nearby minimum.
psd_x, _ = algorithms.psd(
    x0,
    functions.SEPARABLE_QUARTIC.grad,
    functions.SEPARABLE_QUARTIC.hess,
    epsilon=0.1,
    ell=1.0,
    rho=1.0,
    max_iter=1000,
)
print(f"psd result: {psd_x}")
```

Expected output:

```
gradient descent result: [0. 0.]
psd result: [-0.69796542  0.69858143]
```
