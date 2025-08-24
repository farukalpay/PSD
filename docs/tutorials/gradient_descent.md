# Gradient Descent Tutorial

This tutorial demonstrates the reference `gradient_descent` optimiser on the separable quartic function.

```python
import numpy as np
from psd import algorithms, functions

x0 = np.array([1.0, -1.0])
x_star, iters = algorithms.gradient_descent(x0, functions.SEPARABLE_QUARTIC.grad, step_size=0.1)
print(f"x*: {x_star}")
print(f"iterations: {iters}")
```

Expected output:

```
x*: [ 0.7071069 -0.7071069]
iterations: 27
```
