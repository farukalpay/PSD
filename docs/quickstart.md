# Quick Start

The PSD package can be used with only a few lines of code. The example below
optimises a simple quartic function using the reference gradient-descent
implementation.

```python
import numpy as np
from psd import algorithms, functions

x0 = np.array([1.0, -1.0])
x_star, _ = algorithms.gradient_descent(x0, functions.SEPARABLE_QUARTIC.grad)
print(x_star)
```

For more elaborate examples, see {doc}`tutorials/mnist` or the
`examples/` directory in the repository.

