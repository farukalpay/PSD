# Perturbed Saddle-escape Descent (PSD)

## Project Summary

This repository implements the **Perturbed Saddle-escape Descent (PSD)**
algorithm for escaping saddle points in non-convex optimisation problems.
It contains reference NumPy implementations, framework specific optimisers
for PyTorch and TensorFlow, and utilities for reproducing the synthetic
experiments reported in the accompanying manuscript.

## Features

* Reference implementations of PSD, PSD-Probe and baseline gradient descent
  variants in pure NumPy.
* Suite of analytic test functions with gradients and Hessians.
* Synthetic data generator producing the tables and figures used in the
  paper (`experiments.py`).
* Framework specific optimisers: `PSDTorch`, `PSDTensorFlow` and a
  `PSDOptimizer`/`PerturbedAdam` package for PyTorch.
* Example training scripts for MNIST and CIFAR-10.

## Technology Stack

The core project depends on the following libraries:

| Library | Purpose |
| ------- | ------- |
| `numpy` | numerical routines for reference implementations |
| `torch`, `torchvision` | deep-learning framework and datasets |
| `optuna` | hyper-parameter search utilities |
| `matplotlib` | visualisation in notebooks |

Python 3.8 or later is required.

## Installation

Install the published optimiser package:

```bash
pip install psd-optimizer
```

Or install the repository in editable mode for development:

```bash
git clone https://github.com/farukalpay/PSD.git
cd PSD
pip install -e ".[dev]"
```

## Usage

### Using the Reference Algorithms

The core PSD routines and test functions can be imported from the
``psd`` package:

```python
import numpy as np
from psd import algorithms, functions

x0 = np.array([1.0, -1.0])
x_star, _ = algorithms.gradient_descent(x0, functions.SEPARABLE_QUARTIC.grad)
```

This structure allows you to experiment with the reference NumPy
implementations directly in your projects.

### Generating Synthetic Data

```bash
python experiments.py
```

The command writes CSV summaries to `results/` and training curves to
`data/`.

### Training with the PyTorch Optimiser

```python
from psd_optimizer import PSDOptimizer

model = ...
opt = PSDOptimizer(model.parameters(), lr=1e-3)

def closure():
    opt.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    return loss

opt.step(closure)
```

Example scripts using this API are available in the `notebooks/`
directory.

### Training a Small Language Model

An illustrative example for fine-tuning a compact transformer with
``PSDOptimizer`` is provided in ``scripts/train_small_language_model.py``.
The script downloads a tiny GPT-style model from the Hugging Face Hub and
optimizes it on a short dummy corpus.

Run the example with default settings:

```bash
python scripts/train_small_language_model.py
```

Specify a different pretrained model and number of epochs:

```bash
python scripts/train_small_language_model.py --model distilgpt2 --epochs 5
```

## Documentation

Further materials are available:

* `notebooks/10_minute_start.ipynb` – an interactive notebook showcasing the optimiser.
* `docs/section_1_5_extension.md` – theoretical notes on extending PSD to stochastic settings.
* `notebooks/navigation.ipynb` – links to all example notebooks including `advanced_usage.ipynb`.

## Testing

After installing the repository in editable mode, run the test suite to
verify that everything works:

```bash
pytest
```

The current suite is small but helps prevent regressions.

## Repository Structure

```
psd/              # Reference implementations and framework-specific optimisers
    algorithms.py # PSD and baseline algorithms
    functions.py  # Analytic test functions and registry
psd_optimizer/    # PyTorch optimiser package
experiments.py    # Synthetic data generation
```

## Contributing

Contributions are welcome!  Please open an issue or pull request on GitHub
and see `CONTRIBUTING.md` for guidelines.  By participating you agree to
abide by the `CODE_OF_CONDUCT.md`.

## License

This project is released under the MIT License.  See `LICENSE` for details.

