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
pip install -e .
```

## Usage

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

Example scripts using this API are available in the `examples/`
directory.

## Repository Structure

```
algorithms.py     # Reference implementations of PSD and baselines
functions.py      # Analytic test functions and registry
experiments.py    # Synthetic data generation
psd_optimizer/    # PyTorch optimiser package
psd/              # Framework specific optimisers for Torch and TensorFlow
```

## Contributing

Contributions are welcome!  Please open an issue or pull request on GitHub
and see `CONTRIBUTING.md` for guidelines.  By participating you agree to
abide by the `CODE_OF_CONDUCT.md`.

## License

This project is released under the MIT License.  See `LICENSE` for details.

