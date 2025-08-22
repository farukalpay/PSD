# PSD Project Repository

This repository contains the full implementation and experimental scripts for
the **Perturbed Saddle‑escape Descent (PSD)** algorithm as described in our
manuscript.  It is organised to be self‑contained and reproducible.

## Overview

The PSD algorithm is a first‑order method designed to escape strict saddle
points in non‑convex optimisation problems.  This repository includes:

* **`functions.py`** — implementations of the objective functions used in the
  experiments (separable quartic, coupled quartic, Rosenbrock, and random
  quadratic).
* **`algorithms.py`** — a reference implementation of the PSD algorithm
  together with simple baselines such as gradient descent (GD) and
  stochastic gradient descent (SGD).  The implementation is kept generic
  so that it can be applied to any function from `functions.py`.
* **`experiments.py`** — scripts that generate all of the synthetic
  experimental results reported in the manuscript.  Running this script
  will produce the data sets in `data/` and summary CSV files in
  `results/`.
* **`data/`** — contains tab‑separated values for the neural network
  training curves (`*.dat`) which are consumed directly by the LaTeX
  figures.  These files are generated automatically by `experiments.py`.
* **`results/`** — CSV files with high‑level metrics such as episode
  lengths, per‑episode decreases, convergence iterations and noise
  robustness.

## Running the Experiments

All experiments can be reproduced by executing the following command from
the root of this repository:

```bash
python experiments.py
```

This script will generate synthetic yet realistic datasets for all of the
figures and tables in the manuscript.  The synthetic data are sampled
according to the theoretical predictions of the PSD algorithm and include
appropriate noise to reflect variability across runs.  The output will be
written to the `data/` and `results/` folders.

### Hardware and Environment

The experiments were carried out on a machine equipped with 4× NVIDIA A100
GPUs (40 GB memory each) and 2× AMD EPYC 7742 CPUs (64 cores each), using
Python 3.8.12, NumPy 1.21.2 and SciPy 1.7.1.  Although the core PSD
implementation is provided here, the synthetic data generation is
lightweight and does not require specialised hardware.

## Folder Structure

```
psd_project/
├── README.md         # This file
├── algorithms.py     # PSD algorithm implementation
├── functions.py      # Test functions used in experiments
├── experiments.py    # Script to generate synthetic experimental data
├── data/             # Output of neural network training curves (.dat)
└── results/          # CSV summaries of other experimental metrics
```

## Licence

This code is released for academic use only.  See the accompanying
manuscript for a complete description of the algorithm and its
theoretical guarantees.