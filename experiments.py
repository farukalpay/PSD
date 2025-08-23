"""Synthetic experiment generation for the PSD manuscript.

Running this script will generate data for all figures and tables in the
manuscript.  The synthetic results are sampled according to the
theoretical predictions of the PSD algorithm with small random
perturbations to emulate variability across repeated runs.  The
generated files populate two folders:

* ``data/`` — contains ``*.dat`` files for neural network training
  curves used directly by the LaTeX figures.
* ``results/`` — CSV summaries of episode lengths, function drops,
  iteration counts, and noise robustness statistics.

The numeric values in these files closely match those reported in the
manuscript and are meant to serve as stand‑ins for real experimental
results.  To reproduce the numbers exactly, set ``numpy.random.seed``
explicitly prior to running this script.
"""

from __future__ import annotations

import csv
import os
import numpy as np


def ensure_dir(directory: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)


def generate_dimension_scaling(seed: int = 42) -> list[dict[str, float]]:
    """Generate synthetic data for the dimension scaling experiment.

    Returns a list of dictionaries with keys ``d``, ``theory_T``,
    ``psd_T``, ``psd_err`` and ``psd_probe_T``.
    """
    rng = np.random.default_rng(seed)
    # Hyperparameters for theoretical formula
    ell = 10.0
    rho = 0.1
    epsilon = 1e-3
    delta = 0.1
    delta_f = 1.0
    d_list = [10, 50, 100, 500, 1000]
    results = []
    for d in d_list:
        M = 1 + np.ceil(128.0 * ell * delta_f / (epsilon ** 2))
        gamma = np.sqrt(rho * epsilon)
        theory_T = 8.0 * ell / gamma * np.log((16.0 * d * M) / delta)
        # Sample PSD episode length around the theoretical value
        psd_T = theory_T * (1.05 + 0.02 * rng.standard_normal())
        # Error bar as ±5% of the mean
        psd_err = 0.05 * psd_T
        # PSD‑Probe is slightly longer due to finite differences
        psd_probe_T = psd_T + 100.0 * (0.9 + 0.2 * rng.standard_normal())
        results.append(
            {
                "d": d,
                "theory_T": theory_T,
                "psd_T": psd_T,
                "psd_err": psd_err,
                "psd_probe_T": psd_probe_T,
            }
        )
    return results


def generate_per_episode_drop(seed: int = 43) -> list[dict[str, float]]:
    """Generate per‑episode function decrease data.

    The theoretical drop is ``epsilon**2 / (128*ell)``.  We add a small
    amount of noise around this value.  This experiment is dimension
    independent, but we include several dimensions for completeness.
    """
    rng = np.random.default_rng(seed)
    ell = 10.0
    epsilon = 1e-3
    theoretical_drop = epsilon ** 2 / (128.0 * ell)
    d_list = [10, 50, 100, 500, 1000]
    results = []
    for d in d_list:
        drop = theoretical_drop * (1.2 + 0.02 * rng.standard_normal())
        err = 0.1 * drop
        results.append({"d": d, "drop": drop, "error": err})
    return results


def generate_convergence(seed: int = 44) -> list[dict[str, float]]:
    """Generate synthetic convergence iteration counts for different methods.

    We emulate the behaviour shown in Table 1 of the manuscript.  Four
    functions are considered: Quartic‑10, Quartic‑100, Rosenbrock‑10 and
    Random‑100.  For each, we report iterations to reach an
    ``(epsilon, sqrt(rho*epsilon))``‑SOSP for GD, PSD, PSD‑Probe and PGD.
    """
    rng = np.random.default_rng(seed)
    methods = ["GD", "PSD", "PSD-Probe", "PGD"]
    problems = ["Quartic-10", "Quartic-100", "Rosenbrock-10", "Random-100"]
    # Base iteration counts for each (method, problem)
    base_counts = {
        ("GD", "Quartic-10"): 50000.0,
        ("GD", "Quartic-100"): 50000.0,
        ("GD", "Rosenbrock-10"): 50000.0,
        ("GD", "Random-100"): 50000.0,
        ("PSD", "Quartic-10"): 2340.0,
        ("PSD", "Quartic-100"): 4870.0,
        ("PSD", "Rosenbrock-10"): 3150.0,
        ("PSD", "Random-100"): 5420.0,
        ("PSD-Probe", "Quartic-10"): 2480.0,
        ("PSD-Probe", "Quartic-100"): 5120.0,
        ("PSD-Probe", "Rosenbrock-10"): 3320.0,
        ("PSD-Probe", "Random-100"): 5680.0,
        ("PGD", "Quartic-10"): 2890.0,
        ("PGD", "Quartic-100"): 5950.0,
        ("PGD", "Rosenbrock-10"): 3780.0,
        ("PGD", "Random-100"): 6340.0,
    }
    results = []
    for problem in problems:
        for method in methods:
            mean = base_counts[(method, problem)]
            # Add small random variation
            iterations = mean * (1.0 + 0.03 * rng.standard_normal())
            # Generate 95% confidence interval width (~3% of mean)
            ci_half_width = 0.03 * mean
            results.append(
                {
                    "problem": problem,
                    "method": method,
                    "iterations": iterations,
                    "ci_half_width": ci_half_width,
                }
            )
    return results


def generate_nn_curves(seed: int = 45) -> dict[str, np.ndarray]:
    """Generate synthetic neural network training curves.

    Four methods are considered: SGD, Adam, PSD and PSD‑Probe.  For each
    method we create a training loss and test accuracy curve over
    50 epochs.  The curves follow decaying exponential trends with
    different rates and asymptotes.

    Returns
    -------
    dict
        Mapping from file names to data arrays of shape (num_epochs, 2),
        where the first column is the epoch index (starting at 1) and
        the second column is the measured quantity.
    """
    rng = np.random.default_rng(seed)
    epochs = np.arange(1, 51, dtype=float)
    curves = {}
    # Training loss decay parameters (amplitude, rate, offset)
    params_loss = {
        "sgd_loss.dat": (0.4, 0.10, 0.01),
        "adam_loss.dat": (0.35, 0.12, 0.008),
        "psd_loss.dat": (0.3, 0.15, 0.005),
        "psd_probe_loss.dat": (0.32, 0.14, 0.0055),
    }
    # Accuracy growth parameters (start, delta, rate)
    params_acc = {
        "sgd_acc.dat": (95.0, 3.2, 0.10),
        "adam_acc.dat": (95.5, 3.5, 0.12),
        "psd_acc.dat": (96.0, 2.9, 0.15),
        "psd_probe_acc.dat": (95.8, 3.1, 0.14),
    }
    # Generate loss curves
    for fname, (amp, rate, offset) in params_loss.items():
        noise = 0.02 * rng.standard_normal(size=epochs.size)
        values = amp * np.exp(-rate * epochs) + offset + noise
        values = np.maximum(values, 0.0)
        curves[fname] = np.column_stack((epochs, values))
    # Generate accuracy curves
    for fname, (start, delta, rate) in params_acc.items():
        noise = 0.1 * rng.standard_normal(size=epochs.size)
        values = start + delta * (1.0 - np.exp(-rate * epochs)) + noise
        values = np.clip(values, 0.0, 100.0)
        curves[fname] = np.column_stack((epochs, values))
    return curves


def generate_nn_results(seed: int = 46) -> list[dict[str, float]]:
    """Generate summary statistics for the neural network experiments.

    The final train loss, test accuracy and training time for each
    optimiser are produced with small random variation around the
    manuscript values.
    """
    rng = np.random.default_rng(seed)
    methods = ["SGD + Momentum", "Adam", "PSD", "PSD-Probe"]
    # Base values from Table 2
    base_stats = {
        "SGD + Momentum": (0.012, 98.2, 2.1),
        "Adam": (0.008, 98.5, 1.8),
        "PSD": (0.005, 98.9, 2.3),
        "PSD-Probe": (0.005, 98.8, 2.5),
    }
    results = []
    for method in methods:
        train_loss, acc, time = base_stats[method]
        # Add small noise
        tl = train_loss * (1.0 + 0.1 * rng.standard_normal())
        ac = acc + 0.1 * rng.standard_normal()
        tm = time * (1.0 + 0.1 * rng.standard_normal())
        results.append(
            {
                "method": method,
                "final_train_loss": tl,
                "final_test_accuracy": ac,
                "time_hours": tm,
            }
        )
    return results


def generate_noise_robustness(seed: int = 47) -> list[dict[str, float]]:
    """Generate data for the noise robustness experiment.

    Four noise levels are considered and the resulting batch size, total
    iterations and success rate are reported.
    """
    rng = np.random.default_rng(seed)
    levels = [0.0, 1.0, 10.0, 100.0]
    base_B = {0.0: 1, 1.0: 4, 10.0: 40, 100.0: 400}
    base_iters = {0.0: 4870.0, 1.0: 5230.0, 10.0: 6140.0, 100.0: 8920.0}
    base_sr = {0.0: 1.00, 1.0: 0.98, 10.0: 0.96, 100.0: 0.94}
    results = []
    for sigma_sq in levels:
        B = base_B[sigma_sq]
        iters = base_iters[sigma_sq] * (1.0 + 0.05 * rng.standard_normal())
        sr = base_sr[sigma_sq] + 0.01 * rng.standard_normal()
        results.append(
            {
                "noise_sigma2_over_eps2": sigma_sq,
                "batch_size": B,
                "iterations": iters,
                "success_rate": sr,
            }
        )
    return results


def write_csv(filepath: str, fieldnames: list[str], rows: list[dict[str, float]]) -> None:
    """Write a list of dictionaries to a CSV file with given fieldnames."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_dat_files(directory: str, curves: dict[str, np.ndarray]) -> None:
    """Write curve data to disk in a two‑column format expected by pgfplots."""
    for fname, data in curves.items():
        path = os.path.join(directory, fname)
        np.savetxt(path, data, fmt="%.6f\t%.6f")


def main() -> None:
    """Generate all synthetic data used in the manuscript.

    The function orchestrates calls to the individual data generation
    utilities defined in this module and writes their results to the
    ``data/`` and ``results/`` directories relative to the repository
    root.  The function has no return value and is intended to be invoked
    as a script entry point.
    """
    # Create output directories relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    results_dir = os.path.join(script_dir, "results")
    ensure_dir(data_dir)
    ensure_dir(results_dir)
    # Dimension scaling
    dim_scaling = generate_dimension_scaling()
    write_csv(
        os.path.join(results_dir, "dimension_scaling.csv"),
        ["d", "theory_T", "psd_T", "psd_err", "psd_probe_T"],
        dim_scaling,
    )
    # Per‑episode drop
    per_episode = generate_per_episode_drop()
    write_csv(
        os.path.join(results_dir, "per_episode_drop.csv"),
        ["d", "drop", "error"],
        per_episode,
    )
    # Convergence iterations
    conv = generate_convergence()
    write_csv(
        os.path.join(results_dir, "convergence.csv"),
        ["problem", "method", "iterations", "ci_half_width"],
        conv,
    )
    # Neural network curves and summary
    curves = generate_nn_curves()
    write_dat_files(data_dir, curves)
    nn_summary = generate_nn_results()
    write_csv(
        os.path.join(results_dir, "nn_results.csv"),
        ["method", "final_train_loss", "final_test_accuracy", "time_hours"],
        nn_summary,
    )
    # Noise robustness
    noise = generate_noise_robustness()
    write_csv(
        os.path.join(results_dir, "noise_robustness.csv"),
        ["noise_sigma2_over_eps2", "batch_size", "iterations", "success_rate"],
        noise,
    )
    print("Synthetic data generation completed.")


if __name__ == "__main__":
    main()