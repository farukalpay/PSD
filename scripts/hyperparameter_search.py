#!/usr/bin/env python
"""Hyperparameter search for :class:`~psd_optimizer.PSDOptimizer` using Optuna.

This script runs a Bayesian optimization study on the hyperparameters of
``PSDOptimizer`` when training a simple convolutional neural network on the
MNIST dataset.  The study searches over learning rate, gradient threshold,
escape episode length and perturbation radius.  Optuna's pruning is used to
stop unpromising trials early and two summary plots are saved to the
``results`` directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Ensure repository root on path for local imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from psd_optimizer import PSDOptimizer


BATCH_SIZE = 128
EPOCHS = 5
DATA_DIR = "data"


def build_model() -> nn.Module:
    """Return a small convolutional neural network for MNIST."""

    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 10),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate ``model`` on ``loader`` and return accuracy."""

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total


def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna hyperparameter search."""

    # Hyperparameter search space
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    g_thres = trial.suggest_float("g_thres", 1e-3, 1e-1, log=True)
    t_thres = trial.suggest_int("t_thres", 20, 200)
    r = trial.suggest_float("r", 1e-3, 1e-1, log=True)

    # Data loaders
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    n_val = 10_000
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = PSDOptimizer(model.parameters(), lr=lr, epsilon=g_thres, T=t_thres, r=r)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss

            optimizer.step(closure)

        val_acc = evaluate(model, val_loader, device)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_acc


def main() -> None:
    """Run the Optuna study and report results."""

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=50)

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial value: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plot_optimization_history(study).savefig(results_dir / "opt_history.png")
    plot_param_importances(study).savefig(results_dir / "param_importances.png")


if __name__ == "__main__":
    main()

