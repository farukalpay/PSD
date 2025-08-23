#!/usr/bin/env python
"""Benchmark PSD against SGD and Adam on CIFAR-10.

This script trains a small convolutional neural network on CIFAR-10 using
three optimizers: SGD, Adam and PSDOptimizer. Training and validation loss and
accuracy are recorded for each epoch and plotted at the end of the run.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ensure repo root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from psd_optimizer import PSDOptimizer


@dataclass
class EpochStats:
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_acc: List[float]


def get_dataloaders(batch_size: int, seed: int = 0, data_dir: str = "data") -> Tuple[DataLoader, DataLoader]:
    """Return training and validation dataloaders for CIFAR-10."""
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def build_model() -> nn.Module:
    """Create a small CNN suitable for CIFAR-10."""
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 10),
    )
    return model


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    max_batches: int | None = None,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if max_batches is not None and batch_idx >= max_batches:
            break

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            with torch.enable_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            return loss

        if isinstance(optimizer, PSDOptimizer):
            loss = optimizer.step(closure)
        else:
            loss = closure()
            optimizer.step()

        with torch.no_grad():
            outputs = model(inputs)
            loss_val = criterion(outputs, targets)
            running_loss += loss_val.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    max_batches: int | None = None,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if max_batches is not None and batch_idx >= max_batches:
                break
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / total, correct / total


def run_benchmark(
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> Dict[str, EpochStats]:
    train_loader, val_loader = get_dataloaders(batch_size)
    criterion = nn.CrossEntropyLoss()

    optimizers: Dict[str, optim.Optimizer] = {
        "SGD": lambda params: optim.SGD(params, lr=lr, momentum=0.9),
        "Adam": lambda params: optim.Adam(params, lr=lr),
        "PSD": lambda params: PSDOptimizer(params, lr=lr),
    }

    results: Dict[str, EpochStats] = {}
    for name, optim_ctor in optimizers.items():
        print(f"\n=== Training with {name} ===")
        torch.manual_seed(0)
        model = build_model().to(device)
        optimizer = optim_ctor(model.parameters())
        stats = EpochStats([], [], [], [])

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(
                model,
                optimizer,
                train_loader,
                device,
                criterion,
                max_batches=max_train_batches,
            )
            val_loss, val_acc = evaluate(
                model,
                val_loader,
                device,
                criterion,
                max_batches=max_val_batches,
            )

            stats.train_loss.append(train_loss)
            stats.train_acc.append(train_acc)
            stats.val_loss.append(val_loss)
            stats.val_acc.append(val_acc)

            print(
                f"Epoch {epoch:02d}/{epochs} - "
                f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
            )
        results[name] = stats
    return results


def plot_results(results: Dict[str, EpochStats], out_dir: str, epochs: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sns.set(style="whitegrid")
    x = list(range(1, epochs + 1))

    plt.figure(figsize=(10, 6))
    for name, stats in results.items():
        plt.plot(x, stats.train_loss, label=f"{name} Train")
        plt.plot(x, stats.val_loss, linestyle="--", label=f"{name} Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CIFAR-10 Loss vs Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cifar10_loss.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, stats in results.items():
        plt.plot(x, stats.val_acc, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("CIFAR-10 Validation Accuracy vs Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cifar10_accuracy.png"), dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PSD against SGD and Adam on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs per optimizer")
    parser.add_argument("--batch-size", type=int, default=128, help="training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for all optimizers")
    parser.add_argument("--output-dir", type=str, default="results", help="directory for output plots")
    parser.add_argument("--max-train-batches", type=int, default=None, help="limit training batches (for quick tests)")
    parser.add_argument("--max-val-batches", type=int, default=None, help="limit validation batches (for quick tests)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    results = run_benchmark(
        args.epochs,
        args.batch_size,
        args.lr,
        device,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    plot_results(results, args.output_dir, args.epochs)


if __name__ == "__main__":
    main()
