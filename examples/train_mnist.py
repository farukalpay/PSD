import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from psd_optimizer import PSDOptimizer


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, 1)
        self.fc = nn.Linear(26 * 26 * 16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        return self.fc(x)


def train(model: nn.Module, loader: DataLoader, optimizer: PSDOptimizer, criterion: nn.Module) -> float:
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000)

    model = SimpleCNN().to(device)
    optimizer = PSDOptimizer(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 2):  # single epoch for quickstart
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, test_loader)
        logging.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%.2f%%",
            epoch,
            train_loss,
            val_loss,
            val_acc * 100,
        )
