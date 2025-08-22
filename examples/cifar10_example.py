"""Train a simple CIFAR-10 classifier using the PSD optimizer."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from psd_optimizer import PSDOptimizer


def main() -> None:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(train, batch_size=64, shuffle=True)

    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 10),
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = PSDOptimizer(model.parameters(), lr=0.01, epsilon=1e-3, r=1e-2, T=5)

    model.train()
    for epoch in range(1):
        for batch in loader:
            def closure():
                optimizer.zero_grad()
                x, y = batch
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                return loss
            optimizer.step(closure)
        print(f"Epoch {epoch+1} complete")


if __name__ == "__main__":
    main()
