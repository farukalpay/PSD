import argparse
import hashlib
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ------------------------- Seeding utilities -------------------------

def make_subseed(master_seed: int, component_id: str, run_id: str, stream_id=0) -> int:
    """Derive a deterministic subseed from identifiers.

    Uses SHA256 to hash the identifiers and returns a 64-bit integer.
    """
    payload = f"{master_seed}|{component_id}|{run_id}|{stream_id}".encode()
    digest = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(digest, "big")


def numpy_gen(subseed: int) -> np.random.Generator:
    """Return a NumPy generator using the Philox bit generator."""
    return np.random.Generator(np.random.Philox(subseed))


def torch_gen(device: str, subseed: int) -> torch.Generator:
    """Return a torch Generator seeded with subseed for a given device."""
    g = torch.Generator(device)
    g.manual_seed(subseed)
    return g


def set_torch_deterministic():
    """Configure PyTorch for deterministic behavior."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------- Model definition -------------------------

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


# ------------------------- Training utilities -------------------------

@dataclass
class RunConfig:
    master_seed: int
    run_id: str
    epochs: int
    batch_size: int


def seed_python_np_torch(seed: int):
    random.seed(seed)
    # NumPy's legacy seeding expects 32-bit integers.
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def train(cfg: RunConfig):
    device = "cpu"
    set_torch_deterministic()

    transform = transforms.ToTensor()
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)

    # Seed before model/optimizer initialization for reproducible parameters
    init_seed = make_subseed(cfg.master_seed, "trainer", cfg.run_id, stream_id="init")
    seed_python_np_torch(init_seed)
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    first_labels = None
    first_weights_before = None
    first_weights_after = None

    for epoch in range(cfg.epochs):
        # Seed main process for this epoch
        trainer_seed = make_subseed(cfg.master_seed, "trainer", cfg.run_id, stream_id=epoch)
        seed_python_np_torch(trainer_seed)

        # Generator for shuffling
        dl_gen_seed = make_subseed(cfg.master_seed, "dataloader_gen", cfg.run_id, stream_id=epoch)
        dl_gen = torch_gen(device, dl_gen_seed)

        def worker_init_fn(worker_id: int):
            # Incorporating epoch and worker_id into the stream_id ensures each worker
            # and each epoch has a distinct, reproducible seed stream.
            w_seed = make_subseed(cfg.master_seed, "dataloader", cfg.run_id, stream_id=f"{epoch}|{worker_id}")
            seed_python_np_torch(w_seed)

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=worker_init_fn,
            generator=dl_gen,
        )

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            if epoch == 0 and batch_idx == 0:
                first_labels = target[:3].tolist()
                # Use non-border pixels so gradients are non-zero.
                first_weights_before = (
                    model.net[0].weight[0, 100:103].detach().clone().tolist()
                )

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if epoch == 0 and batch_idx == 0:
                first_weights_after = (
                    model.net[0].weight[0, 100:103].detach().clone().tolist()
                )
            break  # Only first batch needed for determinism check

    return first_labels, first_weights_before, first_weights_after


def main():
    parser = argparse.ArgumentParser(description="Deterministic MNIST training demo")
    parser.add_argument("--master-seed", type=int, default=2025)
    parser.add_argument("--run-id", type=str, default="demoA")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = RunConfig(args.master_seed, args.run_id, args.epochs, args.batch_size)

    for run in range(2):
        labels, w_before, w_after = train(cfg)
        print(f"Run {run + 1}: labels {labels}")
        print(f"Run {run + 1}: first-layer weights before step {w_before}")
        print(f"Run {run + 1}: first-layer weights after step  {w_after}\n")


if __name__ == "__main__":
    main()
