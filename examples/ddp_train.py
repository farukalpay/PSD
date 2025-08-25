"""Deterministic DDP training on CIFAR-10 with hashed seeding."""

from __future__ import annotations

import argparse
import hashlib
import random
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, get_worker_info
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------------
# Seeding utilities
# ---------------------------------------------------------------------------

def make_subseed(
    master_seed: int, component_id: str, run_id: str, stream_id: str | int = "0"
) -> tuple[int, str]:
    """Hash inputs with SHA-256 and use the first 8 bytes as an int subseed."""
    msg = f"{master_seed}|{component_id}|{run_id}|{stream_id}".encode()
    digest = hashlib.sha256(msg).hexdigest()
    subseed = int.from_bytes(bytes.fromhex(digest)[:8], "big")
    return subseed, digest


def np_gen(subseed: int) -> np.random.Generator:
    """Construct a NumPy Generator based on Philox bit generator."""
    return np.random.Generator(np.random.Philox(subseed))


def torch_gen(device: torch.device | str, subseed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(subseed)
    return gen


def set_torch_deterministic() -> None:
    """Force deterministic behaviour in PyTorch."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data augmentation with an explicit generator
# ---------------------------------------------------------------------------


class DeterministicFlip:
    """Random horizontal flip driven by an explicit RNG stream."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        self.generator: torch.Generator | None = None

    def set_generator(self, gen: torch.Generator) -> None:
        self.generator = gen

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.generator is None:
            raise RuntimeError("Generator not set for DeterministicFlip")
        if torch.rand(1, generator=self.generator).item() < self.p:
            img = TF.hflip(img)
        return img


# ---------------------------------------------------------------------------
# Worker initialisation
# ---------------------------------------------------------------------------


def worker_init_fn_factory(
    master_seed: int, run_id: str, rank: int, epoch: int
) -> Callable[[int], None]:
    def init_fn(worker_id: int) -> None:
        stream = f"rank={rank}|epoch={epoch}|worker={worker_id}"
        dl_seed, _ = make_subseed(master_seed, "dataloader", run_id, stream)
        random.seed(dl_seed)
        np_gen(dl_seed)
        torch.manual_seed(dl_seed)

        aug_seed, _ = make_subseed(master_seed, "augment", run_id, stream)
        info = get_worker_info()
        flip = info.dataset.transform.transforms[-1]
        flip.set_generator(torch_gen("cpu", aug_seed))

    return init_fn


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8 * 32 * 32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-seed", type=int, default=0)
    parser.add_argument("--run-id", type=str, default="RUN0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    seed_tensor = torch.tensor(args.master_seed if rank == 0 else 0, dtype=torch.long)
    dist.broadcast(seed_tensor, src=0)
    master_seed = seed_tensor.item()

    set_torch_deterministic()
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    stream = f"rank={rank}|epoch=0|worker=main"
    model_seed, model_digest = make_subseed(master_seed, "model_init", args.run_id, stream)
    random.seed(model_seed)
    np_gen(model_seed)
    torch.manual_seed(model_seed)

    model = SmallCNN().to(device)
    ddp_model = DDP(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    flip = DeterministicFlip()
    transform = transforms.Compose([transforms.ToTensor(), flip])
    dataset = datasets.CIFAR10("data", train=True, download=True, transform=transform)

    for epoch in range(args.epochs):
        stream = f"rank={rank}|epoch={epoch}|worker=main"
        trainer_seed, _ = make_subseed(master_seed, "trainer", args.run_id, stream)
        random.seed(trainer_seed)
        np_gen(trainer_seed)
        torch.manual_seed(trainer_seed)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        sampler.set_epoch(epoch)
        dl_seed, _ = make_subseed(master_seed, "dataloader", args.run_id, stream)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=2,
            worker_init_fn=worker_init_fn_factory(master_seed, args.run_id, rank, epoch),
            generator=torch_gen("cpu", dl_seed),
        )

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            if epoch == 0 and batch_idx == 0:
                print(f"Rank {rank} first labels: {target[:3].tolist()}")
                convw = ddp_model.module.conv.weight.view(-1)
                print(f"Rank {rank} conv.weight[:3] before step: {convw[:3].tolist()}")
            loss.backward()
            optimizer.step()
            break  # one batch is enough for the demo

    dist.barrier()
    print(f"Rank {rank} model_init subseed: {model_seed} digest: {model_digest}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
