#!/usr/bin/env python
"""Train a small language model with PSDOptimizer.

This script demonstrates how to fine-tune a tiny causal language model using
:class:`~psd_optimizer.PSDOptimizer`.  It downloads a compact model from the
🤗 Transformers library, tokenizes a short dummy corpus and performs a few
gradient steps with PSD.  The example is intentionally minimal to keep the
focus on how the optimizer integrates with transformer-style models.

Example
-------
Run the script with the default tiny GPT-2 model:

```
python scripts/train_small_language_model.py
```

Specify a different pretrained model from the Hub:

```
python scripts/train_small_language_model.py --model distilgpt2 --epochs 5
```
"""

from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from psd_optimizer import PSDOptimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a small LM with PSDOptimizer")
    parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="Pretrained model name on the Hugging Face Hub",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro batch size")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    text = "Hello world!" * 128
    inputs = tokenizer([text] * 64, return_tensors="pt", padding=True)
    dataset = inputs["input_ids"]
    loader = DataLoader(dataset, batch_size=args.batch_size)

    optimizer = PSDOptimizer(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(loader):
            batch = batch.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(batch, labels=batch)
                loss = outputs.loss / args.accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                val_outputs = model(dataset.to(device)[:args.batch_size], labels=dataset.to(device)[:args.batch_size])
                val_loss = val_outputs.loss
        scheduler.step(val_loss)
        if val_loss.item() < best_val:
            best_val = val_loss.item()
            model.save_pretrained("checkpoint")
        logging.info(
            "Epoch %d: val_loss=%.4f lr=%.2e", epoch + 1, val_loss.item(), scheduler.optimizer.param_groups[0]['lr']
        )


if __name__ == "__main__":
    main()
