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

import torch
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    text = "Hello world!" * 32
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    optimizer = PSDOptimizer(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(args.epochs):
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        print(f"Epoch {epoch + 1}: loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
