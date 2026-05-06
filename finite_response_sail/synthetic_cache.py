"""Create tiny synthetic finite-response caches for smoke tests.

This module is not a paper experiment. It builds deterministic toy response-set
payloads with the same schema as `response_set.py`, so reviewers can validate
the objective, feature, training, and stress-evaluation code without public
datasets, private caches, model downloads, or W&B.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .response_set import CACHE_VERSION, response_set_path


SYNTHETIC_MODEL_NAME = "synthetic-smoke-cache"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _candidate(split: str, n: int, k: int, K: int) -> Dict[str, object]:
    token_length = 5 + k
    length_norm = token_length / 16.0
    redundancy = k / max(K - 1, 1)
    compression_ratio = token_length / 32.0
    base_logprob = -2.0 - 0.07 * k + 0.01 * n
    overlap = 0.15 + 0.08 * ((n + k) % K)
    true_reward = 0.25 + 0.08 * k + 0.01 * ((n + k) % 3)
    return {
        "prompt_id": f"{split}-{n}",
        "text": f"Synthetic response {k} for {split} prompt {n}.",
        "token_ids": list(range(100 + 10 * n + k, 100 + 10 * n + k + token_length)),
        "token_length": token_length,
        "eos_flag": True,
        "truncated_flag": False,
        "gen_temperature": [0.3, 0.6, 0.9, 1.2][k % 4],
        "base_logprob_sum": float(base_logprob * token_length),
        "base_logprob_per_token": float(base_logprob),
        "true_reward": float(true_reward),
        "length_norm": float(length_norm),
        "redundancy": float(redundancy),
        "compression_ratio": float(compression_ratio),
        "prompt_overlap_rouge1": float(overlap),
        "prompt_overlap_rouge2": float(overlap / 2.0),
        "prompt_overlap_rougeL": float(overlap),
    }


def _build_payload(split: str, size: int, K: int, seed: int) -> Dict[str, object]:
    prompts = [f"Synthetic finite-response prompt {split} {i}." for i in range(size)]
    candidates: List[List[Dict[str, object]]] = [
        [_candidate(split, i, k, K) for k in range(K)] for i in range(size)
    ]
    return {
        "meta": {
            "cache_version": CACHE_VERSION,
            "prompt_source": "synthetic_smoke",
            "split": split,
            "N": size,
            "K": K,
            "model": SYNTHETIC_MODEL_NAME,
            "reward_mode": "diagnostic_synthetic",
            "max_new_tokens": 16,
            "top_p": 1.0,
            "temperatures": [0.3, 0.6, 0.9, 1.2] * max(K // 4, 1),
            "seed": seed,
        },
        "prompt_ids": [f"{split}-{i}" for i in range(size)],
        "prompts": prompts,
        "dialogues": prompts,
        "refs": ["" for _ in range(size)],
        "row_meta": [{} for _ in range(size)],
        "candidates": candidates,
    }


def build(cache_dir: Path, K: int, train_size: int, val_size: int, test_size: int, seed: int) -> None:
    if K % 4 != 0:
        raise ValueError(f"K must be a multiple of 4; got {K}")
    _set_seed(seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sizes = {"train": train_size, "val": val_size, "test": test_size}
    for split, size in sizes.items():
        path = response_set_path(cache_dir, K, split)
        payload = _build_payload(split, size, K, seed)
        torch.save(payload, path)
        print(f"[write] {path} (synthetic, N={size}, K={K})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tiny synthetic smoke caches")
    parser.add_argument("--cache-dir", default="finite_response_sail/cache_smoke")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--val-size", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(Path(args.cache_dir), args.K, args.train_size, args.val_size, args.test_size, args.seed)


if __name__ == "__main__":
    main()
