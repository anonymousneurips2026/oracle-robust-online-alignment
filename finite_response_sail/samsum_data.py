"""Legacy SAMSum helpers retained for auditability only.

SAMSum is not part of the active reproduction path for the current anonymous
release because it is not disclosed in the current paper. The active cache
builder reads generic JSONL prompt manifests through `response_set.py`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

PROMPT_TEMPLATE = (
    "Summarize the following dialogue.\n\n"
    "Dialogue:\n{dialogue}\n\n"
    "Summary:"
)


@dataclass
class Config:
    train_size: int = 1500
    val_size: int = 200
    test_size: int = 400
    seed: int = 0
    dataset_name: str = "samsum"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def rouge_lsum(prediction: str, reference: str, scorer) -> float:
    """Return ROUGE-Lsum F-measure using an existing RougeScorer."""
    return float(scorer.score(reference, prediction)["rougeLsum"].fmeasure)


def _format_pairs(split, limit: int) -> List[Tuple[str, str]]:
    if limit is not None and limit > 0:
        split = split.select(range(min(limit, len(split))))
    out: List[Tuple[str, str]] = []
    for row in split:
        dialogue = str(row["dialogue"])
        summary = str(row["summary"])
        out.append((PROMPT_TEMPLATE.format(dialogue=dialogue), summary))
    return out


def load_samsum(cfg: Config):
    """Load SAMSum and return train/val/test prompt-summary pairs.

    This legacy helper is not used by the active paper-aligned release path.
    Hugging Face's SAMSum split names are `train`, `validation`, and `test`.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in user envs.
        raise ImportError(
            "The legacy SAMSum helper requires the optional `datasets` package. "
            "It is not needed for the active anonymous release path."
        ) from exc

    set_seed(cfg.seed)
    ds = load_dataset(cfg.dataset_name)
    train = _format_pairs(ds["train"], cfg.train_size)
    val = _format_pairs(ds["validation"], cfg.val_size)
    test = _format_pairs(ds["test"], cfg.test_size)
    return train, val, test
