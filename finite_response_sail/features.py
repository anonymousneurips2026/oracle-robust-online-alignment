"""Build feature maps psi(x, y_i) for the two feature sets.

Feature sets:
    controlled -- uses true ROUGE-Lsum as a feature (oracle-aligned mechanism test)
    no_oracle  -- uses only features computable without the synthetic reward

Every scalar feature is train-split-normalized (z-score then tanh-squashed into
[-1, 1]). For the controlled set the features are concatenated with a bias of 1
and the full vector is L2-normalized so ||psi|| <= 1. The active anonymous
release uses only Qwen-derived or string-computed cache fields; legacy caches
with `minilm_sim` are still accepted for compatibility, but MiniLM is not part
of the default reproduction path.

Cached to cache/psi_K{K}_<featureset>_<split>.pt with:
    psi: float32 tensor [N, K, d]
    feature_names: list[str]
    scaler: dict (train stats and projection matrix)
    meta: dict  -- mirrors the response-set meta (includes K, model, temperatures,
                   max_new_tokens, top_p, cache_version) so train.py can validate
                   the cached pool against the current CLI args.
"""
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from rouge_score import rouge_scorer

from .response_set import response_set_path


def psi_path(cache_dir: Path, K: int, featureset: str, split: str) -> Path:
    return Path(cache_dir) / f"psi_K{K}_{featureset}_{split}.pt"


def scaler_path(cache_dir: Path, K: int, featureset: str) -> Path:
    return Path(cache_dir) / f"scaler_K{K}_{featureset}.pt"


# ---------------------- helpers ----------------------


def _tanh_zscore(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    z = (x - mean) / max(std, 1e-6)
    return np.tanh(z)


def _stack_scalar(candidates: List[List[Dict[str, Any]]], key: str) -> np.ndarray:
    return np.asarray([[c[key] for c in cs] for cs in candidates], dtype=np.float32)


def _has_candidate_key(payload: Dict[str, Any], key: str) -> bool:
    cands = payload["candidates"]
    return bool(cands and cands[0] and key in cands[0][0])


def _controlled_aux_key(payload: Dict[str, Any]) -> str:
    if _has_candidate_key(payload, "prompt_overlap_rougeL"):
        return "prompt_overlap_rougeL"
    if _has_candidate_key(payload, "minilm_sim"):
        return "minilm_sim"
    raise KeyError("controlled features need prompt_overlap_rougeL or legacy minilm_sim")


# ---------------------- overlap features ----------------------


_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def _overlap_features(dialogue: str, candidate_text: str) -> Tuple[float, float, float]:
    if not candidate_text.strip():
        return 0.0, 0.0, 0.0
    s = _SCORER.score(dialogue, candidate_text)
    return (
        s["rouge1"].fmeasure,
        s["rouge2"].fmeasure,
        s["rougeL"].fmeasure,
    )


def build_overlap_matrix(dialogues: List[str], candidates: List[List[Dict[str, Any]]]) -> np.ndarray:
    """Returns [N, K, 3] with (rouge1, rouge2, rougeL) overlap between dialogue and candidate."""
    N, K = len(candidates), len(candidates[0])
    out = np.zeros((N, K, 3), dtype=np.float32)
    for n, (d, cs) in enumerate(zip(dialogues, candidates)):
        for k, c in enumerate(cs):
            out[n, k] = _overlap_features(d, c["text"])
    return out


# ---------------------- scalers ----------------------


def _fit_scaler_controlled(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    cands = payload["candidates"]
    aux_key = _controlled_aux_key(payload)
    stats: Dict[str, Tuple[float, float]] = {}
    for key in ("true_reward", "length_norm", "redundancy",
                "compression_ratio", "base_logprob_per_token", aux_key):
        arr = _stack_scalar(cands, key)
        stats[key] = (float(arr.mean()), float(arr.std() + 1e-8))
    return {"kind": "controlled", "stats": stats, "aux_key": aux_key}


def _fit_scaler_no_oracle(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    cands = payload["candidates"]
    stats: Dict[str, Tuple[float, float]] = {}
    for key in ("length_norm", "redundancy", "compression_ratio",
                "base_logprob_per_token"):
        arr = _stack_scalar(cands, key)
        stats[key] = (float(arr.mean()), float(arr.std() + 1e-8))

    # Overlap features on the TRAIN split.
    overlaps = build_overlap_matrix(payload["dialogues"], cands)  # [N, K, 3]
    for i, name in enumerate(("overlap_rouge1", "overlap_rouge2", "overlap_rougeL")):
        arr = overlaps[..., i]
        stats[name] = (float(arr.mean()), float(arr.std() + 1e-8))

    return {
        "kind": "no_oracle",
        "stats": stats,
    }


# ---------------------- apply scaler to produce psi ----------------------


def _apply_controlled(
    payload: Dict[str, Any], scaler: Dict[str, Any],
) -> Tuple[np.ndarray, List[str]]:
    assert scaler["kind"] == "controlled"
    cands = payload["candidates"]
    N, K = len(cands), len(cands[0])
    aux_key = scaler.get("aux_key") or _controlled_aux_key(payload)
    aux_name = "prompt_overlap_rougeL" if aux_key == "prompt_overlap_rougeL" else "legacy_minilm_sim"
    names = ["true_rouge_lsum", "length_norm", "redundancy",
             "compression_ratio", "base_logprob_per_token", aux_name, "bias"]
    out = np.zeros((N, K, len(names)), dtype=np.float32)

    key_to_col = {
        "true_reward": 0, "length_norm": 1, "redundancy": 2,
        "compression_ratio": 3, "base_logprob_per_token": 4, aux_key: 5,
    }
    for key, col in key_to_col.items():
        mu, sd = scaler["stats"][key]
        arr = _stack_scalar(cands, key)
        out[..., col] = _tanh_zscore(arr, mu, sd)
    out[..., 6] = 1.0   # bias
    # Vector-level L2 normalize.
    norms = np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1e-8)
    out = out / norms
    return out, names


def _apply_no_oracle(
    payload: Dict[str, Any], scaler: Dict[str, Any],
) -> Tuple[np.ndarray, List[str]]:
    assert scaler["kind"] == "no_oracle"
    cands = payload["candidates"]
    dialogues = payload["dialogues"]
    N, K = len(cands), len(cands[0])

    scalar_names = ["length_norm", "redundancy", "compression_ratio",
                    "overlap_rouge1", "overlap_rouge2", "overlap_rougeL",
                    "base_logprob_per_token", "bias"]
    scalar = np.zeros((N, K, len(scalar_names)), dtype=np.float32)

    # Non-overlap scalars.
    for i, key in enumerate(("length_norm", "redundancy", "compression_ratio",
                             "base_logprob_per_token")):
        mu, sd = scaler["stats"][key]
        scalar[..., scalar_names.index(key)] = _tanh_zscore(
            _stack_scalar(cands, key), mu, sd
        )

    # Overlap scalars (computed from dialogue + candidate text).
    overlaps = build_overlap_matrix(dialogues, cands)  # [N, K, 3]
    for i, name in enumerate(("overlap_rouge1", "overlap_rouge2", "overlap_rougeL")):
        mu, sd = scaler["stats"][name]
        scalar[..., scalar_names.index(name)] = _tanh_zscore(overlaps[..., i], mu, sd)
    scalar[..., scalar_names.index("bias")] = 1.0

    norms = np.linalg.norm(scalar, axis=-1, keepdims=True).clip(min=1e-8)
    return (scalar / norms).astype(np.float32), scalar_names


# ---------------------- public API ----------------------


def fit_scaler(train_payload: Dict[str, Any], featureset: str) -> Dict[str, Any]:
    if featureset == "controlled":
        return _fit_scaler_controlled(train_payload)
    if featureset == "no_oracle":
        return _fit_scaler_no_oracle(train_payload)
    raise ValueError(featureset)


def apply_scaler(payload: Dict[str, Any], scaler: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    if scaler["kind"] == "controlled":
        return _apply_controlled(payload, scaler)
    if scaler["kind"] == "no_oracle":
        return _apply_no_oracle(payload, scaler)
    raise ValueError(scaler["kind"])


def build_and_cache(featureset: str, cache_dir: Path, K: int):
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _load(split):
        p = response_set_path(cache_dir, K, split)
        if not p.exists():
            raise FileNotFoundError(
                f"response-set cache not found: {p}\n"
                f"Run `response_set.py --K {K}` first to build it."
            )
        return torch.load(p, weights_only=False)

    train = _load("train")
    val = _load("val")
    test = _load("test")

    print(f"[scaler] fitting '{featureset}' on train (K={K}, N={len(train['candidates'])})")
    scaler = fit_scaler(train, featureset)
    torch.save(scaler, scaler_path(cache_dir, K, featureset))

    for split, payload in [("train", train), ("val", val), ("test", test)]:
        psi, names = apply_scaler(payload, scaler)
        print(f"[apply] featureset={featureset} split={split}  psi={psi.shape}"
              f"  max||psi||={float(np.linalg.norm(psi, axis=-1).max()):.4f}")
        out_path = psi_path(cache_dir, K, featureset, split)
        torch.save(
            {
                "psi": torch.from_numpy(psi),
                "feature_names": names,
                "featureset": featureset,
                "split": split,
                "meta": payload["meta"],   # mirrors response-set meta incl. K, model, etc.
            },
            out_path,
        )
        print(f"[write] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--featureset", choices=["controlled", "no_oracle", "both"], default="both")
    parser.add_argument("--cache-dir", default="finite_response_sail/cache")
    parser.add_argument("--K", type=int, default=12,
                        help="response-set K to read (must match an existing response-set cache)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    sets = ["controlled", "no_oracle"] if args.featureset == "both" else [args.featureset]
    for fs in sets:
        build_and_cache(fs, cache_dir, args.K)


if __name__ == "__main__":
    main()
