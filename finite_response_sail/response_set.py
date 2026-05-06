"""Build finite-response candidate caches from anonymous JSONL prompt manifests.

The active release path is intentionally dataset-agnostic. It reads prompt
manifests supplied by the reviewer/author, generates a fixed candidate set with
the frozen reference model, and stores all finite-response fields consumed by
training and evaluation. The submitted paper discloses Qwen as the frozen
candidate/reference-logit provider; no public prompt dataset is hardcoded here.

Expected manifests:

    data/prompts/train.jsonl
    data/prompts/val.jsonl
    data/prompts/test.jsonl

Each JSONL row must include:

    {"id": "...", "prompt": "...", "reward_reference": "..."}

The default reward mode uses ROUGE-Lsum against `reward_reference` to instantiate
a deterministic synthetic true reward. If the exact paper cache used another
synthetic reward, provide that cache or adapt this reward hook before claiming
exact numerical reproduction.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_GEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 96
CACHE_VERSION = "v2_manifest"


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    prompt: str
    reward_reference: Optional[str]
    row_meta: Dict[str, Any]


def response_set_path(cache_dir: Path, K: int, split: str) -> Path:
    """Canonical cache filename; encodes K so pools cannot be silently reused."""
    return Path(cache_dir) / f"response_set_K{K}_{split}.pt"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _temperature_schedule(K: int) -> List[float]:
    """K=12 -> {0.3, 0.6, 0.9, 1.2} x 3 samples."""
    if K % 4 != 0:
        raise ValueError(f"K must be a multiple of 4; got {K}")
    per = K // 4
    return [t for t in (0.3, 0.6, 0.9, 1.2) for _ in range(per)]


def _load_jsonl(path: Path, limit: Optional[int], reward_mode: str) -> List[PromptRecord]:
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt manifest not found: {path}. "
            "See docs/CACHE_GENERATION.md for the required JSONL schema."
        )
    records: List[PromptRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{lineno}: {exc}") from exc
            if "id" not in row or "prompt" not in row:
                raise ValueError(f"{path}:{lineno} must contain `id` and `prompt` fields")
            ref = (
                row.get("reward_reference")
                or row.get("reference")
                or row.get("target")
                or row.get("summary")
            )
            if reward_mode == "reference_rouge_lsum" and not ref:
                raise ValueError(
                    f"{path}:{lineno} needs `reward_reference` for "
                    "--reward-mode reference_rouge_lsum"
                )
            meta = {k: v for k, v in row.items() if k not in {"id", "prompt", "reward_reference"}}
            records.append(
                PromptRecord(
                    prompt_id=str(row["id"]),
                    prompt=str(row["prompt"]),
                    reward_reference=None if ref is None else str(ref),
                    row_meta=meta,
                )
            )
            if limit is not None and limit > 0 and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"Prompt manifest is empty after filtering: {path}")
    return records


def _build_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return prompt


def _type_token_redundancy(token_ids: List[int]) -> float:
    if not token_ids:
        return 0.0
    return 1.0 - (len(set(token_ids)) / len(token_ids))


def _overlap_scores(prompt: str, candidate: str, scorer) -> Dict[str, float]:
    if not candidate.strip():
        return {"prompt_overlap_rouge1": 0.0, "prompt_overlap_rouge2": 0.0, "prompt_overlap_rougeL": 0.0}
    score = scorer.score(prompt, candidate)
    return {
        "prompt_overlap_rouge1": float(score["rouge1"].fmeasure),
        "prompt_overlap_rouge2": float(score["rouge2"].fmeasure),
        "prompt_overlap_rougeL": float(score["rougeL"].fmeasure),
    }


def _normalize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _assign_diagnostic_reward(candidates: List[Dict[str, Any]]) -> None:
    """Non-paper synthetic reward for demos when no reward reference is supplied."""
    logp = _normalize_1d(np.asarray([c["base_logprob_per_token"] for c in candidates]))
    redundancy = _normalize_1d(np.asarray([c["redundancy"] for c in candidates]))
    length = np.asarray([c["length_norm"] for c in candidates], dtype=np.float64)
    length_pref = 1.0 - np.clip(np.abs(length - 0.45) / 0.45, 0.0, 1.0)
    overlap = _normalize_1d(np.asarray([c["prompt_overlap_rougeL"] for c in candidates]))
    reward = 0.45 * logp + 0.25 * (1.0 - redundancy) + 0.20 * length_pref + 0.10 * overlap
    for c, r in zip(candidates, reward):
        c["true_reward"] = float(r)


def _generate_one_prompt(
    model,
    tokenizer,
    chat_prompt: str,
    K: int,
    max_new_tokens: int,
    top_p: float,
    temps: List[float],
    device: str,
) -> List[Dict[str, Any]]:
    by_temp: Dict[float, int] = {}
    for t in temps:
        by_temp[t] = by_temp.get(t, 0) + 1

    enc = tokenizer(chat_prompt, return_tensors="pt").to(device)
    prompt_len = enc["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    results: List[Dict[str, Any]] = []

    for temp, n_samples in by_temp.items():
        sequences = model.generate(
            **enc,
            do_sample=True,
            temperature=float(temp),
            top_p=float(top_p),
            max_new_tokens=max_new_tokens,
            num_return_sequences=n_samples,
            pad_token_id=pad_id,
            return_dict_in_generate=False,
        )
        gen = sequences[:, prompt_len:]
        for b in range(n_samples):
            tok_ids = gen[b].tolist()
            if eos_id is not None and eos_id in tok_ids:
                eos_pos = tok_ids.index(eos_id)
                tok_ids_clean = tok_ids[:eos_pos]
                eos_flag = True
                truncated = False
            else:
                tok_ids_clean = tok_ids
                eos_flag = False
                truncated = len(tok_ids_clean) >= max_new_tokens
            results.append(
                {
                    "text": tokenizer.decode(tok_ids_clean, skip_special_tokens=True),
                    "token_ids": tok_ids_clean,
                    "token_length": len(tok_ids_clean),
                    "eos_flag": bool(eos_flag),
                    "truncated_flag": bool(truncated),
                    "gen_temperature": float(temp),
                }
            )
    return results


@torch.no_grad()
def _forced_decode_logprobs(
    model,
    tokenizer,
    chat_prompt: str,
    candidates: List[Dict[str, Any]],
    device: str,
    pad_id: int,
    micro_batch: int = 4,
) -> None:
    if not candidates:
        return

    prompt_ids = tokenizer(chat_prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)
    max_cand_len = max(max((c["token_length"] for c in candidates), default=1), 1)
    full_len = prompt_len + max_cand_len

    input_ids = []
    attn = []
    for c in candidates:
        seq = prompt_ids + c["token_ids"]
        seq = seq[:full_len]
        a = [1] * len(seq)
        pad_n = full_len - len(seq)
        input_ids.append(seq + [pad_id] * pad_n)
        attn.append(a + [0] * pad_n)

    ids_all = torch.tensor(input_ids, device=device)
    am_all = torch.tensor(attn, device=device)
    tok_lp_region = torch.full(
        (len(candidates), max_cand_len),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )

    start = prompt_len - 1
    for b0 in range(0, len(candidates), micro_batch):
        b1 = min(b0 + micro_batch, len(candidates))
        ids = ids_all[b0:b1]
        am = am_all[b0:b1]
        outputs = model(input_ids=ids, attention_mask=am, use_cache=False)
        logits_region = outputs.logits[:, start:start + max_cand_len, :]
        del outputs
        targets_region = ids[:, prompt_len:prompt_len + max_cand_len]
        log_probs = F.log_softmax(logits_region.float(), dim=-1)
        lp = log_probs.gather(-1, targets_region.unsqueeze(-1)).squeeze(-1)
        tok_lp_region[b0:b1, : lp.shape[1]] = lp
        del logits_region, log_probs, lp

    for b, c in enumerate(candidates):
        length = c["token_length"]
        if length == 0:
            c["base_logprob_sum"] = 0.0
            c["base_logprob_per_token"] = 0.0
            continue
        total = tok_lp_region[b, :length].sum().item()
        c["base_logprob_sum"] = float(total)
        c["base_logprob_per_token"] = float(total / length)

    if device == "cuda":
        torch.cuda.empty_cache()


def _split_manifest(prompt_dir: Path, split: str) -> Path:
    names = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}
    return prompt_dir / names[split]


def _iter_splits(split: str) -> Iterable[str]:
    return ("train", "val", "test") if split == "all" else (split,)


def build_split(
    split: str,
    prompt_dir: Path,
    size_limit: Optional[int],
    reward_mode: str,
    model_name: str,
    K: int,
    max_new_tokens: int,
    top_p: float,
    cache_dir: Path,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = response_set_path(cache_dir, K, split)
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return

    records = _load_jsonl(_split_manifest(prompt_dir, split), size_limit, reward_mode)
    print(f"[data] split={split} N={len(records)} source={prompt_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        print("[warn] CUDA unavailable; candidate generation on CPU will be slow")

    print(f"[model] loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    overlap_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    reward_scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    temps = _temperature_schedule(K)
    all_candidates: List[List[Dict[str, Any]]] = []
    chat_prompts = [_build_chat_prompt(tokenizer, row.prompt) for row in records]
    pad_id = tokenizer.pad_token_id

    print(f"[gen] K={K} temps={temps} max_new={max_new_tokens} top_p={top_p}")
    for n, (row, chat_prompt) in enumerate(zip(records, chat_prompts)):
        cands = _generate_one_prompt(
            model,
            tokenizer,
            chat_prompt,
            K=K,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temps=temps,
            device=device,
        )
        _forced_decode_logprobs(model, tokenizer, chat_prompt, cands, device, pad_id)

        prompt_token_len = max(1, len(tokenizer(row.prompt, add_special_tokens=False)["input_ids"]))
        for cand in cands:
            cand["prompt_id"] = row.prompt_id
            cand["length_norm"] = float(min(cand["token_length"] / max_new_tokens, 1.0))
            cand["redundancy"] = float(_type_token_redundancy(cand["token_ids"]))
            cand["compression_ratio"] = float(cand["token_length"] / prompt_token_len)
            cand.update(_overlap_scores(row.prompt, cand["text"], overlap_scorer))
            if reward_mode == "reference_rouge_lsum":
                cand["true_reward"] = float(
                    reward_scorer.score(row.reward_reference or "", cand["text"])["rougeLsum"].fmeasure
                )
        if reward_mode == "diagnostic_synthetic":
            _assign_diagnostic_reward(cands)

        all_candidates.append(cands)
        if (n + 1) % 50 == 0 or (n + 1) == len(records):
            print(f"  [gen] {n + 1}/{len(records)} prompts done")

    payload = {
        "meta": {
            "cache_version": CACHE_VERSION,
            "prompt_source": "jsonl_manifest",
            "split": split,
            "N": len(records),
            "K": K,
            "model": model_name,
            "reward_mode": reward_mode,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "temperatures": temps,
            "manifest_file": _split_manifest(prompt_dir, split).name,
        },
        "prompt_ids": [row.prompt_id for row in records],
        "prompts": [row.prompt for row in records],
        "dialogues": [row.prompt for row in records],
        "refs": [row.reward_reference or "" for row in records],
        "row_meta": [row.row_meta for row in records],
        "candidates": all_candidates,
    }
    torch.save(payload, out_path)
    print(f"[write] {out_path} (N={len(records)}, K={K})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build finite-response candidate caches")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--prompt-dir", default="data/prompts")
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--train-size", type=int, default=1500)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--model", default=DEFAULT_GEN_MODEL)
    parser.add_argument(
        "--reward-mode",
        choices=["reference_rouge_lsum", "diagnostic_synthetic"],
        default="reference_rouge_lsum",
        help=(
            "How to instantiate candidate true_reward. `reference_rouge_lsum` "
            "requires reward_reference in each JSONL row. `diagnostic_synthetic` "
            "is for smoke/demo caches, not paper-number reproduction."
        ),
    )
    parser.add_argument("--cache-dir", default="finite_response_sail/cache")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _set_seed(args.seed)
    prompt_dir = Path(args.prompt_dir)
    cache_dir = Path(args.cache_dir)
    limits = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    for sp in _iter_splits(args.split):
        build_split(
            split=sp,
            prompt_dir=prompt_dir,
            size_limit=limits[sp],
            reward_mode=args.reward_mode,
            model_name=args.model,
            K=args.K,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            cache_dir=cache_dir,
        )


if __name__ == "__main__":
    main()
