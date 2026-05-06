"""Guards against silent K / model mismatches between cache and CLI args.

Does NOT spin up the LLM; writes a hand-crafted payload schema and calls
`train._assert_cache_matches` directly.
"""
import pytest

from finite_response_sail.train import _assert_cache_matches


def _meta(K=12, model="Qwen/Qwen2.5-0.5B-Instruct"):
    return {
        "cache_version": "v1",
        "split": "train",
        "N": 4,
        "K": K,
        "model": model,
        "max_new_tokens": 96,
        "top_p": 0.95,
        "temperatures": [0.3, 0.6, 0.9, 1.2] * (K // 4),
    }


def test_match_passes():
    _assert_cache_matches(
        _meta(K=12), K=12, featureset="controlled", split="train",
        expect_model="Qwen/Qwen2.5-0.5B-Instruct", source="dummy.pt",
    )


def test_K_mismatch_raises():
    with pytest.raises(RuntimeError, match="Cache K mismatch"):
        _assert_cache_matches(
            _meta(K=12), K=16, featureset="controlled", split="train",
            expect_model="Qwen/Qwen2.5-0.5B-Instruct", source="dummy.pt",
        )


def test_model_mismatch_raises():
    with pytest.raises(RuntimeError, match="Cache model mismatch"):
        _assert_cache_matches(
            _meta(model="other/model"),
            K=12, featureset="controlled", split="train",
            expect_model="Qwen/Qwen2.5-0.5B-Instruct", source="dummy.pt",
        )


def test_empty_expect_model_skips_model_check():
    _assert_cache_matches(
        _meta(model="other/model"),
        K=12, featureset="controlled", split="train",
        expect_model="", source="dummy.pt",
    )


def test_missing_K_raises():
    bad = _meta()
    del bad["K"]
    with pytest.raises(RuntimeError, match="Cache K mismatch"):
        _assert_cache_matches(
            bad, K=12, featureset="controlled", split="train",
            expect_model="Qwen/Qwen2.5-0.5B-Instruct", source="dummy.pt",
        )
