#!/usr/bin/env bash
# Release entry point in the style of the reference code package.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
PROMPT_DIR="${PROMPT_DIR:-data/prompts}"
CACHE_DIR="${CACHE_DIR:-finite_response_sail/cache}"
K="${K:-12}"
GEN_MODEL="${GEN_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
REWARD_MODE="${REWARD_MODE:-reference_rouge_lsum}"
TRAIN_SIZE="${TRAIN_SIZE:-1500}"
VAL_SIZE="${VAL_SIZE:-200}"
TEST_SIZE="${TEST_SIZE:-400}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"

usage() {
    cat <<'EOF'
Usage: bash run.sh <subcommand> [options]

Subcommands:
  cache [--dry-run]      Build response-set and controlled-feature caches.
  smoke                  Run tiny synthetic smoke test.
  reproduce              Run the full documented pipeline.
  reproduce-dry-run      Print full reproduction commands without training.
  test                   Compile Python files and run pytest.

Environment overrides:
  PYTHON, PROMPT_DIR, CACHE_DIR, K, GEN_MODEL, REWARD_MODE,
  TRAIN_SIZE, VAL_SIZE, TEST_SIZE, MAX_NEW_TOKENS
EOF
}

run_or_print() {
    local dry_run="$1"
    shift
    if [ "${dry_run}" = "1" ]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

cmd_cache() {
    local dry_run=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --dry-run) dry_run=1; shift;;
            -h|--help) usage; exit 0;;
            *) echo "[error] unknown cache flag: $1" >&2; exit 2;;
        esac
    done

    echo "[cache] required manifests:"
    echo "  ${PROMPT_DIR}/train.jsonl"
    echo "  ${PROMPT_DIR}/val.jsonl"
    echo "  ${PROMPT_DIR}/test.jsonl"
    echo "[cache] see docs/CACHE_GENERATION.md for the JSONL schema"

    cd "${ROOT}"
    run_or_print "${dry_run}" env PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m finite_response_sail.response_set \
        --split all \
        --prompt-dir "${PROMPT_DIR}" \
        --K "${K}" \
        --train-size "${TRAIN_SIZE}" \
        --val-size "${VAL_SIZE}" \
        --test-size "${TEST_SIZE}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --model "${GEN_MODEL}" \
        --reward-mode "${REWARD_MODE}" \
        --cache-dir "${CACHE_DIR}" \
        --seed 0
    run_or_print "${dry_run}" env PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m finite_response_sail.features \
        --featureset controlled \
        --K "${K}" \
        --cache-dir "${CACHE_DIR}"
}

cmd_test() {
    cd "${ROOT}"
    "${PY}" -m compileall -q .
    env PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m pytest finite_response_sail/tests -q
}

sub="${1:-}"
shift || true
case "${sub}" in
    cache) cmd_cache "$@" ;;
    smoke) cd "${ROOT}" && bash scripts/run_smoke_test.sh "$@" ;;
    reproduce) cd "${ROOT}" && bash scripts/reproduce_all.sh "$@" ;;
    reproduce-dry-run) cd "${ROOT}" && bash scripts/reproduce_all.sh --dry-run "$@" ;;
    test) cmd_test "$@" ;;
    ""|-h|--help) usage ;;
    *) echo "[error] unknown subcommand: ${sub}" >&2; usage; exit 2 ;;
esac
