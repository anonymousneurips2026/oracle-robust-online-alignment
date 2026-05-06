#!/usr/bin/env bash
# Tiny synthetic smoke test for the anonymous release.
#
# This validates cache loading, feature construction, training, and stress
# evaluation without public datasets, private caches, model downloads, W&B, or
# API keys. It is not a paper-number reproduction.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_run_helpers.sh
source "${SCRIPT_DIR}/_run_helpers.sh"

cd "${REPO_ROOT}"

SMOKE_TAG="${SMOKE_TAG:-smoke_synthetic}"
SMOKE_CACHE_DIR="${SMOKE_CACHE_DIR:-${MODULE_DIR}/cache_smoke}"
SMOKE_RESULTS_DIR="${SMOKE_RESULTS_DIR:-${MODULE_DIR}/results_smoke}"
SMOKE_K="${SMOKE_K:-4}"
SMOKE_TRAIN_SIZE="${SMOKE_TRAIN_SIZE:-8}"
SMOKE_VAL_SIZE="${SMOKE_VAL_SIZE:-4}"
SMOKE_TEST_SIZE="${SMOKE_TEST_SIZE:-4}"
SMOKE_MODEL="synthetic-smoke-cache"

echo "[smoke] synthetic cache=${SMOKE_CACHE_DIR}"
echo "[smoke] results=${SMOKE_RESULTS_DIR}"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m finite_response_sail.synthetic_cache \
    --cache-dir "${SMOKE_CACHE_DIR}" \
    --K "${SMOKE_K}" \
    --train-size "${SMOKE_TRAIN_SIZE}" \
    --val-size "${SMOKE_VAL_SIZE}" \
    --test-size "${SMOKE_TEST_SIZE}" \
    --seed 0

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m finite_response_sail.features \
    --featureset controlled \
    --cache-dir "${SMOKE_CACHE_DIR}" \
    --K "${SMOKE_K}"

for q in 0 1; do
    rho_alg="$("${PY}" - "$q" <<'PY'
import sys
print(format(0.05 * float(sys.argv[1]), "g"))
PY
)"
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m finite_response_sail.train \
        --mode biased_feedback \
        --objective exact \
        --featureset controlled \
        --ref base_lm_lognorm \
        --bias_attr redundancy \
        --rho_mis 0.05 \
        --rho_alg "${rho_alg}" \
        --beta 0.1 \
        --tau 2.0 \
        --R 10.0 \
        --seed 0 \
        --rounds 1 \
        --steps_per_round 5 \
        --batch_size 2 \
        --lr 0.05 \
        --K "${SMOKE_K}" \
        --expect-model "${SMOKE_MODEL}" \
        --tag "${SMOKE_TAG}" \
        --cache-dir "${SMOKE_CACHE_DIR}" \
        --results_dir "${SMOKE_RESULTS_DIR}" \
        --smoke
done

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m finite_response_sail._true_to_worst_stress_eval \
    --results-dir "${SMOKE_RESULTS_DIR}/${SMOKE_TAG}" \
    --out-dir "${SMOKE_RESULTS_DIR}/${SMOKE_TAG}/reports/true_to_worst_stress" \
    --cache-dir "${SMOKE_CACHE_DIR}" \
    --rho-mis 0.05 \
    --smoke

echo "[smoke] complete"
