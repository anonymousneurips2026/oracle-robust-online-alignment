#!/usr/bin/env bash
# Full finite-response reproduction pipeline. Use --dry-run to print commands.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_run_helpers.sh
source "${SCRIPT_DIR}/_run_helpers.sh"

DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift;;
        -h|--help)
            sed -n '1,80p' "$0"
            exit 0;;
        *)
            echo "[error] unknown flag: $1" >&2
            exit 2;;
    esac
done

cd "${REPO_ROOT}"

TAG="${TAG:-v4_true_to_worst_rhoalg_grid}"
KL_TAG="${KL_TAG:-v4_kl_baseline_eta01}"
GEN_MODEL="${GEN_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PROMPT_DIR="${PROMPT_DIR:-data/prompts}"
REWARD_MODE="${REWARD_MODE:-reference_rouge_lsum}"
K="${K:-12}"
TRAIN_SIZE="${TRAIN_SIZE:-1500}"
VAL_SIZE="${VAL_SIZE:-200}"
TEST_SIZE="${TEST_SIZE:-400}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
SEEDS_SPEC="${SEEDS_SPEC:-0-9}"
RHO_MIS_LIST_SPEC="${RHO_MIS_LIST_SPEC:-0.05 0.075 0.10}"
Q_LIST_SPEC="${Q_LIST_SPEC:-0 0.025 0.05 0.10 0.25 0.50 1.00}"
Q_LIST_075_EXTRA_SPEC="${Q_LIST_075_EXTRA_SPEC:-0.005 0.01}"

run() {
    if [ "${DRY_RUN}" = "1" ]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

echo "[reproduce] tag=${TAG} cache=${CACHE_DIR} results=${RESULTS_DIR}"
if [ "${DRY_RUN}" = "0" ]; then
    for manifest in "${PROMPT_DIR}/train.jsonl" "${PROMPT_DIR}/val.jsonl" "${PROMPT_DIR}/test.jsonl"; do
        if [ ! -f "${manifest}" ]; then
            echo "[error] missing prompt manifest: ${manifest}" >&2
            echo "[error] see docs/CACHE_GENERATION.md for the required cache-generation inputs" >&2
            exit 2
        fi
    done
fi

run env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m finite_response_sail.response_set \
    --split all --K "${K}" --train-size "${TRAIN_SIZE}" --val-size "${VAL_SIZE}" \
    --test-size "${TEST_SIZE}" --max-new-tokens "${MAX_NEW_TOKENS}" \
    --model "${GEN_MODEL}" --prompt-dir "${PROMPT_DIR}" --reward-mode "${REWARD_MODE}" \
    --cache-dir "${CACHE_DIR}" --seed 0

run env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m finite_response_sail.features \
    --featureset controlled --cache-dir "${CACHE_DIR}" --K "${K}"

run bash "${SCRIPT_DIR}/run_true_to_worst_rhoalg_grid.sh" \
    --tag "${TAG}" \
    --rho-mis-list "${RHO_MIS_LIST_SPEC}" \
    --q-list "${Q_LIST_SPEC}" \
    --seeds "${SEEDS_SPEC}"

run bash "${SCRIPT_DIR}/run_true_to_worst_rhoalg_grid.sh" \
    --tag "${TAG}" \
    --rho-mis 0.075 \
    --q-list "${Q_LIST_075_EXTRA_SPEC}" \
    --seeds "${SEEDS_SPEC}"

IFS=' ' read -r SEED_LO SEED_HI <<< "$(echo "${SEEDS_SPEC}" | awk -F- '{print $1, $2}')"
for seed in $(seq "${SEED_LO}" "${SEED_HI}"); do
    run env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m finite_response_sail.train \
        --mode biased_feedback --objective exact --featureset controlled \
        --ref base_lm_lognorm --bias_attr redundancy \
        --rho_mis 0.075 --rho_alg 0 --eta_kl 0.1 \
        --beta 0.1 --tau 2.0 --R 10.0 --seed "${seed}" \
        --rounds "${ROUNDS:-6}" --steps_per_round "${STEPS_PER_ROUND:-400}" \
        --batch_size "${BATCH_SIZE:-32}" --lr "${LR:-0.05}" --K "${K}" \
        --expect-model "${GEN_MODEL}" --tag "${KL_TAG}" \
        --cache-dir "${CACHE_DIR}" --results_dir "${RESULTS_DIR}"
done

run env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m finite_response_sail._true_to_worst_stress_eval \
    --results-dir "${RESULTS_DIR}/${TAG}" \
    --out-dir "reports/true_to_worst_stress" \
    --cache-dir "${CACHE_DIR}" \
    --rho-mis "0.05,0.075,0.10"

run env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m finite_response_sail._true_to_worst_stress_eval \
    --results-dir "${RESULTS_DIR}/${TAG}" \
    --out-dir "reports/common_worst" \
    --cache-dir "${CACHE_DIR}" \
    --rho-mis "0.05,0.075,0.10" \
    --common-worst nominal

run env PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m finite_response_sail._true_to_worst_stress_eval_kl \
    --include-results-dir "${RESULTS_DIR}/${TAG}" \
    --include-results-dir "${RESULTS_DIR}/${KL_TAG}" \
    --out-dir "reports/kl_baseline_eta01" \
    --cache-dir "${CACHE_DIR}" \
    --rho-mis "0.075"

echo "[reproduce] complete"
