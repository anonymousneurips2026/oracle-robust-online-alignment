#!/usr/bin/env bash
# Shared helpers for finite-response release scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODULE_DIR="${REPO_ROOT}/finite_response_sail"
PY="${PYTHON:-python}"
CACHE_DIR="${CACHE_DIR:-${MODULE_DIR}/cache}"
RESULTS_DIR="${RESULTS_DIR:-${MODULE_DIR}/results}"

format_float() {
    "${PY}" - "$1" <<'PY'
import sys
print(format(float(sys.argv[1]), "g"))
PY
}

build_run_id() {
    local rho_mis="$1"
    local rho_alg="$2"
    local seed="$3"
    local eta_kl="${4:-0.0}"
    local K="${5:-12}"
    "${PY}" - "$rho_mis" "$rho_alg" "$seed" "$eta_kl" "$K" <<'PY'
import sys
rho_mis, rho_alg, seed, eta_kl, K = sys.argv[1:]
base = (
    "mode-biased_feedback_obj-exact_fs-controlled_ref-base_lm_lognorm"
    "_bias-redundancy"
    f"_rhoMis-{format(float(rho_mis), 'g')}_rhoAlg-{format(float(rho_alg), 'g')}"
    f"_beta-0.1_tau-2_R-10_K-{int(K)}_seed-{int(seed)}"
)
if float(eta_kl) > 0.0:
    base += f"_etaKL-{format(float(eta_kl), 'g')}"
print(base)
PY
}

build_run_dir() {
    local tag="$1"
    local rho_mis="$2"
    local rho_alg="$3"
    local seed="$4"
    local eta_kl="${5:-0.0}"
    local K="${6:-12}"
    printf "%s/%s/%s\n" "${RESULTS_DIR}" "${tag}" "$(build_run_id "$rho_mis" "$rho_alg" "$seed" "$eta_kl" "$K")"
}

is_run_complete() {
    local run_dir="$1"
    [ -s "${run_dir}/metrics.json" ] \
        && [ -s "${run_dir}/theta.pt" ] \
        && [ -s "${run_dir}/oracle_tensors_test.npz" ]
}

launch_train() {
    local tag="$1"
    local rho_mis="$2"
    local q="$3"
    local seed="$4"
    local eta_kl="${5:-0.0}"
    local K="${K:-12}"
    local rho_alg
    rho_alg="$("${PY}" - "$rho_mis" "$q" <<'PY'
import sys
print(format(float(sys.argv[1]) * float(sys.argv[2]), "g"))
PY
)"

    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" -m finite_response_sail.train \
        --mode biased_feedback \
        --objective exact \
        --featureset controlled \
        --ref base_lm_lognorm \
        --bias_attr redundancy \
        --rho_mis "${rho_mis}" \
        --rho_alg "${rho_alg}" \
        --eta_kl "${eta_kl}" \
        --beta 0.1 \
        --tau 2.0 \
        --R 10.0 \
        --seed "${seed}" \
        --rounds "${ROUNDS:-6}" \
        --steps_per_round "${STEPS_PER_ROUND:-400}" \
        --batch_size "${BATCH_SIZE:-32}" \
        --lr "${LR:-0.05}" \
        --K "${K}" \
        --expect-model "${EXPECT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" \
        --tag "${tag}" \
        --cache-dir "${CACHE_DIR}" \
        --results_dir "${RESULTS_DIR}" \
        ${TRAIN_EXTRA_ARGS:-}
}
