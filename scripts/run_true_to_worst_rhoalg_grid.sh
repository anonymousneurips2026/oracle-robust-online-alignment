#!/usr/bin/env bash
# Train the rho_alg grid for the true-to-worst stress experiment.
#
# Iterates over (rho_mis, q, seed) triples where q = rho_alg / rho_mis.
# Honors --dry-run, --force, --seeds, --rho-mis-list, --q-list. Skips runs
# whose artifact set is already complete (per is_run_complete in
# scripts/_run_helpers.sh).
#
# By default this release trains the requested grid from the active cache.
# Optional --reuse-tags can be passed by a local user, but legacy artifacts are
# not used by default for anonymous reproduction.
#
# Usage:
#   bash scripts/run_true_to_worst_rhoalg_grid.sh \
#       [--rho-mis 0.075]            # single rho_mis (overrides --rho-mis-list)
#       [--rho-mis-list "0.05 0.075 0.10"]
#       [--q-list "0 0.025 0.05 0.10 0.25 0.50 1.00"]
#       [--seeds 0-9] [--gpu-ids 0,1] [--force] [--dry-run]
#
# Honors CUDA_VISIBLE_DEVICES.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_run_helpers.sh
source "${SCRIPT_DIR}/_run_helpers.sh"

TAG="v4_true_to_worst_rhoalg_grid"
REUSE_TAGS_SPEC=""
RHO_MIS_LIST_SPEC="0.05 0.075 0.10"
RHO_MIS_SINGLE=""
Q_LIST_SPEC="0 0.025 0.05 0.10 0.25 0.50 1.00"
SEEDS_SPEC="0-9"
FORCE=0
DRY_RUN=0
GPU_IDS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG="$2"; shift 2;;
        --reuse-tags) REUSE_TAGS_SPEC="$2"; shift 2;;
        --rho-mis) RHO_MIS_SINGLE="$2"; shift 2;;
        --rho-mis-list) RHO_MIS_LIST_SPEC="$2"; shift 2;;
        --q-list) Q_LIST_SPEC="$2"; shift 2;;
        --seeds) SEEDS_SPEC="$2"; shift 2;;
        --gpu-ids) GPU_IDS="$2"; shift 2;;
        --force) FORCE=1; shift;;
        --dry-run) DRY_RUN=1; shift;;
        -h|--help)
            sed -n '1,22p' "$0"
            exit 0;;
        *)
            echo "[error] unknown flag: $1" >&2
            sed -n '1,22p' "$0" >&2
            exit 2;;
    esac
done

if [ -n "${RHO_MIS_SINGLE}" ]; then
    RHO_MIS_LIST_SPEC="${RHO_MIS_SINGLE}"
fi

LO="${SEEDS_SPEC%-*}"; HI="${SEEDS_SPEC#*-}"
SEEDS=$(seq "${LO}" "${HI}")

read -ra RHO_MIS_LIST <<< "${RHO_MIS_LIST_SPEC}"
read -ra Q_LIST <<< "${Q_LIST_SPEC}"
read -ra REUSE_TAGS <<< "${REUSE_TAGS_SPEC}"

echo "[ttw-grid] tag=${TAG} reuse_tags=[${REUSE_TAGS[*]:-}] rho_mis_list=[${RHO_MIS_LIST[*]}] q_list=[${Q_LIST[*]}] seeds=[${SEEDS_SPEC}]"
echo "[ttw-grid] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

cd "${REPO_ROOT}"

mkdir -p "${MODULE_DIR}/logs/${TAG}"

# Build the list of (rho_mis, q, seed) jobs that aren't already complete in
# either the active tag or any of the reuse tags.
declare -a JOBS_RM=()
declare -a JOBS_Q=()
declare -a JOBS_SEED=()
declare -a JOBS_RA=()
skipped=0
for rm in "${RHO_MIS_LIST[@]}"; do
    for q in "${Q_LIST[@]}"; do
        rho_alg_abs=$("${PY}" -c "print(format(float('${q}') * float('${rm}'), 'g'))")
        for seed in ${SEEDS}; do
            already=0
            if [ "${FORCE}" = "0" ]; then
                # Check the active tag.
                rd="$(build_run_dir "${TAG}" "${rm}" "${rho_alg_abs}" "${seed}")"
                if is_run_complete "${rd}"; then already=1; fi
                # Check each reuse tag.
                if [ "${already}" = "0" ]; then
                    for rt in "${REUSE_TAGS[@]:-}"; do
                        [ -z "${rt}" ] && continue
                        rd2="$(build_run_dir "${rt}" "${rm}" "${rho_alg_abs}" "${seed}")"
                        if is_run_complete "${rd2}"; then already=1; break; fi
                    done
                fi
            fi
            if [ "${already}" = "1" ]; then
                skipped=$((skipped + 1))
                continue
            fi
            JOBS_RM+=("${rm}")
            JOBS_Q+=("${q}")
            JOBS_SEED+=("${seed}")
            JOBS_RA+=("${rho_alg_abs}")
        done
    done
done

total=${#JOBS_RM[@]}
echo "[ttw-grid] jobs to run: ${total}; already-complete: ${skipped}"

if [ "${DRY_RUN}" = "1" ]; then
    for i in $(seq 0 $((total - 1))); do
        echo "[dry-run] would train: rho_mis=${JOBS_RM[$i]} q=${JOBS_Q[$i]} (rho_alg=${JOBS_RA[$i]}) seed=${JOBS_SEED[$i]}"
    done
    echo "[ttw-grid] done: launched=0 skipped=${skipped} failed=0 (dry-run)"
    exit 0
fi

if [ "${total}" = "0" ]; then
    echo "[ttw-grid] nothing to do."
    exit 0
fi

# Resolve GPU slots. Default: split CUDA_VISIBLE_DEVICES, else auto-detect, else CPU.
declare -a GPU_LIST
if [ -n "${GPU_IDS}" ]; then
    IFS=',' read -ra GPU_LIST <<< "${GPU_IDS}"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
else
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_LIST=()
        while IFS= read -r line; do
            GPU_LIST+=("${line}")
        done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    else
        GPU_LIST=("")
    fi
fi
n_slots=${#GPU_LIST[@]}
[ "${n_slots}" -lt 1 ] && n_slots=1
echo "[ttw-grid] GPU slots: [${GPU_LIST[*]:-CPU}]  n_slots=${n_slots}"

declare -a SLOT_PID
declare -a SLOT_DESC
for s in $(seq 0 $((n_slots - 1))); do SLOT_PID[$s]=0; SLOT_DESC[$s]=""; done

next=0
finished=0
launched=0
failed=0
while [ ${finished} -lt ${total} ]; do
    for s in $(seq 0 $((n_slots - 1))); do
        if [ ${next} -ge ${total} ]; then break; fi
        if [ "${SLOT_PID[$s]}" = "0" ]; then
            rm="${JOBS_RM[$next]}"
            q="${JOBS_Q[$next]}"
            seed="${JOBS_SEED[$next]}"
            gpu="${GPU_LIST[$s]}"
            log_f="${MODULE_DIR}/logs/${TAG}/rm${rm}_q${q}_seed${seed}.log"
            desc="rho_mis=${rm} q=${q} seed=${seed} gpu=${gpu}"
            echo "[ttw-grid] launch (${next}/${total}) ${desc} -> ${log_f}"
            CUDA_VISIBLE_DEVICES="${gpu}" \
                bash -c "$(declare -f launch_train is_run_complete build_run_dir)
                         export PY=\"${PY}\" REPO_ROOT=\"${REPO_ROOT}\" MODULE_DIR=\"${MODULE_DIR}\" CACHE_DIR=\"${CACHE_DIR}\" RESULTS_DIR=\"${RESULTS_DIR}\"
                         launch_train '${TAG}' '${rm}' '${q}' '${seed}' '0.0'" \
                > "${log_f}" 2>&1 &
            SLOT_PID[$s]=$!
            SLOT_DESC[$s]="${desc}"
            launched=$((launched + 1))
            next=$((next + 1))
        fi
    done
    wait -n 2>/dev/null || true
    for s in $(seq 0 $((n_slots - 1))); do
        pid="${SLOT_PID[$s]}"
        if [ "${pid}" != "0" ] && ! kill -0 "${pid}" 2>/dev/null; then
            wait "${pid}" 2>/dev/null
            rc=$?
            if [ "${rc}" != "0" ]; then
                echo "[ttw-grid] FAIL ${SLOT_DESC[$s]} rc=${rc}" >&2
                failed=$((failed + 1))
            else
                echo "[ttw-grid] OK   ${SLOT_DESC[$s]}"
            fi
            SLOT_PID[$s]=0
            SLOT_DESC[$s]=""
            finished=$((finished + 1))
        fi
    done
done

echo "[ttw-grid] done: launched=${launched} skipped=${skipped} failed=${failed}"
[ "${failed}" = "0" ]
