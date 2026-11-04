#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${RECOVERY_DOCKER_IMAGE:-mrtt-common-beta-recovery:4.4.1}"
N_REPS="${RECOVERY_N_REPS:-10}"
MAX_CONCURRENT="${RECOVERY_MAX_CONCURRENT:-3}"
CPUS_PER_CONTAINER="${RECOVERY_CPUS_PER_CONTAINER:-4}"
MEMORY_PER_CONTAINER="${RECOVERY_MEMORY_PER_CONTAINER:-20g}"
CHAINS="${RECOVERY_CHAINS:-4}"
WARMUP="${RECOVERY_WARMUP:-500}"
SAMPLING="${RECOVERY_SAMPLING:-500}"
ADAPT_DELTA="${RECOVERY_ADAPT_DELTA:-0.95}"
MAX_TREEDEPTH="${RECOVERY_MAX_TREEDEPTH:-12}"
BASE_SEED="${RECOVERY_BASE_SEED:-20261006}"
FIT_BASE_SEED="${RECOVERY_FIT_BASE_SEED:-8675309}"
SAVE_FULL="${SAVE_FULL_FIT:-0}"
SAVE_SIM="${SAVE_SIM_DATA:-0}"
OVERWRITE="${RECOVERY_OVERWRITE:-0}"

if ! [[ "${N_REPS}" =~ ^[1-9][0-9]*$ && "${MAX_CONCURRENT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RECOVERY_N_REPS and RECOVERY_MAX_CONCURRENT must be positive integers" >&2
  exit 2
fi

command -v docker >/dev/null || { echo "docker not found" >&2; exit 127; }
docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1 || {
  echo "Docker image not found: ${IMAGE_NAME}. Run build_image.sh first." >&2
  exit 2
}

RESULT_ROOT="${PROJECT_ROOT}/computational_modelling/results/recovery_common_beta_replicated"
LOG_DIR="${RESULT_ROOT}/logs/docker"
mkdir -p "${LOG_DIR}"

docker_base=(
  docker run --rm
  --user "$(id -u):$(id -g)"
  --cpus "${CPUS_PER_CONTAINER}"
  --memory "${MEMORY_PER_CONTAINER}"
  -e HOME=/tmp
  -v "${PROJECT_ROOT}:/workspace"
  -w /workspace
)

echo "Compiling or loading the server-specific Stan cache..."
if ! "${docker_base[@]}" \
  -e RECOMPILE_STAN="${RECOMPILE_STAN:-0}" \
  "${IMAGE_NAME}" \
  Rscript computational_modelling/scripts/05_compile_common_beta_server.R \
  >"${LOG_DIR}/compile.log" 2>&1; then
  echo "Stan compilation failed. See ${LOG_DIR}/compile.log" >&2
  exit 1
fi

SCENARIOS=(
  common_beta_reference
  common_beta_low_alpha_high_beta
  common_beta_high_alpha_low_beta
)

run_one() {
  local scenario="$1"
  local rep="$2"
  local log_file="${LOG_DIR}/${scenario}_rep_$(printf '%03d' "${rep}").log"
  "${docker_base[@]}" \
    -e RECOVERY_SCENARIO="${scenario}" \
    -e RECOVERY_REP="${rep}" \
    -e RECOVERY_CHAINS="${CHAINS}" \
    -e RECOVERY_WARMUP="${WARMUP}" \
    -e RECOVERY_SAMPLING="${SAMPLING}" \
    -e RECOVERY_ADAPT_DELTA="${ADAPT_DELTA}" \
    -e RECOVERY_MAX_TREEDEPTH="${MAX_TREEDEPTH}" \
    -e RECOVERY_BASE_SEED="${BASE_SEED}" \
    -e RECOVERY_FIT_BASE_SEED="${FIT_BASE_SEED}" \
    -e SAVE_FULL_FIT="${SAVE_FULL}" \
    -e SAVE_SIM_DATA="${SAVE_SIM}" \
    -e RECOVERY_OVERWRITE="${OVERWRITE}" \
    "${IMAGE_NAME}" \
    Rscript computational_modelling/scripts/06_fit_common_beta_recovery_job.R \
    >"${log_file}" 2>&1
}

active=0
failures=0
for scenario in "${SCENARIOS[@]}"; do
  for rep in $(seq 1 "${N_REPS}"); do
    echo "Launching ${scenario} replicate ${rep}"
    run_one "${scenario}" "${rep}" &
    active=$((active + 1))
    if (( active >= MAX_CONCURRENT )); then
      if ! wait -n; then failures=$((failures + 1)); fi
      active=$((active - 1))
    fi
  done
done

while (( active > 0 )); do
  if ! wait -n; then failures=$((failures + 1)); fi
  active=$((active - 1))
done

echo "Recovery containers finished with ${failures} failed task(s). Running summary."
if ! "${docker_base[@]}" \
  -e RECOVERY_N_REPS="${N_REPS}" \
  "${IMAGE_NAME}" \
  Rscript computational_modelling/scripts/07_summarise_common_beta_replicated.R \
  >"${LOG_DIR}/summary.log" 2>&1; then
  echo "Summary failed or no jobs completed. See ${LOG_DIR}/summary.log" >&2
  exit 1
fi

echo "Summary complete."
echo "Tables: ${RESULT_ROOT}/summary"
echo "Figures: ${PROJECT_ROOT}/computational_modelling/figures/recovery_common_beta_replicated"
if (( failures > 0 )); then exit 1; fi

