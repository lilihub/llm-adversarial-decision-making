#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <scenario> <replicate>" >&2
  exit 2
fi

SCENARIO="$1"
REP="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${RECOVERY_DOCKER_IMAGE:-mrtt-common-beta-recovery:4.4.1}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --cpus "${RECOVERY_CPUS_PER_CONTAINER:-4}" \
  --memory "${RECOVERY_MEMORY_PER_CONTAINER:-20g}" \
  -e HOME=/tmp \
  -e RECOVERY_SCENARIO="${SCENARIO}" \
  -e RECOVERY_REP="${REP}" \
  -e RECOVERY_CHAINS="${RECOVERY_CHAINS:-4}" \
  -e RECOVERY_WARMUP="${RECOVERY_WARMUP:-500}" \
  -e RECOVERY_SAMPLING="${RECOVERY_SAMPLING:-500}" \
  -e RECOVERY_ADAPT_DELTA="${RECOVERY_ADAPT_DELTA:-0.95}" \
  -e RECOVERY_MAX_TREEDEPTH="${RECOVERY_MAX_TREEDEPTH:-12}" \
  -e RECOVERY_OVERWRITE="${RECOVERY_OVERWRITE:-0}" \
  -e SAVE_FULL_FIT="${SAVE_FULL_FIT:-0}" \
  -e SAVE_SIM_DATA="${SAVE_SIM_DATA:-0}" \
  -v "${PROJECT_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE_NAME}" \
  Rscript computational_modelling/scripts/06_fit_common_beta_recovery_job.R

