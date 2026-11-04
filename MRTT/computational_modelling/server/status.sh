#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)/results/recovery_common_beta_replicated/jobs"

if [[ ! -d "${JOBS_ROOT}" ]]; then
  echo "No recovery job directory exists yet: ${JOBS_ROOT}"
  exit 0
fi

for SCENARIO_DIR in "${JOBS_ROOT}"/common_beta_*; do
  [[ -d "${SCENARIO_DIR}" ]] || continue
  SCENARIO="$(basename "${SCENARIO_DIR}")"
  COMPLETED="$(find "${SCENARIO_DIR}" -name COMPLETED -type f | wc -l | tr -d ' ')"
  FAILED="$(find "${SCENARIO_DIR}" -name FAILED.txt -type f | wc -l | tr -d ' ')"
  RUNNING="$(find "${SCENARIO_DIR}" -name RUNNING.txt -type f | wc -l | tr -d ' ')"
  echo "${SCENARIO}: completed=${COMPLETED} failed=${FAILED} running=${RUNNING}"
done

