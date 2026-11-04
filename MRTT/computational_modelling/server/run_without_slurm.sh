#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CM_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

N_REPS="${RECOVERY_N_REPS:-10}"
SCENARIOS=(
  common_beta_reference
  common_beta_low_alpha_high_beta
  common_beta_high_alpha_low_beta
)

Rscript computational_modelling/scripts/05_compile_common_beta_server.R
for SCENARIO in "${SCENARIOS[@]}"; do
  for REP in $(seq 1 "${N_REPS}"); do
    RECOVERY_SCENARIO="${SCENARIO}" RECOVERY_REP="${REP}" \
      Rscript computational_modelling/scripts/06_fit_common_beta_recovery_job.R
  done
done
RECOVERY_N_REPS="${N_REPS}" \
  Rscript computational_modelling/scripts/07_summarise_common_beta_replicated.R

