#!/usr/bin/env bash
#SBATCH --job-name=m1cb_recovery
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=computational_modelling/results/recovery_common_beta_replicated/logs/recovery_%A_%a.out
#SBATCH --error=computational_modelling/results/recovery_common_beta_replicated/logs/recovery_%A_%a.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CM_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${R_MODULE:-}" ]]; then
  module load "${R_MODULE}"
fi

N_REPS="${RECOVERY_N_REPS:-10}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"
ZERO_BASED=$((TASK_ID - 1))
SCENARIO_INDEX=$((ZERO_BASED / N_REPS))
REP=$((ZERO_BASED % N_REPS + 1))

SCENARIOS=(
  common_beta_reference
  common_beta_low_alpha_high_beta
  common_beta_high_alpha_low_beta
)

if (( SCENARIO_INDEX < 0 || SCENARIO_INDEX >= ${#SCENARIOS[@]} )); then
  echo "Invalid array task ${TASK_ID} for ${N_REPS} replicates" >&2
  exit 2
fi

export RECOVERY_SCENARIO="${SCENARIOS[$SCENARIO_INDEX]}"
export RECOVERY_REP="${REP}"
export RECOVERY_CHAINS="${RECOVERY_CHAINS:-4}"
export RECOVERY_WARMUP="${RECOVERY_WARMUP:-500}"
export RECOVERY_SAMPLING="${RECOVERY_SAMPLING:-500}"
export RECOVERY_ADAPT_DELTA="${RECOVERY_ADAPT_DELTA:-0.95}"
export RECOVERY_MAX_TREEDEPTH="${RECOVERY_MAX_TREEDEPTH:-12}"
export SAVE_FULL_FIT="${SAVE_FULL_FIT:-0}"
export SAVE_SIM_DATA="${SAVE_SIM_DATA:-0}"

Rscript computational_modelling/scripts/06_fit_common_beta_recovery_job.R

