#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CM_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p computational_modelling/results/recovery_common_beta_replicated/logs

N_REPS="${RECOVERY_N_REPS:-10}"
MAX_CONCURRENT="${RECOVERY_MAX_CONCURRENT:-5}"
TOTAL_TASKS=$((3 * N_REPS))

COMPILE_JOB="$(sbatch --parsable computational_modelling/server/slurm_compile.sbatch)"
ARRAY_JOB="$(sbatch --parsable \
  --dependency="afterok:${COMPILE_JOB}" \
  --array="1-${TOTAL_TASKS}%${MAX_CONCURRENT}" \
  --export="ALL,RECOVERY_N_REPS=${N_REPS}" \
  computational_modelling/server/slurm_recovery_array.sbatch)"
SUMMARY_JOB="$(sbatch --parsable \
  --dependency="afterok:${ARRAY_JOB}" \
  --export="ALL,RECOVERY_N_REPS=${N_REPS}" \
  computational_modelling/server/slurm_summarise.sbatch)"

echo "Compile job: ${COMPILE_JOB}"
echo "Recovery array job: ${ARRAY_JOB} (${TOTAL_TASKS} tasks, max ${MAX_CONCURRENT} concurrent)"
echo "Summary job: ${SUMMARY_JOB}"
echo "Monitor with: squeue -j ${COMPILE_JOB},${ARRAY_JOB},${SUMMARY_JOB}"

