#!/usr/bin/env bash
#SBATCH --job-name=m1cb_summary
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=computational_modelling/results/recovery_common_beta_replicated/logs/summary_%j.out
#SBATCH --error=computational_modelling/results/recovery_common_beta_replicated/logs/summary_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CM_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${R_MODULE:-}" ]]; then
  module load "${R_MODULE}"
fi

Rscript computational_modelling/scripts/07_summarise_common_beta_replicated.R

