#!/usr/bin/env bash
#SBATCH --job-name=m1cb_compile
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=computational_modelling/results/recovery_common_beta_replicated/logs/compile_%j.out
#SBATCH --error=computational_modelling/results/recovery_common_beta_replicated/logs/compile_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CM_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${R_MODULE:-}" ]]; then
  module load "${R_MODULE}"
fi

Rscript computational_modelling/scripts/05_compile_common_beta_server.R

