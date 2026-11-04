#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Project: ${CM_ROOT}"
command -v Rscript
Rscript --version
command -v make
command -v "${CXX:-g++}" || true

Rscript -e 'pkgs <- c("rstan", "Rcpp", "ggplot2"); missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; if (length(missing)) stop("Missing R packages: ", paste(missing, collapse=", ")); cat("Required R packages are available.\n"); print(sessionInfo())'

test -f "${CM_ROOT}/data/processed/rnd_canonical.csv"
test -f "${CM_ROOT}/stan/m1_rw_beta_binomial_common_beta.stan"
echo "Canonical data and Stan source are present."

