#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${RECOVERY_DOCKER_IMAGE:-mrtt-common-beta-recovery:4.4.1}"

command -v docker
docker version
docker image inspect "${IMAGE_NAME}" >/dev/null

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "${PROJECT_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE_NAME}" \
  Rscript -e 'pkgs <- c("rstan", "Rcpp", "ggplot2"); stopifnot(all(vapply(pkgs, requireNamespace, logical(1), quietly=TRUE))); print(sessionInfo()); stopifnot(file.exists("computational_modelling/data/processed/rnd_canonical.csv")); cat("Docker recovery environment OK\n")'

