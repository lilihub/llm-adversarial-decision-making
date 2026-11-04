#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${RECOVERY_DOCKER_IMAGE:-mrtt-common-beta-recovery:4.4.1}"

docker build --pull -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
echo "Built ${IMAGE_NAME}"

