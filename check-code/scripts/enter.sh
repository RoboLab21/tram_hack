#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-check-code-humble}"

docker exec -it "$CONTAINER_NAME" bash -lc \
    'source /opt/ros/humble/setup.bash; if [[ -f /workspace/install/setup.bash ]]; then source /workspace/install/setup.bash; fi; exec bash'
