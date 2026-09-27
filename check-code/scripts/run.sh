#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-check-code:humble}"
CONTAINER_NAME="${2:-check-code-humble}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"

existing_id="$(docker container ls --all --quiet --filter "name=^/${CONTAINER_NAME}$")"

if [[ -n "$existing_id" ]]; then
    running_id="$(docker container ls --quiet --filter "name=^/${CONTAINER_NAME}$")"
    if [[ -n "$running_id" ]]; then
        echo "Контейнер '$CONTAINER_NAME' уже запущен."
        exit 0
    fi

    docker start "$CONTAINER_NAME"
else
    docker run \
        --detach \
        --name "$CONTAINER_NAME" \
        --mount "type=bind,source=$PROJECT_DIR,target=/workspace" \
        "$IMAGE_NAME" \
        sleep infinity
fi
