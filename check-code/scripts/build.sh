#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-check-code:humble}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$SCRIPT_DIR")"

docker build --tag "$IMAGE_NAME" "$PROJECT_DIR"
