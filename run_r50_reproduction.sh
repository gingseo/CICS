#!/usr/bin/env bash
set -euo pipefail
CICS_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CICS_PROJECT_DIR"
source "$CICS_PROJECT_DIR/.venv/bin/activate"
exec "$CICS_PROJECT_DIR/.venv/bin/python" scripts/reproduce_r50.py "$@"
