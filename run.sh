#!/usr/bin/env bash
# Launch the GUI and its workers using this project's virtual environment.
set -euo pipefail

CICS_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CICS_VENV_DIR="${CICS_PROJECT_DIR}/.venv"

if [[ ! -x "${CICS_VENV_DIR}/bin/python" || ! -f "${CICS_VENV_DIR}/bin/activate" ]]; then
    printf '프로젝트 가상환경이 없습니다: %s\n먼저 실행하세요: bash "%s/scripts/setup_environment.sh"\n' \
        "${CICS_VENV_DIR}" "${CICS_PROJECT_DIR}" >&2
    exit 1
fi

cd -- "${CICS_PROJECT_DIR}"
source "${CICS_VENV_DIR}/bin/activate"

exec "${CICS_VENV_DIR}/bin/python" -m streamlit run \
    "${CICS_PROJECT_DIR}/app/streamlit_app.py" \
    --server.address "${CICS_HOST:-0.0.0.0}" \
    --server.port "${CICS_PORT:-8501}" \
    --server.headless true \
    --browser.gatherUsageStats false \
    "$@"
