#!/usr/bin/env bash
set -euo pipefail
CICS_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$CICS_PROJECT_DIR"
source "$CICS_PROJECT_DIR/.venv/bin/activate"
exec "$CICS_PROJECT_DIR/.venv/bin/python" scripts/run_analysis.py \
  --analysis-config configs/analysis_epoch17.yaml \
  --detector-config runs/reproduction_r50_ms20/eval_test/config.py \
  --checkpoint "$CICS_PROJECT_DIR/runs/reproduction_r50_ms20/test_candidates/epoch_17.pth" \
  --validation runs/reproduction_r50_ms20/eval_test/report.json \
  --selection runs/analysis_epoch17_highres/selection.json \
  --split test --phase A "$@"
