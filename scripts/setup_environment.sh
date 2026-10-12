#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv --system-site-packages .venv
.venv/bin/python -c 'import torch; assert torch.__version__ == "2.9.1+cu128", "Install torch 2.9.1 and torchvision 0.24.1 from the cu128 index first"'
.venv/bin/python -m pip install -r environments/requirements.txt
mkdir -p environments/wheels
if [ ! -f environments/wheels/mmcv-2.1.0-cp311-cp311-linux_x86_64.whl ]; then
    MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4 \
        .venv/bin/python -m pip wheel mmcv==2.1.0 --no-deps --no-build-isolation --no-binary mmcv --wheel-dir environments/wheels
fi
.venv/bin/python -m pip install --no-deps environments/wheels/mmcv-2.1.0-cp311-cp311-linux_x86_64.whl
.venv/bin/python experiments/cli/check_environment.py
