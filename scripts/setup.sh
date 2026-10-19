#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASELINE_PYTHON="${BASELINE_PYTHON:-python3.11}"
CNN_PYTHON="${CNN_PYTHON:-python3.8}"
"$BASELINE_PYTHON" -m venv .venvs/baseline
.venvs/baseline/bin/python -m pip install -r configs/requirements-baseline.txt
"$CNN_PYTHON" -m venv .venvs/cnn
.venvs/cnn/bin/python -m pip install 'pip<25.1'
# The historical TF2.3.1/NumPy1.19.1 runtime has a known metadata conflict.
# Install the explicitly pinned, previously exercised environment as recorded.
.venvs/cnn/bin/python -m pip install --no-deps -r configs/requirements-cnn.txt
