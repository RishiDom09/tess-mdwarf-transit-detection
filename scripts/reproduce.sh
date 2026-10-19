#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASELINE="${BASELINE:-.venvs/baseline/bin/python}"
CNN="${CNN:-.venvs/cnn/bin/python}"
DATA="${DATA:-data}"
RUN="${RUN:-outputs/reproduction}"
WORKERS="${WORKERS:-1}"
base() { "$BASELINE" scripts/run.py "$@" --data "$DATA" --run "$RUN"; }
cnn() { "$CNN" scripts/run.py "$@" --data "$DATA" --run "$RUN"; }
base validate
base download --role train
base prepare --role train
for seed in 1 2 3; do cnn train --seed "$seed"; done
base download --role validation
base prepare --role validation
cnn select
base download --role calibration
base prepare --role calibration
cnn cnn --role calibration
base bls --role calibration --workers "$WORKERS"
base calibrate
base download --role test
base prepare --role test
cnn cnn --role test
base bls --role test --workers "$WORKERS"
base evaluate
base figure
