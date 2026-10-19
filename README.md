# Convolutional Neural Network for Transit Detection around Active M Dwarfs in TESS Observations

Code for the research note's CNN training, BLS baseline, calibrated evaluation, and recovery figure.

## Reproduce

Requires Python 3.11 for data preparation/BLS and Python 3.8 on an x86-64 machine for TensorFlow 2.3.1. The original CNN ran on an Intel Mac; identical retraining results across hardware are not guaranteed. See [METHODS.md](METHODS.md) for the runtime caveat and scientific settings.

```sh
bash scripts/setup.sh
.venvs/baseline/bin/python scripts/run.py validate
bash scripts/reproduce.sh
```

This downloads the fixed TESS products, prepares injections, trains three CNN seeds, selects a checkpoint, calibrates thresholds, evaluates the held-out stars, and generates `outputs/reproduction/figures/recovery_lines.pdf`. Full reproduction is computationally expensive. `WORKERS=4 bash scripts/reproduce.sh` enables parallel BLS searches; choose a worker count your machine can support.

- `src/`: experiment code.
- `configs/`: exact settings, seeds, and dependency pins.
- `manifests/`: 350 target identities, fixed splits, FITS checksums, and injection parameters.
- `scripts/`: setup and reproduction commands.
- `tests/`: focused reproducibility checks.

Downloaded light curves, checkpoints, and results stay in ignored `data/` and `outputs/` folders. Completed stages are verified and reused. Interrupted CNN training requires moving the failed seed directory and restarting that seed. Individual stages are available through `python scripts/run.py --help`.
