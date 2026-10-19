# A Convolutional Neural Network for Transit Detection around Active M Dwarfs Using TESS Observations

Code to reproduce the experiment accompanying this research note, including CNN training, the Box Least Squares (BLS) baseline, and evaluation.

**Status:** Repository preparation in progress. The experiment code has not yet been consolidated here; installation and run commands will be added after verification.

## Reproduction

1. Install the pinned dependencies.
2. Download the TESS light curves and Newton et al. (2017) activity catalog using the supplied target and split manifests.
3. Prepare the light curves, inject transits, and train and select the CNN.
4. Calibrate CNN and BLS thresholds on the separate calibration stars.
5. Evaluate both detectors on the fixed test stars and generate the paper’s figure.

The release will include the experiment settings, random seeds, and manifests. Downloaded data and generated outputs will be stored locally and excluded from Git.
