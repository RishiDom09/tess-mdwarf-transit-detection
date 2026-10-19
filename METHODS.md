# Reproduction contract

This repository reproduces the paper's fixed sample and primary CNN/BLS experiment. It does not rerun exploratory method selection or select a new sample from today's catalogs. Catalog identities, quality flags, H-alpha measurements, source product checksums, and final split assignments are in `manifests/stars.csv`. Original split names in injection IDs are provenance; the `role` column is authoritative. The sample has 100 active/100 inactive training stars, 20/20 validation stars, 60 active calibration stars, and 50 active test stars. The stored trial manifests retain all attempts, including five ineligible training injections and four ineligible test injections.

## Settings and sources

`configs/experiment.json` is authoritative for numerical values. R means replicated published/author-code setting; A means instrument or implementation adaptation; P means a project choice.

| Component | Adopted definition and source |
|---|---|
| Activity | R/A: Newton et al. (2017), Section 2 and the published catalog; H-alpha EW below -1 angstrom denotes active, with catalog identities and measurement-limit flags checked. Yaptangco et al. (2025), Section 2.1 motivates H-alpha labels. Fixed project roster and catalog screening are P. |
| Photometry | P: one recorded SPOC 120-second SAP product per star, finite time/flux, `(QUALITY & 175) == 0`, positive flux/errors, at least 20 days of retained exposure support and span, CROWDSAP at least 0.90. Divide by the background median before injection. FITS hashes prevent silent use of reprocessed products. |
| Transits | R/A: batman (Kreidberg 2015, Section 2), circular orbits and quadratic limb darkening. Coefficients were interpolated from Claret (2017), table 15, at the recorded stellar parameters. Coefficients and geometry are frozen in each trial; no new interpolation is performed. |
| Injection grid | P: five logarithmic periods from 0.8–5.7 days, five radii from 0.5–3 Earth radii, ten phases per cell; phase steps 0.125 day from the recorded initial phase. Non-grazing impact parameters and geometry are recorded per cell. Exposure integration uses 63 midpoint samples. Flux is multiplied by `1 + CROWDSAP * (model - 1)` and quantized to float32 before binning. Eligibility requires at least three events, each with one retained center and at least one-third duration coverage; recorded eligibility is retained. |
| CNN architecture | R: Olmschenk et al. (2021), Section 3.3/Figure 3 and RAMjET `FfiHades`, pinned commit in THIRD_PARTY_NOTICES.md. All-convolutional head, one flux channel, 1,000 positions, leaky slope 0.01, ordinary batch normalization, original dropout placement. |
| CNN input | R: Olmschenk et al., Section 3.5 and author preprocessing: training deletion then random circular roll, wrap/truncate to 1,000, then linear 10th/90th percentile scaling to -1/+1. Evaluation omits deletion and roll. A: complete 15-cadence arithmetic means, explicit float64 preprocessing and float32 network input. Gaps and incomplete bins are not interpolated. |
| Training | R: author `train.py` Adam learning rate 1e-4, probability-output binary cross-entropy, batch 1,000 and window shift 100. P: fresh weights, activity/star/label-balanced sampling, three recorded seed triples, 19,032 updates, fixed checkpoint list. Select by active validation global star/label-balanced AUROC; ties within 1e-12 prefer earlier update then lower seed index. |
| BLS | R/A: Kovacs et al. (2002), Section 2 equations 1–5 and Section 3 equation 6; pinned Astropy implementation. P: 0.5–6-day periods, 35 logarithmic durations from 10 minutes to 4 hours, frequency factor 1, oversample 10, fast SNR objective. SDE uses `(max(SR)-mean(SR))/std(SR, ddof=0)` over every period, with `SR=depth_snr/sqrt(N)`. |
| Detrending | R/A: Hippke et al. (2019), biweight method and Wotan implementation. P: 0.5-day window, cval 5, break tolerance 0.25 day, edge cutoff 0. Upper-only five-iteration 2-sigma clipping uses median and MAD scatter. A: remove timing-defined singleton segments before Wotan; use original represented support for the median and period grid. BLS receives each represented bin once, without CNN wrap duplicates. |
| Calibration | R: Tong et al. (2018), umbrella algorithm, Algorithm 1, Proposition 1, equations 1–2. P: 60 independent active-star controls, alpha=delta=0.05, rank 60 (maximum), strict `score > threshold`, separate conditional coverage per detector. Model selection precedes calibration; thresholds precede final testing. |
| Evaluation | P: flagging rather than period recovery; average within each star, then weight 50 stars equally. Use 10,000 paired whole-star bootstrap draws, seed 2026100405, linear 2.5th/97.5th percentiles. Control intervals are exact two-sided 95% binomial intervals. Global star-balanced AUROC is secondary. Figure cells average stars first, then period/radius cells as in the paper. |

Uninjected controls are not certified planet-free. Calibration coverage assumes representative independent controls and is not a joint 95% guarantee for both detectors. Uncertainty conditions on the selected model and thresholds.

## Runtime and portability

Data preparation/BLS use Python 3.11 and NumPy 1.26.4. CNN training/inference use Python 3.8, TensorFlow 2.3.1, NumPy 1.19.1, h5py 2.10.0, and protobuf 3.20.3. TensorFlow's package metadata specifies NumPy below 1.19; the historical experiment explicitly installed NumPy 1.19.1 despite that conflict. `setup.sh` reproduces that recorded environment with `--no-deps`; this is deliberate, and `pip check` will report the conflict. TensorFlow 2.3.1 has no native Apple Silicon wheels. Do not substitute a modern TensorFlow version and assume equivalent training.

The original CNN used CPU execution on an Intel Mac, four intra-op threads, one inter-op thread, and inference batches of 32. BLAS/OpenMP threads are set to one by the entry script. New training on different hardware can produce different trajectories; bitwise reproduction of the reported model is not promised. Trained weights are not bundled. Downloaded FITS files are checked against recorded SHA-256 values; changed upstream products cause a stop, not replacement. Frozen trial parameters avoid rerunning catalog selection, random geometry, or limb-darkening interpolation.

The publication package replaces machine-specific paths, VM transfer routines, historical receipt chains, and benchmark gates with explicit local commands and per-run hashes/receipts. Scientific functions were extracted from the experiment packages; their source file hashes and import changes are in `manifests/source_map.json`. Exploratory cutoffs and unrelated detector arms are omitted. Reusing a run after changing code/settings/manifests is rejected. No training or detector search runs on import.

## References

- Newton et al. (2017): https://doi.org/10.3847/1538-4357/834/1/85
- Yaptangco et al. (2025): https://doi.org/10.3847/1538-3881/ada898
- Olmschenk et al. (2021): https://doi.org/10.3847/1538-3881/abf4c6
- Kreidberg (2015): https://doi.org/10.1086/683602
- Claret (2017): https://doi.org/10.1051/0004-6361/201629705
- Kovacs et al. (2002): https://doi.org/10.1051/0004-6361:20020802
- Hippke et al. (2019): https://doi.org/10.3847/1538-3881/ab3984
- Tong et al. (2018): https://doi.org/10.1126/sciadv.aao1659

## Package verification

The consolidation was checked against the original implementation: regenerated inputs for two stars in each of the four roles matched the archived/reference arrays exactly; seeded CNN initialization, inference, one optimizer update, and sampler batches matched exactly; BLS preprocessing, grids, and search scores matched on a generated signal. A small synthetic run exercised training, checkpoint selection, calibration, paired evaluation, and figure generation. The fixed manifests and stage receipts have eight focused tests (`python -m pytest`, with pytest 8.3.5 installed in the baseline environment). One MAST product download was verified against its recorded checksum.

These checks establish the package's tested equivalence and orchestration boundaries, not a new full scientific run. The full 19,032-update, three-seed experiment and a clean installation of both environments were not repeated during consolidation. Training parameters are read with round-trip float conversion to preserve the original generated injections; validation retains the original pandas CSV conversion. Scientific results and model weights are not included in the repository.
