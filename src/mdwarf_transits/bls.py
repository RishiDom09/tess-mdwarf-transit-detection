"""Truth-blind, unmodified Astropy search of preprocessed flux."""
import time
import warnings
import numpy as np
from astropy.timeseries import BoxLeastSquares
from .settings import bls_settings as proposal


def grids(original_time):
    c = proposal()
    t = np.asarray(original_time, float)
    if t.ndim != 1 or len(t) < 30 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError('Invalid original time support')
    if np.ptp(t) < 2*c['period_max_days']:
        raise ValueError('Insufficient span for the fixed period range')
    durations = np.geomspace(c['duration_min_days'], c['duration_max_days'], c['duration_count'])
    periods = BoxLeastSquares(t-t[0], np.ones(len(t))).autoperiod(
        durations, minimum_period=c['period_min_days'], maximum_period=c['period_max_days'],
        frequency_factor=c['frequency_factor'])
    # Astropy rounds its frequency count; discard a possible overshoot, without adding an off-grid endpoint.
    periods = periods[(periods >= c['period_min_days']) & (periods <= c['period_max_days'])]
    return periods, durations


def score_spectrum(depth_snr, count):
    p = np.asarray(depth_snr, float)
    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid BLS spectrum; no failure-to-zero conversion')
    sr = p / np.sqrt(count)
    scatter = float(np.std(sr, ddof=0))
    if scatter <= 0:
        raise ValueError('Flat BLS spectrum; SDE undefined, not a zero-score detection')
    return float((sr.max()-sr.mean())/scatter), sr


def search(time_days, flux, periods, durations):
    start = time.monotonic()
    t, y = np.asarray(time_days, float), np.asarray(flux, float)
    p, d = np.asarray(periods, float), np.asarray(durations, float)
    if (t.ndim != 1 or t.shape != y.shape or len(t) < 30 or not np.isfinite(t).all()
            or not np.isfinite(y).all() or (np.diff(t) <= 0).any() or (y <= 0).any()):
        raise ValueError('Invalid input; no implicit dropping or clipping')
    if (not np.isfinite(p).all() or not np.isfinite(d).all() or len(p) < 2
            or len(d) < 1 or (np.diff(p) <= 0).any() or (np.diff(d) <= 0).any()
            or min(d) <= 0 or max(d) >= min(p)):
        raise ValueError('Invalid period/duration grid')
    c = proposal()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = BoxLeastSquares(t-t[0], y, dy=None).power(
            p, d, objective=c['objective'], method=c['method'], oversample=c['oversample'])
    score, sr = score_spectrum(result.depth_snr, len(t))
    best = int(np.argmax(sr))
    values = {k: float(result[k][best]) for k in ('period', 'duration', 'transit_time', 'depth', 'depth_err')}
    values['transit_time'] += float(t[0])
    if not np.isfinite(list(values.values())).all() or values['depth'] <= 0:
        raise ValueError('Nonfinite or non-dimming best fit')
    return dict(status='fit', score=score, **values, points=len(t), periods=len(p),
                best_duration_at_edge=bool(result.duration[best] <= d.min() or result.duration[best] >= d.max()),
                seconds=time.monotonic()-start, warnings=sorted(set(str(w.message) for w in caught))), dict(
                    periods=p, durations=d, sr=sr, depth_snr=np.asarray(result.depth_snr),
                    best_duration_by_period=np.asarray(result.duration),
                    best_epoch_by_period=np.asarray(result.transit_time)+t[0])
