"""Frozen BLS detrending, singleton removal, and upper clipping."""
import numpy as np

def singleton_mask(time):
    """Timing-only removal of one-bin Wotan segments, including endpoints.

    Necessary adaptation for Wotan 1.10's empty singleton window; not a
    general NaN filter. A gap exactly equal to break_tolerance stays connected.
    Shared by calibration and the future Step 6 adapter.
    """
    from .settings import BIWEIGHT_SETTINGS
    t = np.asarray(time, dtype=np.float64)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError('Invalid singleton-mask timestamps')
    cuts = np.r_[0, np.flatnonzero(np.diff(t) > BIWEIGHT_SETTINGS['break_tolerance'])+1, len(t)]
    removed = np.zeros(len(t), dtype=bool)
    removed[cuts[:-1][np.diff(cuts) == 1]] = True
    return removed

def baseline_arrays(arrays):
    from wotan import flatten
    from .clipping import clip_upper
    from .settings import BIWEIGHT_SETTINGS
    from .bls import grids
    t, f = arrays['time'], arrays['flux']
    t, f = np.asarray(t, dtype=np.float64), np.asarray(f, dtype=np.float64)
    isolated = singleton_mask(t)
    if f.shape != t.shape or not np.isfinite(f).all() or (f <= 0).any():
        raise ValueError('Invalid baseline input')
    if (~isolated).sum() < 30 or np.ptp(t[~isolated]) < 12:
        raise ValueError('Insufficient support after singleton removal')
    periods, durations = grids(t)  # original support, BEFORE singleton removal or clipping
    # Preserve original median convention; only the Wotan input support changes.
    detrend_time = t[~isolated]
    flat, trend = flatten(detrend_time, (f/np.median(f))[~isolated], **BIWEIGHT_SETTINGS)
    if (not np.isfinite(flat).all() or not np.isfinite(trend).all()
            or (flat <= 0).any() or (trend <= 0).any()):
        raise ValueError('Detrending failed; no silent removal')
    removed, stats = clip_upper(flat)
    if (~removed).sum() < 30 or np.ptp(detrend_time[~removed]) < 12:
        raise ValueError('Insufficient support after clipping')
    # flat/trend/clipped align with detrend_time, NOT the original arrays['time'].
    # Keep the original-coordinate mask for explicit provenance and injection support checks.
    return dict(flat=flat, trend=trend, clipped=removed, search_time=detrend_time[~removed],
                singleton_removed=isolated, detrend_time=detrend_time,
                detrend_source_indices=np.flatnonzero(~isolated),
                search_flux=flat[~removed], periods=periods, durations=durations), stats
