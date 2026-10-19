"""Upper-only masking of detrended flux; no injection truth or lower clipping.

TLS author tutorial 02, commit 1440ca760a785bf06a56619f705539c3a7377dd7:
sigma_clip(y_filt, sigma_upper=2, sigma_lower=float('inf')).
The robust mad_std scatter replaces the tutorial's default std. This and applying
the rule after biweight rather than its median filter are recorded adaptations.
"""
import numpy as np
from .settings import CLIPPING_SETTINGS


def clip_upper(flat):
    """Return a removal mask and diagnostics; never cap or interpolate flux.

    Every light curve is processed independently. The same rule at two cadences
    can yield different thresholds/masks, and does not identify astrophysical
    flares with certainty. A positive flare overlapping a transit can be masked.
    """
    from astropy.stats import sigma_clip
    f = np.asarray(flat, dtype=np.float64)
    if f.ndim != 1 or len(f) == 0 or not np.isfinite(f).all() or (f <= 0).any():
        raise ValueError('Upper clipping requires finite positive one-dimensional flux')
    c = CLIPPING_SETTINGS
    if c['sigma_lower'] is not None or c['operation'] != 'mask_after_detrending':
        raise ValueError('Only upper-only masking is supported')
    masked, _, upper = sigma_clip(
        f, sigma_upper=c['sigma_upper'], sigma_lower=float('inf'),
        maxiters=c['maxiters'], cenfunc=c['cenfunc'], stdfunc=c['stdfunc'],
        grow=c['grow'], axis=None, masked=True, return_bounds=True, copy=True)
    removed = np.ma.getmaskarray(masked).copy()
    if removed.all() or not np.isfinite(upper):
        raise ValueError('Clipping left no usable flux or a nonfinite threshold (possibly zero scatter)')
    stats = dict(input_points=len(f), clipped_points=int(removed.sum()),
                 clipped_fraction=float(removed.mean()), retained_points=int((~removed).sum()),
                 max_detrended_flux=float(f.max()), max_retained_detrended_flux=float(f[~removed].max()),
                 upper_clip_threshold=float(upper), clipping_settings=dict(c))
    return removed, stats
