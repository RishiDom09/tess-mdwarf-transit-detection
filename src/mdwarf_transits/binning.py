"""Complete 15-cadence means and injected-support accounting."""
from dataclasses import dataclass
import numpy as np
from .preprocessing import prepare_curve

@dataclass(frozen=True)
class BinMap:
    indices: np.ndarray
    bin_ids: np.ndarray
    mean_time: np.ndarray
    anchor: int
    original_samples: int
    rejected_nonempty_bins: int
    empty_bins: int

    def average(self, flux):
        values = np.asarray(flux, dtype=np.float64)
        if values.shape[-1] != self.original_samples or not np.isfinite(values).all():
            raise ValueError("Nonfinite or misaligned flux")
        return np.mean(values[..., self.indices], axis=-1, dtype=np.float64)

def make_bin_map(cadence, time, cadence_days):
    c, t = np.asarray(cadence), np.asarray(time, dtype=np.float64)
    if c.ndim != 1 or len(c) < 15 or c.shape != t.shape:
        raise ValueError("Invalid cadence/time dimensions")
    if not np.isfinite(c).all() or not np.isfinite(t).all() or not np.equal(c, np.floor(c)).all():
        raise ValueError("Invalid cadence/time values")
    c = c.astype(np.int64)
    if not np.isclose(cadence_days * 86400, 120, atol=.1, rtol=0):
        raise ValueError("Expected two-minute source cadence")
    if (np.diff(c) <= 0).any() or (np.diff(t) <= 0).any():
        raise ValueError("Duplicate or nonmonotone samples")
    if np.max(np.abs(np.diff(t) / (np.diff(c) * cadence_days) - 1)) > .01:
        raise ValueError("Cadence/time disagreement")
    ids = (c - c[0]) // 15
    bins, first, counts = np.unique(ids, return_index=True, return_counts=True)
    complete = counts == 15
    if not complete.any():
        raise ValueError("No complete 15-cadence bins")
    indices = first[complete, None] + np.arange(15)[None, :]
    if not (np.diff(c[indices], axis=1) == 1).all():
        raise ValueError("Accepted bin is not consecutive")
    return BinMap(indices, bins[complete], np.mean(t[indices], axis=1), int(c[0]), len(c),
                  int((~complete).sum()), int(ids[-1] + 1 - len(bins)))

def native_support(time, deficit, row):
    time, deficit = np.asarray(time), np.asarray(deficit)
    period, epoch = float(row['period_days']), float(row['t0_btjd'])
    event = np.floor((time - epoch) / period + .5).astype(np.int64)
    distance = np.abs(time - (epoch + event * period))
    possible = distance <= (float(row['duration_days']) + float(row['cadence_days'])) / 2 + 1e-10
    return possible & (deficit > 0), event

def audit_injection(control, positive, time, mapping, row, *, training=False, rng=None):
    deficit = np.asarray(control, np.float64) - np.asarray(positive, np.float64)
    support, events = native_support(time, deficit, row)
    if not support.any():
        raise ValueError("Original positive has no stored injected support")
    prepared = prepare_curve(mapping.average(positive), training=training, rng=rng)
    selected_bins = prepared.source_indices
    selected_original = np.unique(mapping.indices[selected_bins].ravel())
    surviving = selected_original[support[selected_original]]
    original_depth = float(np.max(deficit[support]))
    carried_deficit = mapping.average(deficit)[selected_bins]
    final_depth = float(np.max(carried_deficit))
    scale = (prepared.q90 - prepared.q10) / 2
    return prepared, {"example_id": str(row['example_id']), "ticid": int(row['ticid']),
        "period_index": int(row['period_index']), "radius_index": int(row['radius_index']),
        "phase_index": int(row['phase_index']), "training_transform": bool(training),
        "original_transit_samples": int(support.sum()),
        "surviving_unique_transit_samples": int(len(surviving)),
        "original_events": int(len(np.unique(events[support]))),
        "surviving_unique_events": int(len(np.unique(events[surviving]))),
        "original_depth": original_depth, "surviving_depth": final_depth,
        "depth_ratio": final_depth / original_depth,
        "q10": prepared.q10, "q90": prepared.q90, "half_percentile_range": scale,
        "scaled_transit_depth": final_depth / scale,
        "unique_bins": int(len(np.unique(selected_bins))),
        "lost_all_support": len(surviving) == 0 or final_depth <= 0}
