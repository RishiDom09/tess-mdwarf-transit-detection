"""Source-ordered, one-channel preprocessing with an explicit random stream.

No observation files are read here. Binning the project's two-minute data is a
later step and is deliberately not implemented in this step-1 package.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PreparedCurve:
    values: np.ndarray
    source_indices: np.ndarray
    removed_indices: np.ndarray
    roll: int
    q10: float
    q90: float


def normalization_percentiles(selected):
    """Linear q10/q90 with the preparation runtime's explicit operation order.

    Scope: finite, one-dimensional float64 arrays after length adjustment.
    NumPy 1.26.4 function_base._quantile/_lerp is the numerical reference.
    This does not use a tolerance or change stored tensors.
    """
    x = np.asarray(selected, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Expected at least two finite values for normalization")
    probabilities = np.true_divide(np.array([10.0, 90.0], dtype=np.float64), 100)
    positions = (x.size - 1) * probabilities
    below = np.floor(positions).astype(np.intp)
    above = below + 1
    gamma = positions - below
    partitioned = np.partition(x, np.unique(np.concatenate(([0, -1], below, above))))
    a, b = partitioned[below], partitioned[above]
    difference = np.subtract(b, a)
    result = np.add(a, difference * gamma)
    np.subtract(b, difference * (1 - gamma), out=result, where=gamma >= 0.5)
    return result


def prepare_curve(flux, *, training=False, rng=None, allow_degenerate=False):
    """Replicate deletion -> roll -> length 1000 -> q10/q90 scaling.

    Explicit adaptations: float64 arithmetic then float32 network input, copy
    rather than mutate caller data, provenance indices, invalid-input checks.
    RandomState reproduces the source's legacy np.random API, with no global RNG
    mutation. A flat curve returns zeros only with allow_degenerate=True, for
    source parity; scientific callers must retain the default rejection.
    """
    x = np.asarray(flux, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite one-dimensional flux array")
    indices = np.arange(x.size)
    removed = np.empty(0, dtype=np.int64)
    roll = 0
    if training:
        if not isinstance(rng, np.random.RandomState):
            raise TypeError("Training requires an explicit numpy.random.RandomState")
        maximum = int(x.size * 0.01)
        count = rng.randint(maximum) if maximum != 0 else 0
        # Source explicitly samples WITHOUT replacement, even when count == 0.
        removed = rng.choice(range(x.size), count, replace=False)
        indices = np.delete(indices, removed)
        roll = int(rng.randint(indices.size))
        indices = np.roll(indices, roll)
    if indices.size < 1000:
        indices = np.pad(indices, (0, 1000 - indices.size), mode="wrap")
    else:
        indices = indices[:1000]
    selected = x[indices]
    q10, q90 = normalization_percentiles(selected)
    if q90 == q10:
        if not allow_degenerate:
            raise ValueError("Zero q90-q10 range: source zeros would hide a degenerate input")
        scaled = np.zeros_like(selected)
    else:
        scaled = ((selected - q10) / ((q90 - q10) / 2)) - 1
    values = scaled.astype(np.float32)[:, None]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite result after percentile scaling or float32 conversion")
    return PreparedCurve(values, indices, removed, roll, float(q10), float(q90))
