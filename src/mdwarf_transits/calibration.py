"""Tong et al. (2018), Algorithm 1, Proposition 1, equations 1–2."""
import math

def violation_bound(n, k, alpha):
    if (isinstance(n, bool) or not isinstance(n, int) or n < 1
            or isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= n
            or not 0 < alpha < 1):
        raise ValueError('Invalid order-statistic parameters')
    return math.fsum(math.comb(n, j) * (1-alpha)**j * alpha**(n-j)
                     for j in range(k, n+1))

def rank_threshold(n, alpha, delta):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError('Invalid alpha/delta')
    # Small calibration panel: direct exact-binomial sum, no asymptotic approximation.
    for k in range(1, n + 1):
        tail = violation_bound(n, k, alpha)
        if tail <= delta:
            return k, tail
    raise ValueError('Insufficient independent controls for the prespecified target')
