"""Pure score-table analysis for the preregistered Step 6 companions.

No file access, stellar-data loader, model selection or scoring. The future Step 6
runner must verify its full frozen roster/eligibility/receipts before calling this.
AUROC matches cnn_bls.analysis.roc: global equal-star weights within each label.
"""
import hashlib
import numpy as np
from scipy.stats import binomtest


def auc_kernel(groups, key):
    """K[i,j]: mean positive-star-i credit versus control-star-j, ties half.

    For a bootstrap multiplicity vector m, global balanced AUROC is
    m @ K @ m / sum(m)**2. This is the same statistic as expanding repeated
    stars under fresh cluster IDs, with no need to repeatedly sort all rows.
    """
    controls = np.array([g.loc[g.label.eq(0), key].iloc[0] for g in groups])
    kernel = []
    for g in groups:
        positives = g.loc[g.label.eq(1), key].to_numpy(float)
        diff = positives[:, None] - controls[None, :]
        kernel.append(np.mean((diff > 0) + .5*(diff == 0), axis=0))
    return np.array(kernel)


def analyze_scores(frame, thresholds, plan):
    """Shared bootstrap draws for AUROC, primary and sensitivity recovery."""
    keys = {'cnn': 'cnn_logit', 'bls': 'bls_score'}
    required = {'ticid', 'example_id', 'label', *keys.values()}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError('Missing score-table fields')
    frame = frame.copy()
    frame['ticid'] = frame.ticid.astype(str)
    if (not frame.ticid.str.fullmatch(r'[0-9]+').all() or not frame.example_id.is_unique
            or set(frame.label) != {0, 1} or not np.isfinite(frame[list(keys.values())]).all().all()):
        raise ValueError('Invalid scores, labels or identities')
    groups = [frame.loc[frame.ticid.eq(tic)].sort_values('example_id')
              for tic in sorted(frame.ticid.unique(), key=int)]
    n = len(groups)
    if n != plan['active_stars'] or any((g.label == 0).sum() != 1 or (g.label == 1).sum() < 1 for g in groups):
        raise ValueError('Expected one control and eligible positives for every star')
    if thresholds['decision'] != 'score > threshold':
        raise ValueError('Unexpected decision rule')
    ranks = []  # Paper's primary operating point only.
    if set(thresholds['sensitivity']) != set(map(str, ranks)):
        raise ValueError('Missing prespecified sensitivity cutoffs')
    b = plan['bootstrap']
    repetitions = b['replicates']
    if not isinstance(repetitions, int) or repetitions < 1 or b['percentile_method'] != 'linear':
        raise ValueError('Invalid bootstrap settings')
    draws = np.random.default_rng(b['seed']).integers(n, size=(repetitions, n))
    counts = np.array([np.bincount(d, minlength=n) for d in draws], dtype=float)

    def interval(values):
        return np.percentile(values, b['interval'], method='linear').tolist()

    result = {'bootstrap': dict(b, star_order=[str(g.ticid.iloc[0]) for g in groups],
                               draws_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()),
              'auroc': {'role': 'secondary'}, 'operating_points': {}}
    areas = {}
    for detector, key in keys.items():
        kernel = auc_kernel(groups, key)
        boot = np.sum((counts @ kernel)*counts, axis=1)/n**2
        areas[detector] = boot
        result['auroc'][detector] = {'value': float(kernel.mean()), 'interval': interval(boot)}
    result['auroc']['cnn_minus_bls'] = {
        'value': result['auroc']['cnn']['value']-result['auroc']['bls']['value'],
        'interval': interval(areas['cnn']-areas['bls'])}
    points = [('primary', thresholds)] + [(str(k), thresholds['sensitivity'][str(k)]) for k in ranks]
    for name, threshold_set in points:
        if name != 'primary' and (threshold_set['role'] != 'exploratory_only' or threshold_set['primary_claim_allowed']):
            raise ValueError('Sensitivity cutoff cannot become primary')
        out = {'role': 'primary' if name == 'primary' else 'exploratory_only',
               'violation_bound': threshold_set['violation_bound']}
        recoveries = {}
        for detector, key in keys.items():
            cutoff = threshold_set['detectors'][detector]['threshold']
            if not np.isfinite(cutoff):
                raise ValueError('Nonfinite cutoff')
            per_star = np.array([g.loc[g.label.eq(1), key].gt(cutoff).mean() for g in groups])
            recoveries[detector] = counts @ per_star/n
            control_count = sum(bool(g.loc[g.label.eq(0), key].iloc[0] > cutoff) for g in groups)
            ci = binomtest(control_count, n).proportion_ci(confidence_level=.95, method='exact')
            positives = frame.loc[frame.label.eq(1), key]
            out[detector] = {'threshold': float(cutoff), 'recovery': float(per_star.mean()),
                             'recovery_interval': interval(recoveries[detector]),
                             'injected_flagged': int(positives.gt(cutoff).sum()), 'injected_total': len(positives),
                             'control_flagged': control_count, 'control_total': n,
                             'control_trigger_rate': control_count/n, 'control_interval': [ci.low, ci.high]}
        out['cnn_minus_bls_recovery'] = {'value': out['cnn']['recovery']-out['bls']['recovery'],
                                       'interval': interval(recoveries['cnn']-recoveries['bls'])}
        result['operating_points'][name] = out
    result['interpretation'] = 'AUROC is secondary; intervals condition on frozen models and thresholds and are not simultaneous.'
    return result
