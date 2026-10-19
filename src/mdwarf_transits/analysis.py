"""Complete-roster adapter to the already frozen paired analysis."""
import numpy as np
from .statistics import analyze_scores


def analyze(frame, expected, thresholds, plan):
    wanted = {(r['ticid'], r['example_id'], int(r['label'])) for r in expected}
    actual = {(str(r.ticid), r.example_id, int(r.label)) for r in frame.itertuples()}
    if len(frame) != len(expected) or not frame.example_id.is_unique or actual != wanted:
        raise ValueError('Paired analysis requires every frozen eligible example exactly once')
    result = analyze_scores(frame, thresholds, plan)
    metadata = {r['example_id']: r for r in expected}
    positive = frame[frame.label.eq(1)].copy()
    for field in ('period_index', 'radius_index'):
        positive[field] = [int(metadata[e][field]) for e in positive.example_id]
    result['cells_descriptive'] = {}
    for detector, key in [('cnn', 'cnn_logit'), ('bls', 'bls_score')]:
        positive['flagged'] = positive[key].gt(thresholds['detectors'][detector]['threshold'])
        cells = positive.groupby(['period_index', 'radius_index', 'ticid']).flagged.mean()
        result['cells_descriptive'][detector] = [
            dict(period_index=int(p), radius_index=int(r), recovery=float(g.mean()), stars=len(g))
            for (p, r), g in cells.groupby(level=[0, 1])]
    result['interpretation'] += (' Recovery means light-curve flagging, not period recovery or confirmed planets. '
                                 'Uninjected controls are not certified planet-free stars.')
    return result
