"""Render the fixed trial manifests without resampling their parameters."""
import json
import hashlib
import numpy as np
from .binning import make_bin_map, audit_injection, native_support
from .ordered_preprocessing import prepare_curve

def array_digest(values):
    """Canonical little-endian float64 bytes, shape and dtype included."""
    values = np.ascontiguousarray(values, dtype='<f8')
    header = json.dumps({'shape': list(values.shape), 'dtype': '<f8'}, sort_keys=True).encode()
    return hashlib.sha256(header+b'\n'+values.tobytes()).hexdigest()

def render_star(time_btjd, cadence, background, star, examples, renderer):
    """Pure array adapter, also used with generated signals in tests."""
    t, background = np.asarray(time_btjd), np.asarray(background)
    if background.shape != t.shape or not np.isfinite(background).all() or (background <= 0).any():
        raise ValueError('Invalid source flux')
    mapping = make_bin_map(cadence, t, float(star['cadence_days']))
    control = np.asarray(background, np.float32).astype(np.float64)
    values, provenance, shared_flux, shared_support = [], [], [], []
    for row in examples:
        if row['ticid'] != star['ticid'] or row['background_path'] != star['background_path']:
            raise PermissionError('Wrong star/example pairing')
        if int(row['label']) == 0:
            flux = control
            prepared = prepare_curve(mapping.average(control))
            support = np.zeros(len(t), dtype=bool)
            audit = {'example_id': row['example_id'], 'label': 0, 'q10': prepared.q10, 'q90': prepared.q90}
        elif int(row['label']) == 1 and row['eligible'] == 'True':
            model = renderer.integrated_model(t, row, int(row['exposure_samples']))
            flux = np.asarray(renderer.inject(background, model, float(row['crowdsap'])), np.float32).astype(np.float64)
            prepared, audit = audit_injection(control, flux, t, mapping, row)
            if audit['lost_all_support']:
                raise ValueError('Injection lost all represented support: '+row['example_id'])
            audit['label'] = 1
            support, _ = native_support(t, control-flux, row)
        else:
            raise PermissionError('Only controls and stored eligible positives may be rendered')
        values.append(prepared.values)
        kept = np.unique(prepared.source_indices)
        binned = mapping.average(flux)[kept]
        shared_flux.append(binned)
        shared_support.append(support[mapping.indices[kept]].any(axis=1))
        audit.update(shared_flux_sha256=array_digest(binned),
                     complete_bins=len(mapping.bin_ids), represented_unique_bins=len(kept),
                     truncated_bins=max(0, len(mapping.bin_ids)-1000),
                     repeated_positions=max(0, 1000-len(mapping.bin_ids)))
        provenance.append(audit)
    return {'x': np.stack(values), 'example_ids': np.array([r['example_id'] for r in examples]),
            'cnn_source_indices': prepared.source_indices, 'bin_native_indices': mapping.indices,
            'bin_ids': mapping.bin_ids, 'bin_time_btjd': mapping.mean_time,
            'shared_flux': np.stack(shared_flux), 'shared_injected_support': np.stack(shared_support),
            'shared_time_btjd': mapping.mean_time[kept], 'shared_bin_ids': mapping.bin_ids[kept],
            'shared_source_indices': kept}, provenance
