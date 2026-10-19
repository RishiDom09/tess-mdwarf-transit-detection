"""Fixed sample, verified MAST products, and the original input arithmetic."""
import csv
from collections import Counter
from pathlib import Path
import urllib.parse
import urllib.request
import numpy as np
from .settings import REPOSITORY, CONFIG
from .io import digest, read, write, freeze, receipt, completed

def stars():
    with (REPOSITORY/'manifests/stars.csv').open() as f:
        rows = list(csv.DictReader(f))
    expected = {(role, activity): n for role, groups in CONFIG['counts'].items() for activity, n in groups.items()}
    if Counter((r['role'], r['activity_label']) for r in rows) != expected:
        raise ValueError('Unexpected split membership')
    for field in ('ticid', 'TWOMASS', 'gaia_dr3_source_id', 'source_group'):
        values = [r[field] for r in rows]
        if len(set(values)) != len(rows) or any(not v or v.lower() == 'nan' for v in values):
            raise ValueError('Missing or repeated identity: '+field)
    for r in rows:
        if r['activity_source'] != 'Newton2017' or r['file_qc_pass'] != 'True':
            raise ValueError('Invalid activity source or file QC')
        expected_label = 'active' if float(r['EWHa-u']) < -1 else 'inactive'
        if r['activity_label'] != expected_label or int(r['l_EWHa-u']) not in (0, 1) or (expected_label == 'active' and int(r['l_EWHa-u']) != 0):
            raise ValueError('Activity label or limit flag mismatch')
        if any(r[k] != 'False' for k in ('known_host_excluded','known_eb_excluded','prior_identity_link','prior_spatial_link','within_roster_spatial_link')):
            raise ValueError('Failed frozen catalog screening')
    return rows

def examples(role):
    """Keep the historical CSV parsing conventions at their numerical boundaries."""
    selected = [s for s in stars() if s['role'] == role]
    if role == 'calibration':
        trials = []
    elif role == 'validation':
        import pandas as pd
        frame = pd.read_csv(REPOSITORY/('manifests/trials_'+role+'.csv'), dtype={'cell_seed':str})
        trials = frame.to_dict('records')
        for r in trials:
            r['ticid'] = str(r['ticid'])
            r['eligible'] = str(bool(r['eligible']))
    elif role in ('train', 'test'):
        with (REPOSITORY/('manifests/trials_'+role+'.csv')).open() as f:
            trials = list(csv.DictReader(f))
    else:
        raise ValueError('Unknown role')
    result = []
    for s in selected:
        positive = [r for r in trials if r['ticid'] == s['ticid']]
        if role != 'calibration':
            cells = {(int(r['period_index']), int(r['radius_index']), int(r['phase_index'])) for r in positive}
            if len(positive) != 250 or len(cells) != 250:
                raise ValueError('Missing or duplicated injection trials')
            if any(r['split'] != s['old_role'] or r['source_sha256'] != s['source_sha256'] or int(r['exposure_samples']) != 63 for r in positive):
                raise ValueError('Trial lineage mismatch')
        positive = [r for r in positive if r['eligible'] == 'True']
        if role == 'train':
            positive.sort(key=lambda r:(int(r['period_index']), int(r['radius_index']), int(r['phase_index']), r['example_id']))
        else:
            positive.sort(key=lambda r:r['example_id'])
        control = dict(ticid=s['ticid'], example_id=s['control_example_id'], label=0,
                       activity_label=s['activity_label'], background_path=s['background_path'])
        result.extend([control]+positive)
    expected = {'train':50195, 'validation':10040, 'calibration':60, 'test':12546}[role]
    if len(result) != expected or len({r['example_id'] for r in result}) != expected:
        raise ValueError('Eligible example count/identity mismatch')
    return result

def download(data, role=None):
    """Download the exact recorded product rather than querying for a newer sector."""
    folder = Path(data)/'raw'
    folder.mkdir(parents=True, exist_ok=True)
    selected = [s for s in stars() if role is None or s['role'] == role]
    for i, star in enumerate(selected, 1):
        path = folder/star['filename']
        if not path.exists():
            url = 'https://mast.stsci.edu/api/v0.1/Download/file?'+urllib.parse.urlencode({'uri':star['data_uri']})
            temp = path.with_suffix(path.suffix+'.partial')
            with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as f:
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk: break
                    f.write(chunk)
            if digest(temp) != star['source_sha256']:
                raise ValueError('MAST checksum differs from the published product: '+star['filename'])
            temp.replace(path)
        if digest(path) != star['source_sha256']:
            raise ValueError('Input checksum mismatch: '+str(path))
        print('Downloaded/verified', i, '/', len(selected), star['ticid'], flush=True)

def source_arrays(path, star):
    from astropy.io import fits
    if digest(path) != star['source_sha256']:
        raise ValueError('Changed FITS input')
    with fits.open(path, memmap=False) as h:
        if int(h[0].header['TICID']) != int(star['ticid']) or int(h[0].header['SECTOR']) != int(star['sector']):
            raise ValueError('FITS identity mismatch')
        hd, x = h[1].header, h[1].data
        if hd['TIMESYS'] != 'TDB' or hd['BJDREFI'] != 2457000 or abs(hd['TIMEDEL']*86400-120) > .1:
            raise ValueError('FITS time system/cadence mismatch')
        keep = np.isfinite(x['TIME']) & np.isfinite(x['SAP_FLUX']) & ((x['QUALITY'] & CONFIG['data']['quality_mask']) == 0)
        t = np.array(x['TIME'][keep], dtype=np.float64)
        f = np.array(x['SAP_FLUX'][keep], dtype=np.float64)
        e = np.array(x['SAP_FLUX_ERR'][keep], dtype=np.float64)
        cadence = np.array(x['CADENCENO'][keep])
        if (len(t) != int(star['samples']) or not np.isfinite(e).all() or (e <= 0).any()
                or (f <= 0).any() or (np.diff(t) <= 0).any() or len(t)*hd['TIMEDEL'] < 20
                or np.ptp(t) < 20 or not .9 <= hd['CROWDSAP'] <= 1):
            raise ValueError('FITS quality/support mismatch')
        scale = float(np.median(f))
        return t, cadence, f/scale

def prepare(data, run, role):
    from . import injection
    from .binning import make_bin_map, native_support
    from .preprocessing import prepare_curve
    from .render import render_star
    run, data = Path(run), Path(data)
    identity = freeze(run)
    if role == 'calibration' and not completed(run/'selection', identity):
        raise ValueError('Select the model before calibration preparation')
    if role == 'test' and not completed(run/'thresholds', identity):
        raise ValueError('Freeze thresholds before test preparation')
    if role == 'validation' and not all(completed(run/('seed_%d'%s), identity) for s in (1,2,3)):
        raise ValueError('Complete all three training runs before validation')
    selected = [s for s in stars() if s['role'] == role]
    all_rows = examples(role)
    for index, star in enumerate(selected, 1):
        folder = run/'inputs'/role/star['ticid']
        if completed(folder, identity):
            continue
        rows = [r for r in all_rows if r['ticid'] == star['ticid']]
        t, cadence, background = source_arrays(data/'raw'/star['filename'], star)
        folder.mkdir(parents=True, exist_ok=True)
        if role == 'train':
            mapping = make_bin_map(cadence, t, float(star['cadence_days']))
            control = np.asarray(background, np.float32).astype(np.float64)
            fluxes, supports = [], []
            for row in rows:
                if int(row['label']):
                    model = injection.integrated_model(t, row, int(row['exposure_samples']))
                    flux = np.asarray(injection.inject(background, model, float(row['crowdsap'])), np.float32).astype(np.float64)
                    support, _ = native_support(t, control-flux, row)
                else:
                    flux, support = control, np.zeros(len(t), bool)
                binned = mapping.average(flux)
                carried = support[mapping.indices].any(axis=1)
                p = prepare_curve(binned)
                if int(row['label']) and not carried[p.source_indices].any():
                    raise ValueError('Training injection lost all support')
                fluxes.append(binned); supports.append(carried)
            np.savez_compressed(str(folder/'arrays.npz'), flux=np.stack(fluxes), support=np.stack(supports), example_ids=np.array([r['example_id'] for r in rows]))
        else:
            arrays, audits = render_star(t, cadence, background, star, rows, injection)
            np.savez_compressed(str(folder/'arrays.npz'), **arrays)
            write(folder/'audit.json', audits)
        write(folder/'manifest.json', rows)
        names = ['arrays.npz','manifest.json']+(['audit.json'] if role != 'train' else [])
        receipt(folder, names, identity)
        print('Prepared', role, index, '/', len(selected), star['ticid'], flush=True)
    if role == 'train':
        # Repeat the original training-only deterministic support preflight.
        cache = TrainingCache(run)
        rng = np.random.RandomState(CONFIG['training']['support_seed'])
        for tic in sorted(cache.arrays, key=int):
            cells = {}
            for i in cache.positive[tic]:
                r = cache.rows[i]
                cells.setdefault((int(r['period_index']),int(r['radius_index'])),[]).append(i)
            keys = sorted(cells)
            for j in range(CONFIG['training']['training_support_transforms_per_star']):
                options = cells[keys[j % len(keys)]]
                cache.example(options[(j//len(keys)) % len(options)], training=True, rng=rng)
    receipt(run/'inputs'/role, [s['ticid']+'/completion.json' for s in selected], identity)

class TrainingCache:
    def __init__(self, run):
        self.rows = []
        self.arrays = {}; self.positive = {}; self.control = {}
        self.groups = {'active':[], 'inactive':[]}
        identity = freeze(run)
        for star in (s for s in stars() if s['role'] == 'train'):
            tic = star['ticid']; folder = Path(run)/'inputs/train'/tic
            if not completed(folder, identity): raise ValueError('Missing training inputs')
            with np.load(str(folder/'arrays.npz'), allow_pickle=False) as a:
                self.arrays[tic] = {'flux':a['flux'], 'support':a['support']}
            rows = read(folder/'manifest.json')
            self.positive[tic] = []
            for local, row in enumerate(rows):
                row = dict(row, label=int(row['label']), cache_index=local)
                i = len(self.rows); self.rows.append(row)
                if row['label']: self.positive[tic].append(i)
                else: self.control[tic] = i
            self.groups[star['activity_label']].append(tic)
    def example(self, index, training=False, rng=None):
        from .preprocessing import prepare_curve
        row = self.rows[int(index)]; a = self.arrays[row['ticid']]; i = row['cache_index']
        p = prepare_curve(a['flux'][i], training=training, rng=rng)
        if row['label'] and not a['support'][i,p.source_indices].any():
            raise ValueError('Augmentation lost all transit support: '+row['example_id'])
        return p.values, row['label'], row
