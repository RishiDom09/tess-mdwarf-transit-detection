"""Explicit reproduction stages; importing this module never starts a run."""
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
from .settings import CONFIG
from .io import read, write, digest, freeze, receipt, completed

def require_baseline():
    from importlib.metadata import version
    expected = {'numpy':'1.26.4','pandas':'2.3.3','astropy':'7.2.2','scipy':'1.14.1',
                'batman-package':'2.5.2','wotan':'1.10','numba':'0.61.2','llvmlite':'0.44.0'}
    for name, wanted in expected.items():
        if version(name) != wanted:
            raise ValueError('Use the pinned baseline environment: '+name+'=='+wanted)

def train(run, seed):
    from .dataset import TrainingCache
    from .sampling import Sampler
    from .learning import configure, updater, save_checkpoint
    from .model import build_model, build_optimizer
    identity = freeze(run); r = CONFIG['training']; configure(r)
    if not completed(run/'inputs/train', identity): raise ValueError('Prepare all training stars first')
    out = run/('seed_%d'%seed)
    if completed(out, identity): return
    if out.exists(): raise ValueError('Interrupted training cannot resume its dropout stream; move the failed seed directory and restart this seed')
    out.mkdir(parents=True)
    start = time.monotonic()
    cache = TrainingCache(run); seeds = r['seeds'][seed-1]
    sampler = Sampler(cache, seeds[1], seeds[2], r['batch_size'], r['window_shift'])
    model = build_model(seeds[0]); opt = build_optimizer(); update = updater(model,opt)
    save_checkpoint(out/'step_000000', model, opt, sampler, 0)
    for step in range(1, r['updates']+1):
        x,y = sampler.batch(); loss = float(update(x,y).numpy())
        with (out/'training.jsonl').open('a') as log:
            import json
            log.write(json.dumps({'step':step, 'loss':loss, 'draws':sampler.draws})+'\n')
        if step in r['checkpoints']:
            save_checkpoint(out/('step_%06d'%step),model,opt,sampler,step)
            print('Seed',seed,'update',step,'loss',loss,flush=True)
    names = [str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()]
    receipt(out,names,identity,{'seconds':time.monotonic()-start,'updates':r['updates']})

def inputs(run, role, identity):
    import numpy as np
    from .dataset import stars
    base = run/'inputs'/role
    if not completed(base, identity): raise ValueError('Incomplete input role: '+role)
    arrays, rows = [], []
    for star in (s for s in stars() if s['role'] == role):
        folder = base/star['ticid']
        if not completed(folder,identity): raise ValueError('Missing prepared star')
        manifest = read(folder/'manifest.json')
        with np.load(str(folder/'arrays.npz'),allow_pickle=False) as a:
            if a['example_ids'].tolist() != [r['example_id'] for r in manifest]:
                raise ValueError('Prepared example order mismatch')
            arrays.append(a['x'])
        rows.extend(dict(r,label=int(r['label'])) for r in manifest)
    return np.concatenate(arrays),rows

def select(run):
    import numpy as np
    from .learning import configure,predict
    from .model import build_model
    from .selection import metrics,select as choose
    identity=freeze(run);r=CONFIG['training'];configure(r)
    out=run/'selection'
    if completed(out,identity):return
    for seed in (1,2,3):
        if not completed(run/('seed_%d'%seed),identity):raise ValueError('Complete every training seed first')
    x,rows=inputs(run,'validation',identity)
    indices=[i for i,row in enumerate(rows) if row['activity_label']=='active']
    candidates=[];model=build_model(r['inference_initialization_seed'])
    out.mkdir(parents=True,exist_ok=True)
    for seed in (1,2,3):
        for step in r['checkpoints']:
            weight=run/('seed_%d'%seed)/('step_%06d'%step)/'ordinary.weights.h5'
            model.load_weights(str(weight))
            # Original inference order includes both activity groups before selection.
            z=predict(model,x,r['inference_batch_size'])
            active=metrics([rows[i] for i in indices],z[indices])
            candidates.append(dict(seed_index=seed,update=step,auroc=active['auroc']))
            print('Validation',seed,step,flush=True)
    selected=choose(candidates,r['checkpoints'],range(1,4),r['selection_tie_tolerance'])
    source=run/('seed_%d'%selected['seed_index'])/('step_%06d'%selected['update'])/'ordinary.weights.h5'
    shutil.copyfile(str(source),str(out/'weights.h5'))
    write(out/'selected.json',dict(selected,weights_sha256=digest(out/'weights.h5')))
    write(out/'candidates.json',candidates)
    receipt(out,['weights.h5','selected.json','candidates.json'],identity)

def score_cnn(run,role):
    from .learning import configure,predict,array_hash
    from .model import build_model
    identity=freeze(run);r=CONFIG['training'];configure(r)
    if not completed(run/'selection',identity):raise ValueError('Select the model first')
    if role=='test' and not completed(run/'thresholds',identity):raise ValueError('Freeze calibration before testing')
    out=run/'scores'/role/'cnn'
    if completed(out,identity):return
    x,rows=inputs(run,role,identity)
    model=build_model(r['inference_initialization_seed'])
    model.load_weights(str(run/'selection/weights.h5'))
    before=array_hash(model.get_weights());z=predict(model,x,r['inference_batch_size'])
    if before!=array_hash(model.get_weights()):raise ValueError('Inference changed weights')
    write(out/'scores.json',[dict(ticid=r['ticid'],example_id=r['example_id'],label=int(r['label']),cnn_logit=float(v)) for r,v in zip(rows,z)])
    receipt(out,['scores.json'],identity,{'weights_sha256':digest(run/'selection/weights.h5')})

def bls_job(job):
    """One committed example; failures are never turned into zero scores."""
    import numpy as np
    from .baseline import baseline_arrays
    from .bls import search
    run,role,tic,i,row,identity=job
    folder=run/'scores'/role/'bls'/row['example_id']
    if completed(folder,identity):return
    with np.load(str(run/'inputs'/role/tic/'arrays.npz'),allow_pickle=False) as a:
        if a['example_ids'][i]!=row['example_id']:raise ValueError('BLS example order mismatch')
        shared={'time':a['shared_time_btjd'],'flux':a['shared_flux'][i]}
        support=a['shared_injected_support'][i]
    arrays,stats=baseline_arrays(shared)
    retained=arrays['detrend_source_indices'][~arrays['clipped']]
    if int(row['label']) and not support[retained].any():raise ValueError('BLS preprocessing lost all injected support')
    fit,spectrum=search(arrays['search_time'],arrays['search_flux'],arrays['periods'],arrays['durations'])
    write(folder/'score.json',dict(ticid=tic,example_id=row['example_id'],label=int(row['label']),bls_score=fit['score'],fit=fit,clipping=stats))
    # Full spectra are generated locally, never committed to Git.
    np.savez_compressed(str(folder/'spectrum.npz'),**spectrum)
    receipt(folder,['score.json','spectrum.npz'],identity)

def score_bls(run,role,workers):
    require_baseline()
    from .dataset import stars
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    identity=freeze(run)
    if not completed(run/'selection',identity):raise ValueError('Select the model first')
    if role=='test' and not completed(run/'thresholds',identity):raise ValueError('Freeze calibration before testing')
    if not completed(run/'inputs'/role,identity):raise ValueError('Incomplete input role')
    out=run/'scores'/role/'bls'
    if completed(out,identity):return
    jobs=[]
    for s in (s for s in stars() if s['role']==role):
        folder=run/'inputs'/role/s['ticid']
        if not completed(folder,identity):raise ValueError('Missing star inputs')
        for i,row in enumerate(read(folder/'manifest.json')):
            jobs.append((run,role,s['ticid'],i,row,identity))
    if workers==1:
        for i,job in enumerate(jobs,1):
            bls_job(job)
            if i%25==0:print('BLS',role,i,'/',len(jobs),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for i,_ in enumerate(pool.map(bls_job,jobs),1):
                if i%25==0:print('BLS',role,i,'/',len(jobs),flush=True)
    receipt(out,[job[4]['example_id']+'/completion.json' for job in jobs],identity)

def score_table(run,role,identity):
    import pandas as pd
    from .dataset import examples
    expected=examples(role)
    cnn=run/'scores'/role/'cnn';bls=run/'scores'/role/'bls'
    if not completed(cnn,identity) or not completed(bls,identity):raise ValueError('Both detectors must finish before analysis')
    rows=read(cnn/'scores.json')
    wanted={(r['ticid'],r['example_id'],int(r['label'])) for r in expected}
    if len(rows)!=len(wanted) or {(r['ticid'],r['example_id'],r['label']) for r in rows}!=wanted:
        raise ValueError('Incomplete or changed CNN roster')
    for row in rows:
        folder=bls/row['example_id']
        if not completed(folder,identity):raise ValueError('Missing BLS example')
        b=read(folder/'score.json')
        if any(b[k]!=row[k] for k in ('ticid','example_id','label')):raise ValueError('CNN/BLS pairing mismatch')
        row['bls_score']=b['bls_score']
    return pd.DataFrame(rows),expected

def calibrate(run):
    require_baseline()
    import numpy as np
    from .calibration import rank_threshold
    identity=freeze(run);out=run/'thresholds'
    if completed(out,identity):return
    frame,_=score_table(run,'calibration',identity)
    c=CONFIG['calibration'];n=len(frame)
    if n!=c['n'] or frame.ticid.nunique()!=n or not frame.label.eq(0).all():raise ValueError('Wrong calibration controls')
    k,bound=rank_threshold(n,c['alpha'],c['delta'])
    if k!=c['rank'] or c['decision']!='score > threshold':raise ValueError('Changed threshold rule')
    result=dict(c,violation_bound=bound,detectors={},sensitivity={})
    for detector,key in [('cnn','cnn_logit'),('bls','bls_score')]:
        scores=frame[key].to_numpy(float)
        if not np.isfinite(scores).all():raise ValueError('Invalid calibration scores')
        result['detectors'][detector]={'threshold':float(sorted(scores)[k-1]),'rank':k}
    write(out/'thresholds.json',result)
    receipt(out,['thresholds.json'],identity,{'weights_sha256':digest(run/'selection/weights.h5')})

def evaluate(run):
    require_baseline()
    from .analysis import analyze
    identity=freeze(run);out=run/'evaluation'
    if completed(out,identity):return
    if not completed(run/'thresholds',identity):raise ValueError('No frozen thresholds')
    frame,expected=score_table(run,'test',identity)
    summary=analyze(frame,expected,read(run/'thresholds/thresholds.json'),CONFIG['test_plan'])
    write(out/'summary.json',summary)
    frame.to_csv(out/'scores.csv',index=False)
    receipt(out,['summary.json','scores.csv'],identity)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['validate','download','prepare','train','select','cnn','bls','calibrate','evaluate','figure'])
    p.add_argument('--role',choices=['train','validation','calibration','test'])
    p.add_argument('--seed',type=int,choices=[1,2,3])
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--run',type=Path,default=Path('outputs/reproduction'))
    a=p.parse_args();a.run=a.run.resolve();a.data=a.data.resolve()
    if a.workers<1:p.error('--workers must be positive')
    if a.stage=='validate':
        from .dataset import stars,examples
        freeze(a.run)
        print('Stars:',len(stars()))
        for role in ('train','validation','calibration','test'):print(role,len(examples(role)))
    elif a.stage=='download':
        from .dataset import download
        freeze(a.run);download(a.data,a.role)
    elif a.stage=='prepare':
        if not a.role:p.error('prepare requires --role')
        require_baseline()
        from .dataset import prepare
        prepare(a.data,a.run,a.role)
    elif a.stage=='train':
        if not a.seed:p.error('train requires --seed')
        train(a.run,a.seed)
    elif a.stage=='select':select(a.run)
    elif a.stage in ('cnn','bls'):
        if a.role not in ('calibration','test'):p.error('scoring requires --role calibration or test')
        if a.stage=='cnn':score_cnn(a.run,a.role)
        else:score_bls(a.run,a.role,a.workers)
    elif a.stage=='calibrate':calibrate(a.run)
    elif a.stage=='evaluate':evaluate(a.run)
    elif a.stage=='figure':
        identity=freeze(a.run)
        if not completed(a.run/'evaluation',identity):raise ValueError('Complete evaluation first')
        from .figure import make_figure
        make_figure(a.run/'evaluation/summary.json',a.run/'figures')

if __name__=='__main__':main()
