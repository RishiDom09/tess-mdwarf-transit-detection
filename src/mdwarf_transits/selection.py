"""Frozen numerical definitions, independent of model/runtime and renderer."""
import numpy as np

def weights(rows):
    stars=sorted({r['ticid'] for r in rows});n=len(stars)
    w=np.zeros(len(rows),float)
    for star in stars:
        p=[i for i,r in enumerate(rows) if r['ticid']==star and r['label']==1]
        c=[i for i,r in enumerate(rows) if r['ticid']==star and r['label']==0]
        if len(c)!=1 or not p:raise ValueError('Invalid scoring labels')
        w[p]=1/(2*n*len(p));w[c]=1/(2*n)
    return w

def metrics(rows,logits):
    z=np.asarray(logits,float);y=np.array([r['label'] for r in rows]);w=weights(rows)
    if z.shape!=y.shape or not np.isfinite(z).all():raise ValueError('Invalid scores')
    order=np.argsort(z,kind='mergesort');neg=0.;numerator=0.;j=0
    while j<len(order):
        k=j+1
        while k<len(order) and z[order[k]]==z[order[j]]:k+=1
        ix=order[j:k];pw=w[ix][y[ix]==1].sum();nw=w[ix][y[ix]==0].sum()
        numerator+=pw*(neg+.5*nw);neg+=nw;j=k
    paired=[]
    for star in sorted({r['ticid'] for r in rows}):
        control=next(i for i,r in enumerate(rows) if r['ticid']==star and r['label']==0)
        positives=[i for i,r in enumerate(rows) if r['ticid']==star and r['label']==1]
        delta=z[positives]-z[control];paired.append(np.mean((delta>0)+.5*(delta==0)))
    return {'auroc':float(numerator/(w[y==1].sum()*w[y==0].sum())),
            'bce':float(np.sum(w*(np.maximum(z,0)-y*z+np.log1p(np.exp(-np.abs(z)))))),
            'paired_ranking':float(np.mean(paired))}

def select(candidates,steps,seeds,tolerance=1e-12):
    expected={(seed,step) for seed in seeds for step in steps}
    identities=[(c['seed_index'],c['update']) for c in candidates]
    if set(identities)!=expected or len(identities)!=len(expected):raise ValueError('Incomplete or duplicate candidates')
    if any(not np.isfinite(c['auroc']) or not 0<=c['auroc']<=1 for c in candidates):raise ValueError('Invalid AUROC')
    trained=[c for c in candidates if c['update']>0];best=max(c['auroc'] for c in trained)
    return min((c for c in trained if best-c['auroc']<=tolerance),key=lambda c:(c['update'],c['seed_index']))
