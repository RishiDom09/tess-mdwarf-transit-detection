"""Frozen circular-orbit batman renderer; geometry is supplied by trial manifests."""
import numpy as np
import batman

def params(trial):
 p=batman.TransitParams();p.t0=float(trial['t0_btjd']);p.per=float(trial['period_days']);p.rp=float(trial['rp_rs']);p.a=float(trial['a_rs']);p.inc=float(trial['inclination_deg']);p.ecc=0.;p.w=90.;p.u=[float(trial['ld_u1']),float(trial['ld_u2'])];p.limb_dark='quadratic'
 return p

def integrated_model(time,trial,n_samples):
 """Evaluate only possible transit exposures; midpoint quadrature is explicit."""
 t=np.asarray(time,dtype=np.float64);p=params(trial);dt=float(trial['cadence_days'])
 phase=(t-p.t0+.5*p.per)%p.per-.5*p.per
 near=np.abs(phase)<=float(trial['duration_days'])/2+dt/2+1e-10
 out=np.ones(len(t),dtype=np.float64)
 if np.any(near):
  offsets=((np.arange(n_samples)+.5)/n_samples-.5)*dt
  ts=(t[near,None]+offsets[None,:]).reshape(-1)
  values=batman.TransitModel(p,np.ascontiguousarray(ts)).light_curve(p).reshape(-1,n_samples).mean(axis=1)
  out[near]=values
 return out

def inject(background,model,crowding):
 return np.asarray(background,dtype=np.float64)*(1+crowding*(model-1))
