#!/usr/bin/env python3
"""Run from any directory; data and outputs are explicit CLI paths."""
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mdwarf_transits.pipeline import main
if __name__=='__main__': main()
