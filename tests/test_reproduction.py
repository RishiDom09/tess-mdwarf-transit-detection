"""Focused public checks for the fixed experiment, without downloading or scoring stars."""
import numpy as np
import pandas as pd
import pytest
from mdwarf_transits import dataset
from mdwarf_transits.binning import make_bin_map
from mdwarf_transits.ordered_preprocessing import prepare_curve
from mdwarf_transits.calibration import rank_threshold
from mdwarf_transits.selection import select
from mdwarf_transits.baseline import singleton_mask
from mdwarf_transits.analysis import analyze
from mdwarf_transits.io import write, receipt, completed
from mdwarf_transits.settings import CONFIG


def test_fixed_identity_roster_and_all_attempts():
    rows=dataset.stars()
    assert len(rows)==350
    assert len({r['gaia_dr3_source_id'] for r in rows})==350
    for role,n in [('train',50195),('validation',10040),('calibration',60),('test',12546)]:
        examples=dataset.examples(role)
        assert len(examples)==n
        assert {r['ticid'] for r in examples}=={s['ticid'] for s in rows if s['role']==role}
        assert sum(int(r['label'])==0 for r in examples)==sum(s['role']==role for s in rows)


def test_binning_preserves_gaps_and_rejects_partial_bins():
    cadence=np.r_[np.arange(30),np.arange(45,59),np.arange(60,75)]
    mapping=make_bin_map(cadence,100+cadence/720,1/720)
    assert mapping.bin_ids.tolist()==[0,1,4]
    assert mapping.rejected_nonempty_bins==1
    assert mapping.empty_bins==1
    np.testing.assert_array_equal(mapping.average(cadence),[7,22,67])
    with pytest.raises(ValueError):make_bin_map(cadence[::-1],(100+cadence/720)[::-1],1/720)


def test_wrap_then_linear_normalization_and_no_input_mutation():
    flux=np.linspace(.9,1.1,700);before=flux.copy()
    p=prepare_curve(flux)
    assert p.values.shape==(1000,1) and p.values.dtype==np.float32
    np.testing.assert_array_equal(p.source_indices,np.r_[np.arange(700),np.arange(300)])
    np.testing.assert_array_equal(flux,before)
    q=np.percentile(flux[p.source_indices],[10,90],method='linear')
    assert [p.q10,p.q90]==q.tolist()
    with pytest.raises(ValueError):prepare_curve(np.ones(1000))


def test_calibration_requires_enough_independent_controls():
    k,bound=rank_threshold(60,.05,.05)
    assert k==60 and bound==pytest.approx(.95**60)
    with pytest.raises(ValueError):rank_threshold(58,.05,.05)


def test_checkpoint_selection_excludes_initialization_and_resolves_ties():
    candidates=[dict(seed_index=s,update=u,auroc=(1 if u==0 else .8)) for s in [1,2,3] for u in [0,1,8]]
    chosen=select(candidates,[0,1,8],[1,2,3])
    assert (chosen['seed_index'],chosen['update'])==(1,1)
    with pytest.raises(ValueError):select(candidates[:-1],[0,1,8],[1,2,3])


def test_singletons_use_strict_gap_boundary():
    np.testing.assert_array_equal(singleton_mask([0,.25,.5,1,1.5,1.6]),[False,False,False,True,False,False])


def test_paired_analysis_weights_stars_and_handles_threshold_ties():
    rows=[]
    for tic,n in [('1',1),('2',9)]:
        rows.append(dict(ticid=tic,example_id=tic+'c',label=0,cnn_logit=1.,bls_score=1.))
        for i in range(n):rows.append(dict(ticid=tic,example_id=tic+'p'+str(i),label=1,cnn_logit=(2. if tic=='1' else 0.),bls_score=(0. if tic=='1' else 2.)))
    expected=[dict(r,period_index=0,radius_index=0) for r in rows]
    thresholds=dict(decision='score > threshold',violation_bound=.95**60,sensitivity={},detectors={'cnn':{'threshold':1.},'bls':{'threshold':1.}})
    plan=dict(CONFIG['test_plan'],active_stars=2,bootstrap=dict(CONFIG['test_plan']['bootstrap'],replicates=100))
    result=analyze(pd.DataFrame(rows),expected,thresholds,plan)
    primary=result['operating_points']['primary']
    assert primary['cnn']['recovery']==primary['bls']['recovery']==.5
    assert primary['cnn']['control_flagged']==primary['bls']['control_flagged']==0
    assert result['cells_descriptive']['cnn'][0]['recovery']==.5
    with pytest.raises(ValueError):analyze(pd.DataFrame(rows[:-1]),expected,thresholds,plan)


def test_receipts_detect_wrong_run_and_modified_artifacts(tmp_path):
    write(tmp_path/'scores.json',{'score':1.})
    receipt(tmp_path,['scores.json'],'run-a')
    assert completed(tmp_path,'run-a')
    with pytest.raises(ValueError):completed(tmp_path,'run-b')
    write(tmp_path/'scores.json',{'score':2.})
    with pytest.raises(ValueError):completed(tmp_path,'run-a')
