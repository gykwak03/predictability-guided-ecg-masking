"""Phase 1 scientific and deployment contracts on CPU; no real performance claims."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.phase1.config import ROOT as CONFIG_ROOT, resolve_config, semantic_config
from experiments.phase1.data import (LEAD_NAMES, NORMALIZATION, ORDER, PREPROCESSING,
    get_data, load_artifact, select_data, synthetic_data)
from experiments.phase1.masks import (batch_indices, geometry, make_bank, mask_digest,
    patch_location, patchify, per_lead_masks, replay_bank, unpatchify)
from experiments.phase1.model import build_mae, build_estimator, ridge_lags
from experiments.phase1.metrics import measure, aggregate
from experiments.phase1.training import (batch_for, load_checkpoint, seed_all, train_model,
                                         learning_rate)
from experiments.phase1.pipeline import bank_at, run_fit, run_pilot, run_score
from experiments.predictability_estimators import normalized_masked_loss, sample_hidden


@pytest.fixture
def cfg(tmp_path):
    torch.set_num_threads(2)
    return resolve_config(overrides=dict(output_root=str(tmp_path / 'out'), cache_dir=str(tmp_path / 'cache')),
                          smoke=True, environ={})


def test_portable_path_precedence_and_config_relative_paths(tmp_path):
    directory = tmp_path / 'settings'
    directory.mkdir()
    path = directory / 'cfg.json'
    path.write_text(json.dumps(dict(raw_dir='raw', artifact_dir='data', output_root='results', cache_dir='cache')))
    cfg = resolve_config(path, environ={}, cwd=tmp_path)
    assert cfg['artifact_dir'] == str(directory / 'data')
    env = dict(ECG_ARTIFACT_DIR='env-data', ECG_RAW_DIR='env-raw')
    cfg = resolve_config(path, dict(artifact_dir='cli-data'), environ=env, cwd=tmp_path)
    assert cfg['artifact_dir'] == str(tmp_path / 'cli-data')
    assert cfg['raw_dir'] == str(tmp_path / 'env-raw')
    assert cfg['output_root'] == str(directory / 'results')
    assert Path(resolve_config(environ={})['output_root']) == ROOT / 'experiments/artifacts/phase1'
    assert CONFIG_ROOT == ROOT


def test_wrong_os_paths_rejected(tmp_path):
    from experiments.phase1.config import absolute_path
    with pytest.raises(ValueError, match='path is not usable'):
        absolute_path('/data/server-only' if os.name == 'nt' else 'C:\\server-only', tmp_path)


@pytest.mark.parametrize('patch,seconds,q,count', [(25,.1,100,50), (50,.2,50,25), (125,.5,20,10)])
def test_patch_order_roundtrip_time_and_exact_per_lead_masks(patch, seconds, q, count):
    x = np.arange(12*2500).reshape(1, 12, 2500)
    tokens = patchify(x, patch)
    assert np.array_equal(unpatchify(tokens), x)
    token = 7*q+3
    assert np.array_equal(tokens[0, token], x[0, 7, 3*patch:4*patch])
    where = patch_location(token, patch)
    assert where['lead_index'] == 7 and where['start_seconds'] == pytest.approx(3*seconds)
    assert geometry(patch, .5) == (q, count)
    mask = per_lead_masks(8, patch, .5, 128)
    assert mask.dtype == bool and np.all(mask.reshape(8,12,q).sum(-1) == count)
    assert not np.array_equal(mask[0,:q], mask[0,q:2*q])
    with pytest.raises(ValueError, match='Exact per-lead'):
        per_lead_masks(1, patch, .501, 1)


def test_bank_replay_complements_rng_independence_and_tampering(tmp_path):
    frame = synthetic_data()['frames']['train']
    first, masks = make_bank(frame, [0, 3], 50, .5, 2, 42, 'score', pairs=True)
    torch.manual_seed(993)
    np.random.seed(288)
    second, other = make_bank(frame, [3, 0], 50, .5, 2, 42, 'score', pairs=True)
    lookup = dict(zip(second.mask_id, other))
    assert np.array_equal(masks, np.stack([lookup[key] for key in first.mask_id]))
    assert (masks[::2] ^ masks[1::2]).all()
    path = tmp_path / 'bank.csv'
    first.to_csv(path, index=False)
    loaded = pd.read_csv(path, keep_default_na=False)
    assert np.array_equal(replay_bank(loaded), masks)
    loaded.loc[0, 'mask_seed'] += 1
    with pytest.raises(ValueError, match='digest/metadata mismatch'):
        replay_bank(loaded)


def test_fit_score_validation_disjoint_and_fit_only_scale(cfg):
    data = get_data(cfg)
    frames, selection = data['frames'], data['selected']
    pools = [frames['train'].iloc[selection['fit']], frames['train'].iloc[selection['score']],
             frames['validation'].iloc[selection['validation']]]
    assert not set(pools[0].patient_id) & set(pools[1].patient_id)
    assert not (set(pools[0].patient_id) | set(pools[1].patient_id)) & set(pools[2].patient_id)
    altered = synthetic_data()
    altered['arrays']['train'][selection['score']] *= 100
    altered['arrays']['validation'] *= 500
    checked = select_data(altered, cfg)
    assert np.array_equal(data['scale'], checked['scale'])
    scale = np.asarray(data['arrays']['train'][selection['fit']], dtype=float).std(axis=(0,2), ddof=0)
    assert np.allclose(data['scale'], scale, atol=1e-12)


@pytest.mark.parametrize('name', ['ridge','cnn','transformer'])
def test_hidden_input_isolation_and_visible_zero_indicator(name, cfg):
    data = get_data(cfg)
    bank, masks = make_bank(data['frames']['train'], data['selected']['fit'], 50, .5, 1, 42, 'fit')
    x = np.asarray(data['arrays']['train'][bank.artifact_index], dtype=np.float32)
    model = build_estimator(name, cfg)
    if name == 'ridge':
        model.fit(data['arrays']['train'], bank, masks, data['scale'], samples_per_ecg=32)
        predict = lambda value: model.predict(value, masks)
    else:
        model.eval()
        def predict(value):
            with torch.no_grad():
                return model(torch.from_numpy(value), torch.from_numpy(masks)).numpy()
    hidden = sample_hidden(masks, x.shape)
    initial = predict(x)
    assert np.array_equal(initial, predict(np.where(hidden, x+10000, x).astype(np.float32)))
    assert not np.array_equal(initial, predict(np.where(hidden, x, x+2).astype(np.float32)))
    from experiments.predictability_estimators import torch_visible
    visible, observed, _ = torch_visible(torch.zeros_like(torch.tensor(x)), torch.tensor(masks))
    assert torch.all(visible == 0) and torch.all(observed[torch.tensor(hidden)] == 0)
    assert torch.all(observed[~torch.tensor(hidden)] == 1)


def test_hidden_only_nrmse_loss_amplitude_maps_and_baseline_flags():
    target = np.tile([2.,-2.,4.,-4.], (1,2,1))
    mask = np.array([[True,False,False,True]])
    hidden = sample_hidden(mask, target.shape)
    pred = target.copy()
    pred[hidden] += np.broadcast_to(np.array([2,4])[None,:,None], target.shape)[hidden]
    pred[~hidden] = 9999
    frame, patch = measure(pred, target, mask, np.array([2,4]))
    assert frame.nrmse.iloc[0] == pytest.approx(1)
    assert np.allclose(patch[np.isfinite(patch)], 1) and np.isnan(patch).sum() == 2
    loss = normalized_masked_loss(torch.tensor(pred), torch.tensor(target), torch.tensor(mask), torch.tensor([2.,4.]))
    assert loss.item() == pytest.approx(1)
    zero, _ = measure(np.zeros_like(target), target, mask, np.array([2,4]))
    assert aggregate(zero)['amplitude_collapse_flag']
    worse, _ = measure(-target, target, mask, np.array([2,4]))
    assert aggregate(worse)['worse_than_zero']


def test_preprocessing_statistics_dependence_is_separate_from_input_isolation(cfg):
    book = json.loads((ROOT / 'preprocessing/PTBXL_preprocessing.ipynb').read_text(encoding='utf-8'))
    source = ''.join(book['cells'][20]['source'])
    # Extract the actual reviewed normalize_ecg body without notebook side effects.
    import ast
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'normalize_ecg')
    ns = dict(np=np, NORMALIZATION='per-record', EPS=1e-6)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<normalize_ecg>', 'exec'), ns)
    x = synthetic_data()['arrays']['train'][0].copy()
    hidden = sample_hidden(per_lead_masks(1,50,.5,42), x[None].shape)[0]
    changed = np.where(hidden, x+8, x)
    a,b = ns['normalize_ecg'](x), ns['normalize_ecg'](changed)
    assert np.max(np.abs(a-b)[~hidden]) > .1
    assert NORMALIZATION['includes_future_hidden_samples']


def test_full_factor010_cpu_initialization_forward_parity(cfg):
    cfg = dict(cfg, model_size='stmem-small')
    path = ROOT / 'archive/20261028_legacy/experiments/mae_cause_model.py'
    spec = importlib.util.spec_from_file_location('_independent_legacy_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    seed_all(19)
    original = reference.CauseMAE(dict(size='stmem-small', initialization='independent', fixed_time=True,
                                     patch_norm=False, sep=False, leadwise_decoder=False, model_seed=42), patch=50)
    seed_all(19)
    model = build_mae(cfg, 50)
    for key,value in model.state_dict().items():
        assert torch.equal(value, original.state_dict()[key]), key
    x = torch.from_numpy(synthetic_data()['arrays']['train'][:1])
    mask = torch.from_numpy(per_lead_masks(1,50,.5,9))
    original.eval(); model.eval()
    with torch.no_grad():
        a,b = original(x,mask), model(x,mask)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])


def test_fixed_time_frozen_other_weights_learn_and_mae_hidden_isolation(cfg):
    model = build_mae(cfg,125)
    x = torch.from_numpy(synthetic_data()['arrays']['train'][:2])
    masks = torch.from_numpy(per_lead_masks(2,125,.5,9))
    model.eval()
    hidden = torch.from_numpy(sample_hidden(masks.numpy(), x.shape))
    with torch.no_grad():
        changed = x.clone(); changed[hidden] += 1000
        assert torch.equal(model(x,masks)[0], model(changed,masks)[0])
    before = {key: p.detach().clone() for key,p in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model.train()
    prediction,target = model(x,masks)
    loss = (prediction-target).square().mean(-1)[masks].mean()
    loss.backward(); optimizer.step()
    for key,p in model.named_parameters():
        if key in ('enc_time','dec_time'):
            assert not p.requires_grad and p.grad is None and torch.equal(p,before[key])
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all(), key
            assert not torch.equal(p,before[key]), key


def test_effective_batch_masks_do_not_depend_on_microbatch(cfg):
    data = get_data(cfg)
    x,m,identity = batch_for(data,cfg,'mae',4)
    changed = dict(cfg,microbatch=1,accumulation=4)
    x2,m2,identity2 = batch_for(data,changed,'mae',4)
    assert np.array_equal(x,x2) and np.array_equal(m,m2) and identity == identity2
    assert np.array_equal(np.concatenate([m2[i:i+1] for i in range(4)]),m)
    for update in range(1,8):
        assert np.array_equal(batch_indices(12,4,update,42),batch_indices(12,4,update,42))


def test_checkpoint_resume_preserves_weights_update_rng_and_next_batch(cfg,tmp_path):
    cfg = dict(cfg,patch_samples=125,updates=4,eval_every=2,checkpoint_every=1)
    data = get_data(cfg)
    banks = {'validation':bank_at(tmp_path/'banks',data,cfg,'validation',pilot=True)}
    seed_all(71)
    full,_ = train_model(build_mae(cfg,125),'mae',data,cfg,tmp_path/'full',{'contract':'test'},banks)
    seed_all(71)
    train_model(build_mae(cfg,125),'mae',data,cfg,tmp_path/'split',{'contract':'test'},banks,stop_after=2)
    checkpoint = load_checkpoint(tmp_path/'split/checkpoint.pt')
    assert checkpoint['update'] == 2
    assert checkpoint['next_batch'] == batch_for(data,cfg,'mae',3)[2]
    seed_all(992)  # Resume must restore checkpoint RNG and data flow.
    resumed,resource = train_model(build_mae(cfg,125),'mae',data,cfg,tmp_path/'split',{'contract':'test'},banks,resume=True)
    assert resource['update'] == 4
    for key,value in full.state_dict().items():
        assert torch.equal(value,resumed.state_dict()[key]),key
    first = load_checkpoint(tmp_path/'full/checkpoint.pt')
    second = load_checkpoint(tmp_path/'split/checkpoint.pt')
    assert first['sequence'] == second['sequence'] and first['next_batch'] == second['next_batch']
    assert torch.equal(first['rng']['torch'],second['rng']['torch'])
    with pytest.raises(ValueError,match='identity mismatch'):
        train_model(build_mae(cfg,125),'mae',data,cfg,tmp_path/'split',{'contract':'changed'},banks,resume=True)


def write_artifact(folder):
    folder.mkdir()
    source = synthetic_data()
    entries = {}
    for split in ('train','validation'):
        x,frame = source['arrays'][split],source['frames'][split]
        frame.to_csv(folder/f'metadata_{split}.csv',index=False)
        path=folder/f'signals_{split}.npy'
        np.save(path,x)
        entries[split]=dict(count=len(x),metadata=f'metadata_{split}.csv',signals=dict(
            file=path.name,shape=list(x.shape),dtype='float32',bytes=path.stat().st_size))
    manifest=dict(config=PREPROCESSING,lead_order=LEAD_NAMES,preprocessing_order=ORDER,splits=entries)
    (folder/'manifest.json').write_text(json.dumps(manifest))
    # Deliberately invalid test/label files must never be opened by Phase 1.
    for name in ('signals_test.npy','labels_test.npy','metadata_test.csv','labels_train.npy','labels_validation.npy'):
        (folder/name).write_bytes(b'FORBIDDEN')
    return manifest


def test_artifact_never_opens_test_or_label_files_and_rejects_40hz(tmp_path,monkeypatch):
    import builtins
    folder=tmp_path/'artifact'
    manifest=write_artifact(folder)
    original_open=Path.open
    opened=[]
    def guarded(path,*a,**kw):
        if Path(path).parent == folder:
            opened.append(Path(path).name)
            assert 'test' not in Path(path).name and 'label' not in Path(path).name
        return original_open(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    builtin_open = builtins.open
    def guarded_builtin(path,*a,**kw):
        if isinstance(path,(str,Path)) and Path(path).parent == folder:
            assert 'test' not in Path(path).name and 'label' not in Path(path).name
        return builtin_open(path,*a,**kw)
    monkeypatch.setattr(builtins,'open',guarded_builtin)
    assert load_artifact(folder)['test_opened'] is False
    assert not any('test' in path or 'label' in path for path in opened)
    manifest['config']=dict(PREPROCESSING,bandpass_high=40.)
    (folder/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Never refilter 40 Hz'):
        load_artifact(folder)


def test_artifact_alignment_and_patient_overlap_are_errors(tmp_path):
    folder=tmp_path/'artifact'; write_artifact(folder)
    path=folder/'metadata_validation.csv'
    frame=pd.read_csv(path); frame.loc[0,'patient_id']=0; frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='Patient overlap'):
        load_artifact(folder)
    frame.loc[0,'patient_id']=10000; frame.loc[0,'artifact_index']=9; frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match='alignment mismatch'):
        load_artifact(folder)


def test_real_data_missing_is_error_and_never_falls_back(tmp_path,cfg):
    with pytest.raises(FileNotFoundError,match='--artifact-dir'):
        get_data(dict(cfg,synthetic=False,artifact_dir=str(tmp_path/'missing')))
    from experiments.phase1.training import device_for
    with pytest.raises(RuntimeError,match='allocated CUDA'):
        device_for(dict(cfg,synthetic=False))


def test_no_evaluable_masks_are_rejected(cfg):
    frame=synthetic_data()['frames']['train']
    with pytest.raises(ValueError,match='No evaluable masks'):
        make_bank(frame,[],50,.5,1,42,'score')
    with pytest.raises(ValueError,match='No finite evaluable masks'):
        aggregate(pd.DataFrame())
    with pytest.raises(ValueError,match='Sensitivity requires one'):
        run_pilot(dict(cfg,patches=[25,50,125],mask_ratio=.6))


def test_preprocessing_train_validation_only_from_other_cwd(tmp_path,monkeypatch):
    import wfdb
    from preprocessing.run_preprocessing import run
    raw=tmp_path/'raw'; raw.mkdir()
    record_dir=raw/'records500'; record_dir.mkdir()
    time=np.arange(5000)/500
    wave=np.stack([np.sin(2*np.pi*3*time+i/12)+.2*np.sin(2*np.pi*45*time) for i in range(12)],axis=1)
    records=[]
    for number,fold in enumerate([1,9,10],1):
        name=f'ecg{number}'
        if fold != 10:
            wfdb.wrsamp(name,fs=500,units=['mV']*12,sig_name=LEAD_NAMES,p_signal=wave,fmt=['16']*12,write_dir=str(record_dir))
        records.append(dict(ecg_id=number,patient_id=number,strat_fold=fold,scp_codes="{'NORM': 100.0}",filename_hr=f'records500/{name}'))
    pd.DataFrame(records).to_csv(raw/'ptbxl_database.csv',index=False)
    pd.DataFrame(dict(diagnostic=[1],diagnostic_class=['NORM']),index=['NORM']).to_csv(raw/'scp_statements.csv')
    reader=wfdb.rdsamp
    def guarded(record,*args,**kwargs):
        assert Path(record).name!='ecg3'
        return reader(record,*args,**kwargs)
    monkeypatch.setattr(wfdb,'rdsamp',guarded)
    monkeypatch.chdir(tmp_path)
    output=tmp_path/'artifact'
    manifest=run(data_dir='raw',output_dir='artifact',splits=('train','validation'),progress_every=1)
    assert set(manifest['splits'])=={'train','validation'}
    assert not (output/'signals_test.npy').exists() and not (output/'labels_test.npy').exists()
    assert load_artifact(output)['test_opened'] is False
    # CLI aliases and caller-relative paths also replay the same artifact.
    from preprocessing.run_preprocessing import main as preprocessing_main
    monkeypatch.setattr(sys,'argv',['run_preprocessing.py','--raw-dir','raw','--artifact-dir','artifact'])
    preprocessing_main()
    reused=json.loads((output/'manifest.json').read_text())
    assert reused['generated_at_utc']==manifest['generated_at_utc']


def test_learning_rate_contract_and_ridge_seconds_fixed(cfg):
    real=resolve_config(environ={})
    assert learning_rate(100,3000,real,'mae') == pytest.approx(3e-4)
    assert learning_rate(3000,3000,real,'mae') == pytest.approx(3e-5)
    assert ridge_lags(dict(cfg,patch_samples=25)) == [-100,-50,-1,0,1,50,100]
    assert ridge_lags(dict(cfg,patch_samples=25,ridge_lag_mode='legacy_patch_scaled')) == [-50,-25,-1,0,1,25,50]


def test_fit_score_end_to_end_same_banks_and_observed_patch_counts(cfg):
    root,ok=run_fit(cfg,run_id='contract')
    assert ok
    result=json.loads((root/'fit_results.json').read_text())
    assert len({row['train_sequence_sha256'] for row in result if row['estimator']!='ridge'}) == 1
    scored,ok=run_score(cfg,root)
    assert scored == root and ok
    bank=pd.read_csv(root/'banks/score.csv')
    for name in ('ridge','cnn','transformer'):
        metrics=pd.read_csv(root/name/'score/mask_metrics.csv')
        assert metrics.mask_id.tolist() == bank.mask_id.tolist()
        table=pd.read_csv(root/name/'score/aggregate_map.csv')
        assert (table.hidden_observations > 0).all()
        assert table.hidden_observations.sum() == len(bank)*300
        patches=np.load(root/name/'score/ecg_maps.npz')
        assert np.isfinite(patches['nrmse']).all()
    assert not (root/'mask_difficulty.csv').exists()
    with pytest.raises(FileExistsError,match='immutable'):
        run_score(cfg,root)


def test_three_patch_pilot_and_profile_are_separate_synthetic_stages(cfg):
    cfg=dict(cfg,patches=[25,50,125],updates=1,profile_updates=1)
    root,ok=run_pilot(cfg,run_id='all-patches')
    assert ok
    results=json.loads((root/'results.json').read_text())
    assert [row['patch_seconds'] for row in results]==[.1,.2,.5]
    streams=[load_checkpoint(root/f'p{p}_r0.5/checkpoint.pt')['next_batch']['artifact_indices'] for p in cfg['patches']]
    assert streams[0]==streams[1]==streams[2]
    profile,ok=run_pilot(cfg,run_id='single-profile',profile=True)
    assert ok
    assert len(json.loads((profile/'results.json').read_text()))==1
    assert json.loads((profile/'status.json').read_text())['stage']=='profile'


def test_one_failure_preserves_other_estimators_and_reports_partial(cfg,monkeypatch):
    import experiments.phase1.pipeline as pipeline
    build=pipeline.build_estimator
    def fail_one(name,cfg):
        if name=='cnn':
            raise ValueError('deliberate estimator exclusion')
        return build(name,cfg)
    monkeypatch.setattr(pipeline,'build_estimator',fail_one)
    root,ok=run_fit(cfg)
    assert not ok and (root/'ridge/fit_complete.json').exists() and (root/'transformer/fit_complete.json').exists()
    assert json.loads((root/'status.json').read_text())['status']=='partial'
    _,ok=run_score(cfg,root)
    assert not ok and len(pd.read_csv(root/'estimator_summary.csv'))==4
    assert 'incomplete' in json.loads((root/'cnn/score_error.json').read_text())['error']


def test_absolute_entrypoint_from_another_cwd_and_missing_data_error(tmp_path):
    # Interpreter, dependency path and repository are inherited, not guessed.
    env=dict(os.environ)
    cfg=ROOT/'configs/phase1_smoke.json'
    runner=ROOT/'experiments/run_phase1.py'
    result=subprocess.run([sys.executable,str(runner),'plan','--config',str(cfg),'--output-root',str(tmp_path/'plan')],
        cwd=tmp_path,env=env,text=True,capture_output=True,timeout=90)
    assert result.returncode==0,result.stderr
    assert 'Plain ECG-MAE factor_010' in result.stdout
    result=subprocess.run([sys.executable,str(runner),'preflight','--artifact-dir',str(tmp_path/'missing'),
                           '--output-root',str(tmp_path/'out'),'--cache-dir',str(tmp_path/'cache')],
        cwd=tmp_path,env=env,text=True,capture_output=True,timeout=90)
    assert result.returncode!=0 and str(tmp_path/'missing/manifest.json') in result.stderr
    assert '--artifact-dir' in result.stderr and 'SMOKE=' not in result.stdout


def test_slurm_scripts_lf_export_paths_limits_and_no_automatic_submission():
    submit=(ROOT/'scripts/submit_phase1.sh').read_bytes()
    worker=(ROOT/'scripts/phase1_worker.sh').read_bytes()
    assert b'\r' not in submit+worker
    for token in (b'--gres=gpu:normal:1',b'--export=ALL',b'batch_ugrad_advisor_x',b'ugrad_advisor_x',b'"$@"'):
        assert token in submit
    assert b'SLURM_JOB_ID' in worker and b'hours > 24' in submit and b'hours > 4' in submit
    assert submit.index(b'mkdir -p') < submit.index(b'sbatch --parsable')
