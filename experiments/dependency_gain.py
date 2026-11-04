"""Paired context interventions; reconstruction gain is not mutual information."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from experiments.phase1.common import digest, file_sha, source_identity, write_json
from experiments.phase1.data import assert_disjoint, load_artifact, synthetic_data, NORMALIZATION
from experiments.phase1.masks import batch_indices, mask_digest, per_lead_masks, seed_for
from experiments.phase1.model import build_mae
from experiments.phase1.training import (atomic_checkpoint, device_for, learning_rate,
    load_checkpoint, preserve_rng, restore_rng, rng_state, runtime, seed_all, sync)

CONDITIONS = ('w0_c0', 'w1_c0', 'w0_c1', 'w1_c1', 'random')
ARMS = {'within': 'w1_c0', 'cross': 'w0_c1', 'random': 'random'}
VERSION = 'paired-context-v1'


def context_masks(targets, priorities, radius=2):
    """True=hidden. All contexts hide exactly 25/50 patches on EACH lead.

    Same anchor and random priorities in all five contexts. W controls +/-radius
    immediate time patches on the anchor lead; C controls the SAME time patch on
    the other 11 leads. Remaining positions compensate, using shared priorities.
    This is a fixed-budget context intervention, not removal of all information
    from an entire source. No wraparound for temporal neighbours.
    """
    targets, priorities = np.asarray(targets), np.asarray(priorities)
    if priorities.shape != (len(targets), 12, 50):
        raise ValueError('P50 / 12 x 2500 only; priorities must have shape (B,12,50)')
    if not isinstance(radius, int) or not 1 <= radius <= 12:
        raise ValueError('radius must be an integer in [1,12]')
    if targets.shape != (len(targets), 2) or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError('Integer (lead,time) targets required')
    if not np.isfinite(priorities).all():
        raise ValueError('Nonfinite mask priorities')
    result = {}
    for condition in CONDITIONS:
        mask = np.empty_like(priorities, dtype=bool)
        for b, (lead, t) in enumerate(targets):
            if not (0 <= lead < 12 and radius <= t < 50-radius):
                raise ValueError('Anchor must have a complete non-wrapping temporal neighbourhood')
            forced_hidden, forced_visible = np.zeros((12,50), bool), np.zeros((12,50), bool)
            forced_hidden[lead, t] = True
            if condition != 'random':
                w, c = int(condition[1]), int(condition[4])
                neighbours = np.r_[t-radius:t, t+1:t+radius+1]
                (forced_visible if w else forced_hidden)[lead, neighbours] = True
                others = np.arange(12) != lead
                (forced_visible if c else forced_hidden)[others, t] = True
            for channel in range(12):
                available = np.flatnonzero(~(forced_hidden[channel] | forced_visible[channel]))
                count = 25-int(forced_hidden[channel].sum())
                if not 0 <= count <= len(available):
                    raise ValueError('Infeasible fixed-budget context intervention')
                selected = available[np.argsort(priorities[b,channel,available], kind='stable')[:count]]
                mask[b,channel] = forced_hidden[channel]
                mask[b,channel,selected] = True
        result[condition] = mask.reshape(len(targets), 600)
    return result


def draw_contexts(batch, seed, radius):
    rng = np.random.default_rng(seed)
    targets = np.column_stack([rng.integers(12, size=batch), rng.integers(radius,50-radius,size=batch)])
    priorities = rng.random((batch,12,50))
    return targets, context_masks(targets, priorities, radius)


def choose_data(cfg, options):
    """Diagnostic ECG PATIENTS held out of all pretraining/fork gradients."""
    data = synthetic_data() if cfg['synthetic'] else load_artifact(cfg['artifact_dir'])
    frame = data['frames']['train']
    rng = np.random.default_rng(cfg['subset_seed'])
    patients = rng.permutation(frame.patient_id.unique())
    held = []
    for patient in patients:
        held.append(patient)
        if frame.patient_id.isin(held).sum() >= options['probe_ecgs']:
            break
    probe_pool = np.flatnonzero(frame.patient_id.isin(held))
    fit_pool = np.flatnonzero(~frame.patient_id.isin(held))
    def sample(pool, count):
        if count < 1 or len(pool) < count:
            raise ValueError(f'Need {count} ECGs, available {len(pool)} after patient separation')
        return np.sort(rng.choice(pool, count, replace=False))
    train = sample(fit_pool, cfg['train_ecgs'])
    probe = sample(probe_pool, options['probe_ecgs'])
    val = sample(np.arange(len(data['frames']['validation'])), cfg['validation_ecgs'])
    assert_disjoint(frame.iloc[train], frame.iloc[probe], data['frames']['validation'].iloc[val])
    # Fixed lead denominator, computed on GRADIENT-TRAINING ECGs only.
    from experiments.predictability_estimators import lead_scale
    mean, scale = lead_scale(data['arrays']['train'], train)
    selection = pd.concat([data['frames']['validation' if role == 'validation' else 'train'].iloc[ix].assign(pool=role)
        for role, ix in [('train',train), ('gain_probe',probe), ('validation',val)]], ignore_index=True)
    return dict(data, selected=dict(train=train, gain_probe=probe, validation=val),
        selection=selection, scale=scale, mean=mean, selection_id=digest(selection.to_dict('records')))


def panel(data, role, options, seed):
    indices, anchors, masks, rows = [], [], {c: [] for c in CONDITIONS}, []
    for index in data['selected'][role]:
        split = 'validation' if role == 'validation' else 'train'
        row = data['frames'][split].iloc[index]
        for repeat in range(options['panel_repeats']):
            key = seed_for(seed, 'panel', role, int(row.ecg_id), repeat)
            target, contexts = draw_contexts(1, key, options['radius'])
            indices.append(index)
            anchors.append(target[0])
            rows.append(dict(ecg_id=int(row.ecg_id), patient_id=int(row.patient_id),
                artifact_index=int(index), repeat=repeat, lead=int(target[0,0]), time=int(target[0,1]),
                mask_seed=key, role=role))
            for c in CONDITIONS:
                masks[c].append(contexts[c][0])
    return dict(indices=np.asarray(indices), targets=np.stack(anchors), metadata=pd.DataFrame(rows),
                masks={c: np.stack(v) for c,v in masks.items()}, split=split)


def target_losses(prediction, x, targets, scale):
    """Per-ECG loss on the IDENTICAL single hidden anchor, not context-dependent targets."""
    b = torch.arange(len(x), device=x.device)
    target = x.reshape(len(x),12,50,50)[b,targets[:,0],targets[:,1]]
    pred = prediction.reshape_as(x).reshape(len(x),12,50,50)[b,targets[:,0],targets[:,1]]
    return (((pred-target)/scale[targets[:,0],None])**2).mean(-1)


def hidden_losses(prediction, x, masks, scale):
    residual = ((prediction.reshape_as(x)-x)/scale[None,:,None]).reshape(len(x),600,50)
    return (residual.square().mean(-1)*masks).sum(-1)/masks.sum(-1)


def evaluate_panel(model, data, bank, cfg, device):
    rows = []
    scale = torch.as_tensor(data['scale'], dtype=torch.float32, device=device)
    previous = model.training
    try:
        model.eval()
        with preserve_rng(), torch.no_grad():
            for condition in CONDITIONS:
                values, all_values = [], []
                for start in range(0,len(bank['indices']),cfg['microbatch']):
                    stop = start+cfg['microbatch']
                    x = torch.from_numpy(np.asarray(data['arrays'][bank['split']][bank['indices'][start:stop]],
                                                   dtype=np.float32)).to(device)
                    masks = torch.from_numpy(bank['masks'][condition][start:stop]).to(device)
                    targets = torch.as_tensor(bank['targets'][start:stop], device=device)
                    prediction = model(x,masks)[0]
                    values.extend(target_losses(prediction,x,targets,scale).cpu().tolist())
                    all_values.extend(hidden_losses(prediction,x,masks,scale).cpu().tolist())
                rows.append(bank['metadata'].assign(condition=condition, anchor_mse=values, hidden_mse=all_values))
    finally:
        model.train(previous)
    result = pd.concat(rows, ignore_index=True)
    if not np.isfinite(result[['anchor_mse','hidden_mse']]).all().all():
        raise RuntimeError('Nonfinite panel evaluation')
    return result


def gains(frame):
    wide = frame.pivot(index=['ecg_id','patient_id','repeat','lead','time'], columns='condition', values='anchor_mse')
    # Context-averaged (factorial) marginal error reductions; retain signed values.
    wide['gain_within'] = .5*((wide.w0_c0-wide.w1_c0)+(wide.w0_c1-wide.w1_c1))
    wide['gain_cross'] = .5*((wide.w0_c0-wide.w0_c1)+(wide.w1_c0-wide.w1_c1))
    wide['interaction'] = wide.w1_c0+wide.w0_c1-wide.w0_c0-wide.w1_c1
    wide['gain_contrast'] = wide.gain_within-wide.gain_cross
    return wide.reset_index()


def optimizer_for(model, cfg, lr):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            betas=tuple(cfg['betas']), weight_decay=cfg['weight_decay'])


def optimize(model, optimizer, x, mask, targets, cfg, scale, device, loss_kind):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    value = 0.
    scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    for start in range(0,len(x),cfg['microbatch']):
        stop = start+cfg['microbatch']
        xt, mt = torch.from_numpy(x[start:stop]).to(device), torch.from_numpy(mask[start:stop]).to(device)
        prediction = model(xt,mt)[0]
        if loss_kind == 'anchor':
            tt = torch.as_tensor(targets[start:stop], device=device)
            loss = target_losses(prediction,xt,tt,scale).mean()
        else:
            loss = hidden_losses(prediction,xt,mt,scale).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        weight = len(xt)/len(x)
        (loss*weight).backward()
        value += float(loss.detach())*weight
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite training gradient')
    optimizer.step()
    return value


def training_batch(data, cfg, step, namespace):
    size = cfg['microbatch']*cfg['accumulation']
    pos = batch_indices(len(data['selected']['train']),size,step,seed_for(cfg['seed'],namespace))
    ix = data['selected']['train'][pos]
    return np.asarray(data['arrays']['train'][ix],dtype=np.float32), ix


def pretrain(root, data, cfg, options, device):
    """One common cosine horizon; intermediate checkpoints are from the SAME run."""
    folder = root/'pretrain'
    folder.mkdir(exist_ok=True)
    model = build_mae(cfg,50).to(device)
    optimizer = optimizer_for(model,cfg,cfg['lr'])
    path = folder/'latest.pt'
    update, history = 0, []
    if path.exists():
        state = load_checkpoint(path)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        update, history = state['update'], state['history']
        restore_rng(state['rng'])
    for step in range(update+1,max(options['checkpoints'])+1):
        sync(device)
        mark = time.perf_counter()
        x, ix = training_batch(data,cfg,step,'pretrain')
        mask = per_lead_masks(len(x),50,.5,seed_for(cfg['mask_seed'],cfg['seed'],'pretrain',step))
        lr = learning_rate(step,max(options['checkpoints']),cfg,'mae')
        for group in optimizer.param_groups:
            group['lr'] = lr
        loss = optimize(model,optimizer,x,mask,None,cfg,np.ones(12),device,'all-hidden')
        sync(device)
        history.append(dict(update=step,loss=loss,seconds=time.perf_counter()-mark,lr=lr,
                            batch_sha256=digest(ix.tolist()),mask_sha256=mask_digest(mask)))
        if step % options['save_every'] == 0 or step in options['checkpoints']:
            state = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),update=step,
                         history=history,rng=rng_state())
            # Selected base first, then latest: resume cannot skip an unsaved selected checkpoint.
            if step in options['checkpoints']:
                atomic_checkpoint(folder/f'update_{step}.pt',state)
            atomic_checkpoint(path,state)
            pd.DataFrame(history).to_csv(folder/'updates.csv',index=False)
            print(f'seed={cfg["seed"]} random pretrain={step}/{max(options["checkpoints"])} loss={loss:.6f}',flush=True)
    del optimizer, model


def fork_train(root, state, base_sha, arm, data, cfg, options, device):
    folder = root/arm
    folder.mkdir(exist_ok=True)
    model = build_mae(cfg,50).to(device)
    model.load_state_dict(state['model'])
    optimizer = optimizer_for(model,cfg,options['fork_lr'])
    # Same weights, ZERO optimizer states, dropout RNG, ECGs, anchors and priorities.
    # Fresh optimizer is deliberate because the primary fork uses anchor-only loss.
    restore_rng(state['rng'])
    path = folder/'latest.pt'
    update, history = 0, []
    if path.exists():
        previous = load_checkpoint(path)
        if previous['base_sha256'] != base_sha:
            raise ValueError('Fork base checkpoint changed')
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        update, history = previous['update'], previous['history']
        restore_rng(previous['rng'])
    for step in range(update+1,options['fork_updates']+1):
        sync(device)
        mark = time.perf_counter()
        x, ix = training_batch(data,cfg,step,f'fork-{state["update"]}')
        targets, contexts = draw_contexts(len(x),seed_for(cfg['mask_seed'],cfg['seed'],
            'fork',state['update'],step),options['radius'])
        mask = contexts[ARMS[arm]]
        loss = optimize(model,optimizer,x,mask,targets,cfg,data['scale'],device,options['fork_loss'])
        sync(device)
        history.append(dict(update=step,loss=loss,seconds=time.perf_counter()-mark,
            lr=options['fork_lr'],batch_sha256=digest(ix.tolist()),target_sha256=digest(targets.tolist()),
            mask_sha256=mask_digest(mask)))
        if step % options['save_every'] == 0 or step == options['fork_updates']:
            atomic_checkpoint(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                update=step,history=history,rng=rng_state(),base_sha256=base_sha))
            pd.DataFrame(history).to_csv(folder/'updates.csv',index=False)
            print(f'seed={cfg["seed"]} base={state["update"]} arm={arm} fork={step}/{options["fork_updates"]} loss={loss:.6f}',flush=True)
    return model, history


def paired_effects(before, after):
    keys = ['ecg_id','patient_id','repeat','lead','time','condition']
    result = before.merge(after,on=keys,validate='one_to_one',suffixes=('_before','_after'))
    result['anchor_improvement'] = result.anchor_mse_before-result.anchor_mse_after
    result['hidden_improvement'] = result.hidden_mse_before-result.hidden_mse_after
    return result


def rank_association(x,y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    # No row-wise p-value: checkpoints from the same run are NOT independent.
    return float(spearmanr(x,y).statistic)


def summarize(root, synthetic):
    gain_rows, effects = [], []
    for folder in sorted(root.glob('seed_*/base_*')):
        if not (folder/'completed.json').exists():
            continue
        seed, checkpoint = int(folder.parent.name[5:]), int(folder.name[5:])
        gain_rows.append(pd.read_csv(folder/'gains.csv').assign(seed=seed,checkpoint=checkpoint))
        for arm in ARMS:
            effects.append(pd.read_csv(folder/arm/'effects.csv').assign(seed=seed,checkpoint=checkpoint,arm=arm))
    if not gain_rows:
        return None
    gain, effect = pd.concat(gain_rows,ignore_index=True), pd.concat(effects,ignore_index=True)
    # Equal weight per ECG; each ECG has identical repeats and conditions.
    g = gain.groupby(['seed','checkpoint'])[['gain_within','gain_cross','gain_contrast','interaction']].mean()
    baselines = gain.groupby(['seed','checkpoint'])[list(CONDITIONS)].mean().add_prefix('probe_loss_')
    primary = effect.groupby(['seed','checkpoint','arm']).anchor_improvement.mean().unstack('arm')
    full = effect.groupby(['seed','checkpoint','arm']).hidden_improvement.mean().unstack('arm')
    table = g.join(baselines).join(primary.add_prefix('improvement_')).join(full.add_prefix('hidden_improvement_'))
    table['validation_anchor_mse_before'] = effect.groupby(['seed','checkpoint']).anchor_mse_before.mean()
    table['effect_contrast'] = table.improvement_within-table.improvement_cross
    table['within_vs_random'] = table.improvement_within-table.improvement_random
    table['cross_vs_random'] = table.improvement_cross-table.improvement_random
    table['relative_gain_contrast'] = table.gain_contrast/table.probe_loss_w0_c0.clip(lower=1e-12)
    table['relative_effect_contrast'] = table.effect_contrast/table.validation_anchor_mse_before.clip(lower=1e-12)
    table = table.reset_index()
    table.to_csv(root/'hypothesis_units.csv',index=False)
    # Correlation across seed/checkpoint units is EXPLORATORY, not a 9-independent-runs test.
    associations = dict(pooled_exploratory_spearman=rank_association(table.gain_contrast,table.effect_contrast),
        baseline_normalized_exploratory_spearman=rank_association(table.relative_gain_contrast,table.relative_effect_contrast),
        within_gain_vs_random_relative_effect=rank_association(table.gain_within,table.within_vs_random),
        cross_gain_vs_random_relative_effect=rank_association(table.gain_cross,table.cross_vs_random),
        within_seed_spearman={str(s):rank_association(v.gain_contrast,v.effect_contrast) for s,v in table.groupby('seed')},
        independent_training_seeds=int(table.seed.nunique()), seed_checkpoint_units=len(table),
        inference='Descriptive pilot only: checkpoints share a training trajectory; no independent-row p-value.',
        gain_contrast_identity='gain_within - gain_cross == L(w0,c1) - L(w1,c0); identical to this context-loss contrast, not an independent predictor.',
        confirmatory_requirement='Freeze direction/policy on discovery seeds, then evaluate on NEW training seeds/checkpoints.',
        synthetic=synthetic, real_ecg_performance=not synthetic, downstream_evaluated=False)
    write_json(root/'association.json',associations)
    lines = ['# Within/Cross Reconstruction Gain pilot', '',
        'SYNTHETIC SOFTWARE CHECK ONLY — NOT ECG EVIDENCE.' if synthetic else 'Exploratory real ECG reconstruction experiment; no downstream classification results.', '',
        'Positive improvement = lower common-panel loss after additional training.',
        'Primary outcome: equal-weight mean anchor MSE decrease over all five fixed validation contexts.',
        'Gain is measured on held-out TRAIN patients; improvement is measured on validation fold 9.', '',
        table.to_csv(index=False), '', json.dumps(associations,ensure_ascii=False,indent=2), '',
        'No automatic conclusion of support/rejection. Inspect direction and consistency across independent seeds.',
        'Negative reconstruction gains are retained. Fixed-budget swaps and model limitations can produce them.',
        'A two-arm gap correlation alone cannot establish a unique information-gain advantage; compare against context loss/high-loss and Random.',
        'Primary anchor-only fine-tuning is a mechanism test, not standard full-mask MAE pretraining.',
        'Do not infer downstream AUROC improvement from reconstruction loss.']
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return table


def run_experiment(cfg, options, output, resume=False):
    root = Path(output).resolve()
    torch.set_num_threads(cfg['cpu_threads'])
    device = device_for(cfg)
    data = choose_data(cfg,options)
    sources = source_identity()
    for name in ['dependency_gain.py','run_dependency_gain.py']:
        path = Path(__file__).parent/name
        sources[f'experiments/{name}'] = file_sha(path)
    contract = dict(version=VERSION, config={k:v for k,v in cfg.items() if k not in
        ('raw_dir','artifact_dir','output_root','cache_dir')},options=options,
        source=sources,data_id=data['data_id'],selection_id=data['selection_id'])
    if root.exists():
        if not resume:
            raise FileExistsError(f'Output exists: {root}; use --resume without changing the experiment')
        if json.loads((root/'contract.json').read_text(encoding='utf-8')) != contract:
            raise ValueError('Source/config/data/selection changed; create a new output directory')
    else:
        root.mkdir(parents=True)
        write_json(root/'contract.json',contract)
    write_json(root/'status.json',dict(status='running',synthetic=cfg['synthetic'],test_opened=False))
    data['selection'].to_csv(root/'ecg_selection.csv',index=False)
    write_json(root/'data.json',dict(hashes=data['hashes'],normalization=NORMALIZATION,
        scale=data['scale'],scale_pool='gradient-training only',test_opened=False,label_files_opened=False))
    started = time.perf_counter()
    try:
        for seed in options['seeds']:
            local = dict(cfg,seed=seed)
            seed_all(seed)
            folder = root/f'seed_{seed}'
            folder.mkdir(exist_ok=True)
            write_json(folder/'runtime.json',runtime(device,local))
            # Same panel targets/masks across all seeds/checkpoints/arms.
            probe, validation = [panel(data,role,options,cfg['mask_seed']) for role in ('gain_probe','validation')]
            for role,bank in [('gain_probe',probe),('validation',validation)]:
                bank['metadata'].to_csv(folder/f'{role}_panel.csv',index=False)
                np.savez_compressed(folder/f'{role}_masks.npz',targets=bank['targets'],**bank['masks'])
            pretrain(folder,data,local,options,device)
            for checkpoint in options['checkpoints']:
                base = folder/f'base_{checkpoint}'
                if (base/'completed.json').exists():
                    continue
                base.mkdir(exist_ok=True)
                path = folder/'pretrain'/f'update_{checkpoint}.pt'
                state, sha = load_checkpoint(path), file_sha(path)
                model = build_mae(local,50).to(device)
                model.load_state_dict(state['model'])
                before = evaluate_panel(model,data,validation,local,device)
                probe_frame = evaluate_panel(model,data,probe,local,device)
                before.to_csv(base/'validation_before.csv',index=False)
                probe_frame.to_csv(base/'gain_probe_losses.csv',index=False)
                gains(probe_frame).to_csv(base/'gains.csv',index=False)
                del model
                sequences = []
                for arm in ARMS:
                    model, history = fork_train(base,state,sha,arm,data,local,options,device)
                    after = evaluate_panel(model,data,validation,local,device)
                    after.to_csv(base/arm/'validation_after.csv',index=False)
                    paired_effects(before,after).to_csv(base/arm/'effects.csv',index=False)
                    sequences.append([(h['batch_sha256'],h['target_sha256']) for h in history])
                    write_json(base/arm/'resources.json',dict(updates=len(history),
                        training_seconds=sum(h['seconds'] for h in history),base_sha256=sha,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0))
                    del model
                if not all(s == sequences[0] for s in sequences):
                    raise RuntimeError('Fork ECG/target sequences differ')
                write_json(base/'completed.json',dict(base_sha256=sha,matched_batch_and_target_flow=True))
                summarize(root,cfg['synthetic'])
        summarize(root,cfg['synthetic'])
        pretrain_seconds, fork_seconds, updates = 0., 0., 0
        for path in root.glob('seed_*/pretrain/updates.csv'):
            history = pd.read_csv(path)
            pretrain_seconds += float(history.seconds.sum())
            updates += len(history)
        for path in root.glob('seed_*/base_*/*/updates.csv'):
            history = pd.read_csv(path)
            fork_seconds += float(history.seconds.sum())
            updates += len(history)
        write_json(root/'compute_cost.json',dict(pretrain_training_seconds=pretrain_seconds,
            fork_training_seconds=fork_seconds,total_training_seconds=pretrain_seconds+fork_seconds,
            total_optimizer_updates=updates,current_session_wall_seconds=time.perf_counter()-started,
            note='Training totals include completed updates across resumes; session wall time includes panel evaluation and I/O, but excludes earlier sessions.'))
        write_json(root/'status.json',dict(status='completed',synthetic=cfg['synthetic'],test_opened=False,
            downstream_evaluated=False,session_seconds=time.perf_counter()-started))
    except Exception as error:
        write_json(root/'status.json',dict(status='failed',error=str(error),synthetic=cfg['synthetic'],test_opened=False))
        raise
    return root
