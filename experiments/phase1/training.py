"""FP32 effective-batch training, fixed banks, identity-checked resumable state."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from experiments.predictability_estimators import normalized_masked_loss
from .common import digest, write_json
from .config import semantic_config
from .masks import batch_indices, per_lead_masks, seed_for, mask_digest
from .metrics import measure, aggregate


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu())
    if value['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in value['cuda']])


@contextmanager
def preserve_rng():
    value = rng_state()
    try:
        yield
    finally:
        restore_rng(value)


def runtime(device, cfg):
    import platform
    import scipy
    result = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
        scipy=scipy.__version__, pandas=pd.__version__, cuda=torch.version.cuda,
        device=str(device), precision='fp32', effective_batch=cfg['microbatch']*cfg['accumulation'],
        microbatch=cfg['microbatch'], accumulation=cfg['accumulation'],
        gpu_name=None, gpu_memory_bytes=None)
    if device.type == 'cuda':
        prop = torch.cuda.get_device_properties(device)
        result.update(gpu_name=prop.name, gpu_memory_bytes=prop.total_memory)
    return result


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def device_for(cfg, *, real=True):
    device = torch.device(cfg['device'])
    if cfg['synthetic'] and device.type != 'cpu':
        raise ValueError('Synthetic smoke runs on CPU')
    if not cfg['synthetic'] and real:
        import os
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Real-data training/profile/score requires an allocated CUDA GPU; no synthetic fallback')
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Real-data compute must run inside a Slurm allocation, not on a login node')
    return device


def waveform_prediction(model, x, mask, device, kind):
    if kind == 'ridge':
        return model.predict(x, mask)
    xt, mt = torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)
    prediction = model(xt, mt)
    if kind == 'mae':
        prediction = prediction[0].reshape_as(xt)
    return prediction.detach().cpu().numpy()


def evaluate(model, signals, bank, masks, scale, batch_size, device, kind, output=None):
    if kind != 'ridge':
        model.eval()
    rows, patches, examples = [], [], None
    with torch.no_grad(), preserve_rng():
        for start in range(0, len(bank), batch_size):
            meta = bank.iloc[start:start+batch_size]
            x = np.asarray(signals[meta.artifact_index.to_numpy()], dtype=np.float32)
            mask = masks[start:start+len(meta)]
            prediction = waveform_prediction(model, x, mask, device, kind)
            metrics, patch = measure(prediction, x, mask, scale)
            rows.append(pd.concat([meta.reset_index(drop=True), metrics], axis=1))
            patches.append(patch.astype(np.float32))
            if examples is None:
                n = min(2, len(x))
                examples = dict(target=x[:n], prediction=prediction[:n], mask=mask[:n],
                                mask_id=meta.mask_id.to_numpy(dtype=str)[:n])
    frame, patch = pd.concat(rows, ignore_index=True), np.concatenate(patches)
    summary = aggregate(frame)
    if output is not None:
        folder = Path(output)
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder / 'mask_metrics.csv', index=False)
        np.savez_compressed(folder / 'patch_errors.npz', nrmse=patch, mask_id=bank.mask_id.to_numpy(dtype=str))
        np.savez_compressed(folder / 'examples.npz', **examples)
        write_json(folder / 'metrics.json', summary)
        write_maps(folder, bank, patch)
    return summary


def write_maps(folder, bank, patches):
    """Observed RMS of standardized error; NaN is never filled with zero error."""
    valid = np.isfinite(patches)
    counts = valid.sum(0)
    squares = np.where(valid, patches**2, 0).sum(0)
    pooled = np.sqrt(np.divide(squares, counts, out=np.full_like(squares, np.nan), where=counts > 0))
    p = int(bank.patch_samples.iloc[0])
    q = 2500//p
    pd.DataFrame(dict(lead_index=np.repeat(np.arange(12), q), time_patch=np.tile(np.arange(q), 12),
        start_seconds=np.tile(np.arange(q)*p/250, 12), nrmse=pooled.ravel(),
        hidden_observations=counts.ravel())).to_csv(folder / 'aggregate_map.csv', index=False)
    ecg_ids = bank.ecg_id.unique()
    maps, observation_counts = [], []
    for ecg_id in ecg_ids:
        ix = bank.ecg_id.to_numpy() == ecg_id
        cnt = valid[ix].sum(0)
        sq = np.where(valid[ix], patches[ix]**2, 0).sum(0)
        maps.append(np.sqrt(np.divide(sq, cnt, out=np.full_like(sq, np.nan), where=cnt > 0)))
        observation_counts.append(cnt)
    np.savez_compressed(folder / 'ecg_maps.npz', ecg_id=ecg_ids, nrmse=np.stack(maps),
                        hidden_observations=np.stack(observation_counts))


def learning_rate(update, horizon, cfg, kind):
    base = cfg['lr'] if kind == 'mae' else cfg['estimator_lr']
    minimum = cfg['min_lr'] if kind == 'mae' else base*.1
    if update <= cfg['warmup']:
        return base*update/cfg['warmup']
    fraction = min(1., (update-cfg['warmup'])/max(1, horizon-cfg['warmup']))
    return minimum+(base-minimum)*.5*(1+np.cos(np.pi*fraction))


def batch_for(data, cfg, kind, update, bank=None, masks=None):
    batch = cfg['microbatch']*cfg['accumulation']
    if kind == 'mae':
        subset = data['selected']['pilot']
        positions = batch_indices(len(subset), batch, update, cfg['seed'])
        indices = subset[positions]
        mask = per_lead_masks(batch, cfg['patch_samples'], cfg['mask_ratio'],
                              seed_for(cfg['mask_seed'], cfg['seed'], 'train', update))
        ids = [f'ecg{int(data["frames"]["train"].iloc[i].ecg_id)}-u{update}-b{j}' for j, i in enumerate(indices)]
    else:
        positions = batch_indices(len(bank), batch, update, cfg['seed'])
        indices, mask = bank.iloc[positions].artifact_index.to_numpy(), masks[positions]
        ids = bank.iloc[positions].mask_id.tolist()
    identity = dict(ids=ids, artifact_indices=indices.tolist(), mask_sha256=mask_digest(mask))
    return np.asarray(data['arrays']['train'][indices], dtype=np.float32), mask, identity


def atomic_checkpoint(path, value):
    temp = Path(path).with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def load_checkpoint(path):
    # Checkpoints are local trusted experiment artifacts, not arbitrary downloads.
    return torch.load(path, map_location='cpu', weights_only=False)


def train_model(model, kind, data, cfg, folder, identity, evaluation_banks, *,
                fit_bank=None, fit_masks=None, resume=False, stop_after=None, budget=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    device = device_for(cfg)
    model.to(device)
    horizon = cfg['updates'] if kind == 'mae' else cfg['estimator_updates']
    budget = horizon if budget is None else budget
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=cfg['lr'] if kind == 'mae' else cfg['estimator_lr'], betas=tuple(cfg['betas']),
        weight_decay=cfg['weight_decay'] if kind == 'mae' else cfg['estimator_weight_decay'])
    contract = dict(config=semantic_config(cfg), identity=identity, kind=kind, horizon=horizon, budget=budget)
    checkpoint = folder / 'checkpoint.pt'
    history, timings, sequence, update = [], [], [], 0
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f'Existing checkpoint {checkpoint}; explicitly resume its run')
        state = load_checkpoint(checkpoint)
        if state['contract'] != contract:
            raise ValueError('Checkpoint config/data/mask/source identity mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        update, history, timings, sequence = state['update'], state['history'], state['timings'], state['sequence']
        if batch_for(data, cfg, kind, update+1, fit_bank, fit_masks)[2] != state['next_batch']:
            raise ValueError('Checkpoint next data/mask flow mismatch')
        restore_rng(state['rng'])
    elif resume and (folder / 'history.csv').exists():
        raise ValueError('Partial training history has no resumable checkpoint')
    write_json(folder / 'runtime.json', runtime(device, cfg))
    print(json.dumps(runtime(device, cfg)), flush=True)
    scale = torch.tensor(np.ones(12) if kind == 'mae' else data['scale'], dtype=torch.float32, device=device)
    started = time.perf_counter()
    evaluation_seconds, checkpoint_seconds = 0., 0.
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    def snapshot(step, train_loss=None):
        nonlocal evaluation_seconds
        sync(device)
        mark = time.perf_counter()
        for pool, (bank, masks) in evaluation_banks.items():
            split = 'validation' if pool == 'validation' else 'train'
            metrics = evaluate(model, data['arrays'][split], bank, masks, data['scale'],
                               cfg['microbatch'], device, kind)
            history.append(dict(update=step, pool=pool, train_loss=train_loss, **metrics))
        sync(device)
        evaluation_seconds += time.perf_counter()-mark
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)

    if update == 0:
        snapshot(0)
    limit = min(budget, update+stop_after) if stop_after is not None else budget
    for step in range(update+1, limit+1):
        sync(device)
        mark = time.perf_counter()
        x, mask, batch_identity = batch_for(data, cfg, kind, step, fit_bank, fit_masks)
        # Entire effective batch and masks exist before any microbatch is taken.
        sequence.append(digest(batch_identity))
        sync(device)
        batch_ready = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(step, horizon, cfg, kind)
        for group in optimizer.param_groups:
            group['lr'] = lr
        loss_value = 0.
        for start in range(0, len(x), cfg['microbatch']):
            xt = torch.from_numpy(x[start:start+cfg['microbatch']]).to(device)
            mt = torch.from_numpy(mask[start:start+cfg['microbatch']]).to(device)
            prediction = model(xt, mt)
            if kind == 'mae':
                prediction = prediction[0].reshape_as(xt)
            loss = normalized_masked_loss(prediction, xt, mt, scale)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite {kind} loss at update {step}')
            (loss/cfg['accumulation']).backward()
            loss_value += float(loss.detach())/cfg['accumulation']
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        if not torch.isfinite(norm):
            raise RuntimeError(f'Nonfinite {kind} gradients at update {step}')
        optimizer.step()
        sync(device)
        end = time.perf_counter()
        timings.append(dict(update=step, training_seconds=end-mark, compute_seconds=end-batch_ready,
                            batch_prepare_seconds=batch_ready-mark, lr=float(lr), loss=loss_value))
        update = step
        if step % cfg['eval_every'] == 0 or step == limit:
            snapshot(step, loss_value)
            print(f'{kind} update={step}/{budget} loss={loss_value:.6g} seconds={timings[-1]["training_seconds"]:.3f}', flush=True)
        if step % cfg['checkpoint_every'] == 0 or step == limit:
            checkpoint_started = time.perf_counter()
            atomic_checkpoint(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                update=step, rng=rng_state(), contract=contract, next_batch=batch_for(data, cfg, kind, step+1, fit_bank, fit_masks)[2],
                history=history, timings=timings, sequence=sequence))
            pd.DataFrame(timings).to_csv(folder / 'updates.csv', index=False)
            checkpoint_seconds += time.perf_counter()-checkpoint_started
    resources = dict(update=update, budget=budget, complete=update >= budget,
        session_seconds=time.perf_counter()-started, session_evaluation_seconds=evaluation_seconds,
        session_checkpoint_seconds=checkpoint_seconds,
        training_seconds=sum(t['training_seconds'] for t in timings),
        mean_seconds_per_update=float(np.mean([t['training_seconds'] for t in timings])) if timings else None,
        peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        train_sequence_sha256=digest(sequence), fixed_train_validation_banks=True)
    write_json(folder / 'resources.json', resources)
    return model, resources
