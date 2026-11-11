from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import torch

from experiments.wcb.training import device_for, initialize, capture, restore, save_checkpoint, train_block
from .common import RATE_ORDER, ADAPTIVE, TWO_BRANCH, digest, read_json, write_json, seed_for
from .data import Prepared
from .evaluation import evaluate_bank, linear_probe
from .masks import matched_masks, rates_from_logits, limited_update


def policy_candidates(method, x, c, block, seed, replay=None):
    rng = np.random.default_rng(seed_for(seed, 'policy', block))
    direction = np.zeros(3)
    noise = None
    if method == 'coordinate':
        direction[block % 3] = 1
    elif method == 'spsa':
        direction = rng.choice([-1., 1.], 3)
    if method in ('coordinate', 'spsa'):
        rates = [rates_from_logits(x + c['delta'] * direction, c),
                 rates_from_logits(x - c['delta'] * direction, c)]
    elif method == 'reinforce':
        noise = rng.normal(size=3)
        rates = [rates_from_logits(x + c['sigma'] * noise, c),
                 rates_from_logits(x - c['sigma'] * noise, c)]
    elif method == 'conditional_best2':
        rates = [np.full(3, .5), np.full(3, .5)]
    elif method == 'random_policy':
        rates = [rng.uniform(c['rate_low'], c['rate_high'], 3)]
    elif method == 'replay':
        rates = [np.array(replay[block]['committed_rates'], float)]
    else:
        rates = [np.array(c['fixed_rates'], float)]
    return rates, direction, noise


def reward_scale(history, initial_loss, c):
    floor = max(1e-12, abs(initial_loss) * c['normalization_floor'])
    if len(history) < c['normalization_min_history']:
        return max(abs(initial_loss), floor), 'initial_probe_loss'
    low, high = np.percentile(history, [20, 80])
    return max(float(high - low), floor), 'historical_p80_minus_p20'


def policy_gradient(method, rewards, direction, noise, c):
    if method in ('coordinate', 'spsa'):
        return (rewards[0] - rewards[1]) / (2 * c['delta']) * direction
    if method == 'reinforce':
        # Mean Normal-policy score gradient, antithetic actions, pair-mean baseline.
        return (rewards[0] - rewards[1]) / (2 * c['sigma']) * noise
    return np.zeros(3)


def block_data(prepared, seed, step, h, rates, method):
    c = prepared.c
    fit = np.asarray(prepared.selection['fit'])
    batches, all_masks, stats = [], [[] for _ in rates], [[] for _ in rates]
    kind = ('global' if method == 'random' else 'conditional_best2' if method == 'conditional_best2'
            else 'conditional' if method == 'conditional' else 'structured')
    for offset in range(h):
        absolute = step + offset + 1
        rng = np.random.default_rng(seed_for(seed, 'data', absolute))
        indices = rng.choice(fit, c['microbatch'] * c['accumulation'], replace=True)
        masks = [[] for _ in rates]
        for j, _ in enumerate(indices):
            key = seed_for(seed, 'mask', absolute, j)
            for branch, (mask, info) in enumerate(matched_masks(c, rates, key, kind=kind)):
                masks[branch].append(mask)
                stats[branch].append(info)
        batches.append(indices)
        for branch in range(len(rates)):
            all_masks[branch].append(np.asarray(masks[branch]))
    return batches, all_masks, stats


def summarize_masks(stats):
    return dict(inputs=len(stats), modes=sorted({r['mode'] for r in stats}),
                realized={key: float(np.mean([r['realized'][key] for r in stats])) for key in (*RATE_ORDER, 'background')},
                candidates=stats[0]['candidates'],
                target_hidden_mean=float(np.mean([r['target_hidden'] for r in stats])),
                target_tokens=stats[0]['candidates']['target'])


def run(study, plan, method, seed, *, stop_after=None, replay=None, folder=None):
    c = plan['config']
    folder = Path(folder) if folder else Path(study) / 'runs' / method / f'seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    prepared = Prepared(study, plan)
    contract = dict(plan_id=plan['plan_id'], method=method, seed=seed,
                    replay=digest(replay) if replay is not None else None,
                    prepared_id=read_json(prepared.root / 'completed.json')['prepared_id'])
    if (folder / 'completed.json').exists():
        result = read_json(folder / 'completed.json')
        if result['contract'] != contract:
            raise ValueError('Completed run contract mismatch')
        return result
    if stop_after is not None and stop_after < 1:
        raise ValueError('stop_after must be positive')
    device = device_for(c)
    model = initialize(c, seed).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c['lr'],
                                 betas=(.9, .95), weight_decay=c['weight_decay'])
    x, step, block, seconds, attempted = np.zeros(3), 0, 0, 0., 0
    history, reward_history = [], []
    initial_loss = None
    checkpoint = folder / 'checkpoint.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['contract'] != contract:
            raise ValueError('Resume contract mismatch')
        restore(state['training'], model, optimizer)
        x = np.array(state['x'])
        step, block, seconds, attempted = (state[k] for k in ('step', 'block', 'seconds', 'attempted'))
        history, reward_history, initial_loss = (state[k] for k in ('history', 'reward_history', 'initial_loss'))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    evaluate = lambda name, split='control': evaluate_bank(
        model, prepared.x if split == 'control' else prepared.val,
        prepared.root / f'{name}_{split}.npz', device, c['microbatch'])
    started = time.perf_counter()
    before = evaluate('balanced')
    initial_loss = before['loss'] if initial_loss is None else initial_loss
    initial_random = evaluate('random') if step == 0 else None
    seconds += time.perf_counter() - started
    limit = min(c['updates'], step + stop_after) if stop_after else c['updates']
    if replay is not None and sum(r['H'] for r in replay) != c['updates']:
        raise ValueError('Replay must cover the original complete update horizon')
    write_json(folder / 'status.json', dict(status='running', contract=contract, step=step))
    # A time budget stops at a checkpoint boundary, including at least one block.
    while step < limit and not (step > 0 and c['budget_seconds'] and seconds >= c['budget_seconds']):
        started = time.perf_counter()
        h = min(c['H'], c['updates'] - step)
        if replay is not None and replay[block]['H'] != h:
            raise ValueError('Replay block duration mismatch')
        rates, direction, noise = policy_candidates(method, x, c, block, seed, replay)
        ids, masks, stats = block_data(prepared, seed, step, h, rates, method)
        start_state = capture(model, optimizer) if len(rates) == 2 else None
        outcomes, train_losses, plus = [], [], None
        for branch in range(len(rates)):
            if branch:
                restore(start_state, model, optimizer)
            train_losses.append(train_block(model, optimizer, prepared, ids, masks[branch], step, device))
            outcomes.append(evaluate('balanced'))
            attempted += h
            if branch == 0 and len(rates) == 2:
                plus = capture(model, optimizer)
        raw_rewards = np.array([before['loss'] - outcome['loss'] for outcome in outcomes])
        scale, scale_kind = reward_scale(reward_history, initial_loss, c)
        normalized = raw_rewards / scale
        chosen = int(np.argmin([r['loss'] for r in outcomes]))
        if len(rates) == 2 and outcomes[0]['loss'] == outcomes[1]['loss']:
            chosen = int(seed_for(seed, block, 'tie') % 2)
        if plus is not None and chosen == 0:
            restore(plus, model, optimizer)
        old = x.copy()
        gradient = policy_gradient(method, normalized, direction, noise, c)
        if not np.isfinite(gradient).all():
            raise RuntimeError('Nonfinite policy gradient')
        if method in ADAPTIVE:
            x = limited_update(x, x + c['policy_lr'] * gradient, c)
        # Both candidates use the SAME past-only scale; update history afterwards.
        reward_history.extend(raw_rewards.tolist())
        step += h
        block += 1
        before = outcomes[chosen]
        random = evaluate('random') if block % c['random_every'] == 0 or step >= c['updates'] else None
        record = dict(step=step, block=block, H=h, rate_order=list(RATE_ORDER),
                      center_logits_before=old.tolist(), center_logits_after=x.tolist(),
                      center_before=rates_from_logits(old, c).tolist(), center_after=rates_from_logits(x, c).tolist(),
                      candidate_rates=[r.tolist() for r in rates], committed_rates=rates[chosen].tolist(),
                      chosen=chosen, raw_rewards=raw_rewards.tolist(), normalized_rewards=normalized.tolist(),
                      reward_scale=scale, reward_scale_kind=scale_kind, gradient=gradient.tolist(),
                      direction=direction.tolist(), antithetic_noise=None if noise is None else noise.tolist(),
                      reward_kind='context_balanced_all_hidden_lp', context_before_loss=before['loss'] + float(raw_rewards[chosen]),
                      context_balanced=before, candidate_context_balanced=outcomes,
                      random_control=random, random_initial=initial_random if block == 1 else None,
                      train_losses=train_losses, masks=[summarize_masks(s) for s in stats],
                      mask_hashes=[digest([np.packbits(m).tobytes().hex() for m in branch]) for branch in masks])
        if c['probe_every'] and step % c['probe_every'] == 0:
            record['linear_probe'] = linear_probe(model, prepared, folder / f'probe_{step}', seed, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds += time.perf_counter() - started
        record['elapsed_training_pipeline_seconds'] = seconds
        history.append(record)
        state = dict(contract=contract, training=capture(model, optimizer), x=x.tolist(), step=step,
                     block=block, seconds=seconds, attempted=attempted, history=history,
                     reward_history=reward_history, initial_loss=initial_loss)
        save_checkpoint(checkpoint, state)
        write_json(folder / 'history.json', history)
        write_json(folder / 'schedule.json', [dict(step=r['step'], H=r['H'], committed_rates=r['committed_rates']) for r in history])
        write_json(folder / 'status.json', dict(status='running', step=step, attempted_updates=attempted))
        center = dict(zip(RATE_ORDER, rates_from_logits(x, c).round(5).tolist()))
        print(f"{method} seed={seed} step={step} context_balanced={before['loss']:.6g} center={center}", flush=True)
    finished = step >= c['updates'] or bool(c['budget_seconds'] and seconds >= c['budget_seconds'])
    if not finished:
        write_json(folder / 'status.json', dict(status='stopped_at_boundary', step=step))
        return dict(complete=False, step=step)
    mark = time.perf_counter()
    random_control, balanced_control = evaluate('random'), evaluate('balanced')
    random_validation = evaluate('random', 'validation')
    balanced_validation = evaluate('balanced', 'validation')
    linear = linear_probe(model, prepared, folder, seed, device)
    result = dict(contract=contract, complete=True, step=step, attempted_updates=attempted,
                  stop_reason='updates' if step >= c['updates'] else 'compute_budget',
                  rate_order=list(RATE_ORDER), random_control=random_control, context_balanced_control=balanced_control,
                  random_validation=random_validation, context_balanced_validation=balanced_validation,
                  linear_probe=linear, training_pipeline_seconds=seconds,
                  final_evaluation_seconds=time.perf_counter() - mark,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                  annotations_used=False, from_scratch=True, synthetic=c['synthetic'], test_opened=False)
    write_json(folder / 'completed.json', result)
    write_json(folder / 'status.json', dict(status='completed', step=step))
    return result
