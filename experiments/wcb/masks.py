"""One final binary mask per ECG. True=hidden. Exact 50% globally."""
from __future__ import annotations
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

def wave_groups(result,p):
    groups=[]
    leads=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    q=2500//p
    for r in result['intervals']:
        if r['valid']:
            if not 0<=r['onset']<r['offset']<=2500: raise ValueError('Invalid accepted wave bounds')
            ids=np.arange(r['onset']//p,(r['offset']+p-1)//p,dtype=int)+leads.index(r['lead'])*q
            groups.append(dict(lead=leads.index(r['lead']),phase=r['phase'],beat=r['beat'],ids=ids))
    return groups

def global_mask(n,rng):
    mask=np.ones(n,bool); mask[rng.permutation(n)[:n//2]]=False
    return mask

def choose_targets(groups,k,rng):
    if not groups: return []
    phases=np.roll(['P','QRS','T'],int(rng.integers(3))).tolist()
    order=rng.permutation(len(groups)).tolist(); chosen=[]; leads=set()
    for i in range(k):
        phase=phases[i%3]
        choices=[j for j in order if groups[j]['phase']==phase and groups[j]['lead'] not in leads]
        if not choices: choices=[j for j in order if groups[j]['phase']==phase]
        if not choices: choices=order[:]
        if not choices: break
        j=choices[0]; chosen.append(groups[j]); leads.add(groups[j]['lead']); order.remove(j)
    return chosen

def context_sets(target,groups,q,all_hidden=None):
    t=set(map(int,target['ids'])); forbidden=t if all_hidden is None else set(all_hidden)
    b=set()
    for g in groups:
        if g['lead']==target['lead'] and g['phase']==target['phase'] and g['beat']!=target['beat']:
            ids=set(map(int,g['ids']))
            if not ids & forbidden: b.update(ids)
    c={lead*q+int(j%q) for j in t for lead in range(12) if lead!=target['lead']}-forbidden
    w=set(range(target['lead']*q,(target['lead']+1)*q))-forbidden-b
    return tuple(np.array(sorted(s),dtype=int) for s in (w,c,b))

def _solve(n,targets,groups,q,rates,priority,rounding):
    hidden=set().union(*(set(map(int,t['ids'])) for t in targets))
    available=np.array([j for j in range(n) if j not in hidden],dtype=int)
    if len(available)<n//2: return None
    contexts=[context_sets(t,groups,q,hidden) for t in targets]
    sets=[s for triple in contexts for s in triple]
    counts=[]
    for i,s in enumerate(sets):
        expected=float(rates[i%3])*len(s)
        counts.append(int(np.floor(expected)+(rounding[i]<expected%1)))
    if len(targets)==1:
        visible=[]
        for s,count in zip(sets,counts): visible.extend(s[np.argsort(priority[s],kind='stable')[:count]])
        used=set().union(*(set(map(int,s)) for s in sets),hidden)
        u=np.array([j for j in range(n) if j not in used],dtype=int)
        missing=n//2-len(visible)
        if not 0<=missing<=len(u): return None
        visible.extend(u[np.argsort(priority[u],kind='stable')[:missing]])
    else:
        # Compress tokens by identical constraint memberships before integer allocation.
        membership=np.zeros((len(available),len(sets)),dtype=np.uint8)
        for i,s in enumerate(sets): membership[:,i]=np.isin(available,s)
        patterns,inverse=np.unique(membership,axis=0,return_inverse=True)
        cells=[available[inverse==i] for i in range(len(patterns))]
        sizes=np.array([len(x) for x in cells])
        matrix=np.vstack([np.ones(len(cells)),patterns.T])
        rhs=np.array([n//2,*counts],float)
        result=milp(c=np.array([priority[x].mean() for x in cells]),integrality=np.ones(len(cells)),
            bounds=Bounds(np.zeros(len(cells)),sizes),constraints=LinearConstraint(matrix,rhs,rhs),
            options={'presolve':True})
        if result.x is None: return None
        allocations=np.rint(result.x).astype(int)
        if not np.array_equal(matrix@allocations,rhs): return None
        visible=[j for cell,count in zip(cells,allocations)
                 for j in cell[np.argsort(priority[cell],kind='stable')[:count]]]
    mask=np.ones(n,bool); mask[np.asarray(visible,dtype=int)]=False
    if mask.sum()!=n//2 or any(not mask[j] for j in hidden): raise AssertionError('Mask invariant')
    numer=np.zeros(3); denom=np.zeros(3)
    for i,s in enumerate(sets): numer[i%3]+=(~mask[s]).sum(); denom[i%3]+=len(s)
    union=set().union(*(set(map(int,s)) for s in sets),hidden)
    u=np.array([j for j in range(n) if j not in union],int)
    realized=np.divide(numer,denom,out=np.full(3,np.nan),where=denom>0)
    return mask,dict(targets=len(targets),target_tokens=len(hidden),
        realized=[None if not np.isfinite(x) else float(x) for x in realized],
        candidate_counts=denom.tolist(),U_candidates=len(u),U_visible=int((~mask[u]).sum()))

def matched_masks(groups,q,rates_list,seed,k=6,kind='structured',structured_fraction=.8):
    """Trial policies share targets, priorities, rounding and feasibility fallback."""
    rng=np.random.default_rng(seed); n=12*q
    priority=rng.random(n); targets=choose_targets(groups,k,rng)
    rounding=rng.random(3*max(1,len(targets)))
    if kind=='global' or not targets or (kind in ('structured','conditional') and rng.random()>structured_fraction):
        m=np.ones(n,bool); m[np.argsort(priority)[:n//2]]=False
        return [(m.copy(),dict(targets=0,target_tokens=0,mode='random',fallback=False,
                              no_annotations=not bool(groups) and kind!='global')) for _ in rates_list]
    if kind=='conditional':
        hidden=set().union(*(set(map(int,t['ids'])) for t in targets))
        available=np.array([j for j in range(n) if j not in hidden])
        if len(available)<n//2: targets=targets[:1]; hidden=set(map(int,targets[0]['ids'])); available=np.array([j for j in range(n) if j not in hidden])
        m=np.ones(n,bool); m[available[np.argsort(priority[available])[:n//2]]]=False
        return [(m.copy(),dict(targets=len(targets),target_tokens=len(hidden),mode='conditional',fallback=False)) for _ in rates_list]
    sizes=sorted(set([len(targets),min(3,len(targets)),1]),reverse=True)
    for size in sizes:
        answers=[_solve(n,targets[:size],groups,q,r,priority,rounding) for r in rates_list]
        if all(a is not None for a in answers):
            for _,stats in answers: stats.update(mode='structured',fallback=size<len(targets))
            return answers
    raise RuntimeError('Even single-target allocation is infeasible; inspect wave duration and candidates')

def sigmoid(x): return 1/(1+np.exp(-np.clip(np.asarray(x),-30,30)))

def limited_update(old_x,proposed_x,limit):
    old=sigmoid(old_x)
    r=np.clip(sigmoid(proposed_x),np.maximum(1e-6,old-limit),np.minimum(1-1e-6,old+limit))
    return np.log(r/(1-r))

def curriculum(progress):
    knots=np.array([[.5,.5,.5],[.8,.2,.5],[.5,.8,.2],[.2,.5,.8],[.5,.5,.5]])
    return np.array([np.interp(progress,np.linspace(0,1,5),knots[:,j]) for j in range(3)])
