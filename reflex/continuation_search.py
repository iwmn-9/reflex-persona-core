"""Bounded shared schedules, evaluated across all coherent model branches.

One schedule is shared by the seeds. An unavailable action falls back to the
same persona reflex using that branch's observation. This is open-loop proposal
search with feedback fallback, not a branch-clairvoyant policy or optimal tree.
The real caller executes only the selected root and replans next observation.
"""
from dataclasses import replace
import copy
import random
from .core import Policy,compile_batch,MAX_ACTIONS,digest
from .flow_rollout import rollout
from .purpose_plan import goal_forecast


def _forecast(c,nodes,*,horizon,unit,target,max_regret,policy):
    return goal_forecast(c,{k:n['paths'] for k,n in nodes.items()},
        {k:n['audit'] for k,n in nodes.items()},horizon=horizon,unit=unit,
        target=target,max_regret=max_regret,plan_roots={k:n['root'] for k,n in nodes.items()},policy=policy)


def _rank(c,nodes,width,horizon,unit,target,max_regret,policy):
    indexed={f'candidate-{j:03d}':n for j,n in enumerate(nodes)}
    f=_forecast(c,indexed,horizon=horizon,unit=unit,target=target,max_regret=max_regret,policy=policy)
    b=compile_batch(f.contexts);d=policy.decide(b,False);best=max(f.purpose.values());rank=[]
    for j,key in enumerate(b.ids[0]):
        proposal=f.audit['plans'][key]['proposal']
        rank.append((not(f.purpose[key]>=best-max_regret-1e-12),-float(d.scores[0,j]),proposal))
    return [indexed[p] for _,_,p in sorted(rank)[:width]]


def search_forecast(c,initial,*,observe,advance,terminal,assess,horizon,seeds=(0,),
                    width=2,depth=2,unit='public-turns',target,max_regret=.15,observer=None,retained=(),policy=None,samples=0):
    """Keep a reflex baseline and <=width searched continuations per root.

    At each future step, expand the union of publicly legal observed action
    identifiers; apply the SAME schedule to EVERY seed before ranking. Beam
    ranking reuses endpoint competence, full-forecast principle tier and Policy
    scores, with no game-specific priorities or additional reward coefficients.
    Prefixes are scored with a reflex completion to the full horizon, so pruning
    can miss a delayed payoff. width/depth control that approximation explicitly.
    """
    compile_batch([c]);policy=policy or Policy()
    if type(horizon) is not int or not 1<=horizon<=16:raise ValueError('bounded horizon required')
    if type(width) is not int or not 1<=width<=4 or type(depth) is not int or not 0<=depth<horizon:
        raise ValueError('bounded search width/depth required')
    if type(samples) is not int or not 0<=samples<=64:raise ValueError('bounded schedule samples required')
    roots=sorted(a['id'] for a in c['actions'] if a['legal'] and not a['known_failure'])
    if not isinstance(retained,(tuple,list)) or len(retained)>horizon or any(not isinstance(k,str) for k in retained):
        raise ValueError('bounded retained action sequence required')
    if len(roots)*(width+1)+bool(retained)>MAX_ACTIONS:raise ValueError('proposal capacity exceeded; roots cannot be dropped')
    kept={};evaluations=0;layer_counts=[];sampling=[];pilot_evaluations=0
    for ri,root in enumerate(roots):
        cache={}
        def evaluate(schedule):
            nonlocal evaluations
            if schedule not in cache:
                paths,audit=rollout(c,initial,observe=observe,advance=advance,terminal=terminal,
                    assess=assess,horizon=horizon,seeds=seeds,observer=observer,
                    roots=(root,),schedule=schedule,record_choices=True,policy=policy)
                cache[schedule]=dict(root=root,schedule=list(schedule),paths=paths[root],audit=audit[root])
                evaluations+=1
            return cache[schedule]
        baseline=evaluate(());beam=[baseline]
        local=copy.deepcopy(c);local['actions']=[a for a in local['actions'] if a['id']==root]
        for level in range(depth):
            candidates={}
            for node in beam:
                choices=sorted({k for trace in node['audit'] if len(trace['choices'])>level+1 for k in trace['choices'][level+1]})
                if not choices:
                    candidates[tuple(node['schedule'])]=node
                for key in choices:
                    schedule=tuple(node['schedule'])+(key,)
                    candidates[schedule]=evaluate(schedule)
            beam=_rank(local,candidates.values(),width,horizon,unit,target,max_regret,policy)
            layer_counts.append(dict(root=root,depth=level+1,candidates=len(candidates),retained=len(beam)))
        # Random shooting completes a whole legal pilot sequence before ranking.
        # A pilot may use one MODEL branch to discover legal future actions; its
        # frozen sequence is then evaluated on EVERY model branch, like the beam.
        # Nothing selects a different sequence after seeing the real outcome.
        sampled={}
        for trial in range(samples):
            rng=random.Random(int(digest([c['scope'],c['seed'],c['tick'],root,'schedule-pilot',trial])[:16],16))
            _,pilot=rollout(c,initial,observe=observe,advance=advance,terminal=terminal,assess=assess,
                horizon=horizon,seeds=(seeds[trial%len(seeds)],),observer=observer,roots=(root,),
                record_choices=True,policy=policy,schedule_sampler=lambda keys,t:rng.choice(keys))
            pilot_evaluations+=1
            schedule=tuple(pilot[root][0]['actions'][1:]);sampled[schedule]=evaluate(schedule)
        if sampled:
            # Rank the full union with the same persona/purpose evaluator. The
            # baseline remains a separate proposal, even when search discards it.
            pool={tuple(n['schedule']):n for n in beam};pool.update(sampled)
            beam=_rank(local,pool.values(),width,horizon,unit,target,max_regret,policy)
        if samples:sampling.append(dict(root=root,trials=samples,unique_schedules=len(sampled)))
        unique={tuple(n['schedule']):n for n in [baseline]+beam}
        for j,n in enumerate(unique.values()):kept[f'proposal-{ri:03d}-{j:02d}']=n
    retained_status=None
    if retained:
        retained_status='root no longer viable'
        if retained[0] in roots:
            paths,audit=rollout(c,initial,observe=observe,advance=advance,terminal=terminal,assess=assess,
                horizon=horizon,seeds=seeds,observer=observer,roots=(retained[0],),schedule=retained[1:],record_choices=True,policy=policy)
            kept['proposal-retained']=dict(root=retained[0],schedule=list(retained[1:]),paths=paths[retained[0]],audit=audit[retained[0]])
            evaluations+=1;retained_status='reevaluated'
    f=_forecast(c,kept,horizon=horizon,unit=unit,target=target,max_regret=max_regret,policy=policy)
    metadata=copy.deepcopy(f.audit)
    metadata['continuation_search']=dict(width=width,depth=depth,evaluated=evaluations,layers=layer_counts,
        proposals={k:dict(root=n['root'],schedule=n['schedule'],branches=n['audit']) for k,n in kept.items()},
        semantics='one shared action schedule across seeds; observed-illegal actions use persona reflex; only root executed')
    if samples:metadata['continuation_search'].update(samples=samples,pilot_evaluations=pilot_evaluations,sampling=sampling,
        sampling_semantics='uniform legal full-horizon model pilots; freeze sequence then evaluate across all model seeds; no learned dynamics or actual future input')
    preference=None
    if retained:
        from .plan_continuity import ArrivalPreference
        incumbent=next((k for k in f.roots if f.audit['plans'][k].get('proposal')=='proposal-retained'),None)
        if incumbent is not None:
            preference=ArrivalPreference(incumbent,{k:tuple(g['settlement_step'] if g['status']=='success' else None
                for g in f.audit['endpoint'][k]) for k in f.roots})
        metadata['continuation_search']['retained_status']=retained_status if incumbent is not None else 'not in current persona tier or not viable'
    return replace(f,audit=metadata,continuity=preference)
