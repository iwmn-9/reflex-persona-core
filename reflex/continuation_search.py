"""Bounded shared schedules, evaluated across all coherent model branches.

One schedule is shared by the seeds. An unavailable action falls back to the
same persona reflex using that branch's observation. This is open-loop proposal
search with feedback fallback, not a branch-clairvoyant policy or optimal tree.
The real caller executes only the selected root and replans next observation.
"""
from dataclasses import replace
import copy
from .core import Policy,compile_batch,MAX_ACTIONS
from .flow_rollout import rollout
from .purpose_plan import goal_forecast


def _forecast(c,nodes,*,horizon,unit,target,max_regret):
    return goal_forecast(c,{k:n['paths'] for k,n in nodes.items()},
        {k:n['audit'] for k,n in nodes.items()},horizon=horizon,unit=unit,
        target=target,max_regret=max_regret,plan_roots={k:n['root'] for k,n in nodes.items()})


def search_forecast(c,initial,*,observe,advance,terminal,assess,horizon,seeds=(0,),
                    width=2,depth=2,unit='public-turns',target,max_regret=.15,observer=None):
    """Keep a reflex baseline and <=width searched continuations per root.

    At each future step, expand the union of publicly legal observed action
    identifiers; apply the SAME schedule to EVERY seed before ranking. Beam
    ranking reuses endpoint competence, full-forecast principle tier and Policy
    scores, with no game-specific priorities or additional reward coefficients.
    Prefixes are scored with a reflex completion to the full horizon, so pruning
    can miss a delayed payoff. width/depth control that approximation explicitly.
    """
    compile_batch([c])
    if type(horizon) is not int or not 1<=horizon<=16:raise ValueError('bounded horizon required')
    if type(width) is not int or not 1<=width<=4 or type(depth) is not int or not 0<=depth<horizon:
        raise ValueError('bounded search width/depth required')
    roots=sorted(a['id'] for a in c['actions'] if a['legal'] and not a['known_failure'])
    if len(roots)*(width+1)>MAX_ACTIONS:raise ValueError('proposal capacity exceeded; roots cannot be dropped')
    kept={};evaluations=0;layer_counts=[]
    for ri,root in enumerate(roots):
        cache={}
        def evaluate(schedule):
            nonlocal evaluations
            if schedule not in cache:
                paths,audit=rollout(c,initial,observe=observe,advance=advance,terminal=terminal,
                    assess=assess,horizon=horizon,seeds=seeds,observer=observer,
                    roots=(root,),schedule=schedule,record_choices=True)
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
            indexed={f'candidate-{j:03d}':n for j,n in enumerate(candidates.values())}
            f=_forecast(local,indexed,horizon=horizon,unit=unit,target=target,max_regret=max_regret)
            b=compile_batch(f.contexts);d=Policy().decide(b,False)
            best=max(f.purpose.values());rank=[]
            for j,key in enumerate(b.ids[0]):
                proposal=f.audit['plans'][key]['proposal']
                rank.append((not(f.purpose[key]>=best-max_regret-1e-12),-float(d.scores[0,j]),proposal))
            beam=[indexed[p] for _,_,p in sorted(rank)[:width]]
            layer_counts.append(dict(root=root,depth=level+1,candidates=len(candidates),retained=len(beam)))
        unique={tuple(n['schedule']):n for n in [baseline]+beam}
        for j,n in enumerate(unique.values()):kept[f'proposal-{ri:03d}-{j:02d}']=n
    f=_forecast(c,kept,horizon=horizon,unit=unit,target=target,max_regret=max_regret)
    metadata=copy.deepcopy(f.audit)
    metadata['continuation_search']=dict(width=width,depth=depth,evaluated=evaluations,layers=layer_counts,
        proposals={k:dict(root=n['root'],schedule=n['schedule'],branches=n['audit']) for k,n in kept.items()},
        semantics='one shared action schedule across seeds; observed-illegal actions use persona reflex; only root executed')
    return replace(f,audit=metadata)
