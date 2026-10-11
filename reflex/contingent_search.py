"""Persona-ranked observation-history policy patches across common scenarios.

Finite sparse policy search, not POMCP/DESPOT or an optimality guarantee.
No scenario ID, world state, game name or real future is a policy node key.
"""
from dataclasses import replace
import copy
from .core import Policy,compile_batch,digest,MAX_ACTIONS,MAX_OUTCOMES
from .flow_rollout import rollout
from .continuation_search import _rank
from .purpose_plan import goal_forecast
from .observation_plan import ObservationPlan,View,owner,history


def search_forecast(c,initial,*,observe,advance,terminal,assess,horizon,seeds=(0,),
                    width=2,revisions=2,unit='public-turns',target,max_regret=.15,
                    observer=None,policy=None,validation_seeds=None,
                    settlement_weight='discounted',retained=None,prefix='',patch_budget=16):
    compile_batch([c]);history(prefix);policy=policy or Policy()
    if type(width) is not int or not 1<=width<=4 or type(revisions) is not int or not 0<=revisions<=8:
        raise ValueError('bounded conditional width/revisions required')
    if type(patch_budget) is not int or not 1<=patch_budget<=32:
        raise ValueError('bounded public patch budget required')
    if retained is not None and (not isinstance(retained,ObservationPlan) or
        (retained.owner,retained.target,retained.unit)!=(owner(c),target,unit)):
        raise ValueError('same owner, purpose and unit retained policy required')
    if validation_seeds is not None and (not isinstance(validation_seeds,(tuple,list)) or
        not 1<=len(validation_seeds)<=MAX_OUTCOMES or any(type(s) is not int for s in validation_seeds) or
        len(set(validation_seeds))!=len(validation_seeds) or set(seeds)&set(validation_seeds)):
        raise ValueError('independent validation scenario bank required')
    roots=sorted(a['id'] for a in c['actions'] if a['legal'] and not a['known_failure'])
    # Every viable current action keeps its reflex completion. Extra programs
    # share the remaining numeric bank, round-robin in a public identity hash
    # order. A large legal set reduces search, never removes legal roots.
    slots={r:0 for r in roots};available=MAX_ACTIONS-len(roots)
    order=sorted(roots,key=lambda r:digest([owner(c),c['tick'],r,'extra-slot']))
    for level in range(width):
        for r in order:
            if available:slots[r]+=1;available-=1
    empty=ObservationPlan.make(c,target,unit);kept={};evaluations=0;layers=[]
    def evaluate(root,program,bank):
        nonlocal evaluations
        paths,audit=rollout(c,initial,observe=observe,advance=advance,terminal=terminal,
            assess=assess,horizon=horizon,seeds=bank,roots=(root,),observer=observer,policy=policy,
            public_controller=program.choose,public_history=prefix,record_choices=True)
        evaluations+=1
        return dict(root=root,program=program,paths=paths[root],audit=audit[root])
    for ri,root in enumerate(roots):
        cache={}
        def candidate(program):
            key=program.entries
            if key not in cache:cache[key]=evaluate(root,program,seeds)
            return cache[key]
        baseline=candidate(empty);beam=[baseline]
        if retained is not None:beam.append(candidate(retained))
        local=copy.deepcopy(c);local['actions']=[a for a in local['actions'] if a['id']==root]
        beam=_rank(local,beam,width,horizon,unit,target,max_regret,policy,settlement_weight)
        for level in range(revisions if slots[root] else 0):
            expanded={n['program'].entries:n for n in beam}
            for n in beam:
                patches={}
                for trace in n['audit']:
                    for node in trace['public_nodes'][1:]:
                        for key in node['allowed']:
                            if key==node['action']:continue
                            patches[(node['node'],key)]=node['depth']
                # Depth round-robin makes late observations eligible too.
                # This is bounded proposal discovery, not exhaustive planning.
                groups={d:[] for d in sorted(set(patches.values()))}
                for pair,d in patches.items():groups[d].append(pair)
                for pairs in groups.values():pairs.sort(key=lambda p:digest([owner(c),c['tick'],level,p]))
                picked=[]
                while len(picked)<patch_budget and any(groups.values()):
                    for pairs in groups.values():
                        if pairs and len(picked)<patch_budget:picked.append(pairs.pop(0))
                for node,key in picked:
                    # Drop entries unreachable in this candidate's modeled
                    # histories. Unknown actual histories still use reflex.
                    visited={p['node'] for t in n['audit'] for p in t['public_nodes'][1:]}
                    compact=ObservationPlan.make(c,target,unit,{p:k for p,k in n['program'].entries if p in visited})
                    program=compact.patch(node,key)
                    expanded[program.entries]=candidate(program)
            beam=_rank(local,expanded.values(),width,horizon,unit,target,max_regret,policy,settlement_weight)
            layers.append(dict(root=root,revision=level+1,candidates=len(expanded),retained=len(beam)))
        extras=[n for n in beam if n['program'].entries!=baseline['program'].entries][:slots[root]]
        unique={n['program'].entries:n for n in [baseline]+extras}
        for j,n in enumerate(unique.values()):kept[f'conditional-{ri:03d}-{j:02d}']=n
    if retained is not None and len(kept)<MAX_ACTIONS:
        view=View.make(prefix,c)
        root=retained.choose(view) or policy.choose(c,stochastic=False)['action_id']
        # Programs normally omit a forced current root. Re-evaluate the retained
        # continuation under every viable root above; no stale root is forced.
        if root in roots:kept['conditional-retained']=evaluate(root,retained,seeds)
    validation=None
    if validation_seeds is not None:
        for k,n in list(kept.items()):kept[k]=evaluate(n['root'],n['program'],validation_seeds)
        validation=dict(seeds=list(validation_seeds),discovery_seeds=list(seeds),
            semantics='frozen policy re-evaluation; unknown observations use persona reflex; not calibrated model confidence')
    f=goal_forecast(c,{k:n['paths'] for k,n in kept.items()},
        {k:n['audit'] for k,n in kept.items()},horizon=horizon,unit=unit,target=target,
        max_regret=max_regret,plan_roots={k:n['root'] for k,n in kept.items()},policy=policy,
        settlement_weight=settlement_weight)
    metadata=copy.deepcopy(f.audit)
    metadata['contingent_search']=dict(width=width,revisions=revisions,patch_budget=patch_budget,
        extra_slots=slots,all_viable_roots=roots,evaluated=evaluations,layers=layers,
        prefix=prefix,root_node=next(iter(kept.values()))['audit'][0]['public_nodes'][0]['node'],unit=unit,
        proposals={k:dict(root=n['root'],program=n['program'].record(),branches=n['audit']) for k,n in kept.items()},
        semantics='one immutable observation-history program across all model scenarios; unknown nodes use owner reflex; execute only current root')
    if validation is not None:metadata['contingent_search']['validation']=validation
    return replace(f,audit=metadata)
