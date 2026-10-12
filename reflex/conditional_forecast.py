"""Compare capability continuation with future owner decisions in two adapters.

Root banks, purpose scales and concessions are unchanged. Future exchange uses
a bounded one-turn controller, not recursive invocation of the outer six-turn
search. Unsigned actors keep the old declared finite hypotheses.
"""
import copy
import numpy as np
from .core import digest
from .examples import action
from .deliberation import JointForecast
from .owner_continuation import OwnerBranch
from .team_choice import select


def forecast(genre,w,actors,contexts,decisions,*,continuation='personal',horizon=6,inner_plans=8,policy=None):
    if genre not in ('combat','projects') or continuation not in ('personal','exchange'):
        raise ValueError('declared adapter and continuation required')
    if type(horizon) is not int or not 1<=horizon<=16 or type(inner_plans) is not int or not 1<=inner_plans<=64:
        raise ValueError('bounded future horizon and exchange bank required')
    from . import team_combat as tc,team_projects as tp
    from .combat import alive,terminal,resolve,transition_effect
    from .combat_planning import opponent_intents,mission
    rules=tc if genre=='combat' else tp;team=(w.units if genre=='combat' else w.people)[actors[0]].team
    factory=tc.make_person if genre=='combat' else tp.make_context
    # Generate exactly the old current root bank, without evaluating its old
    # six-turn continuation. First-turn candidate generation does not depend
    # on the rollout policy.
    bank=rules.forecast(w,actors,contexts,decisions,horizon=1,partner_model='uncertain')
    future=[copy.deepcopy(c) for c in contexts]
    for c in future:c['actions']=[]
    roots=bank.roots;purpose={};nodes=bank.audit['nodes'];inner_nodes=0;future_steps=0;accepted=0;branches=[]
    unsigned=tuple(i for i in (alive(w,team) if genre=='combat' else tp.members(w,team)) if i not in actors)
    samples=4 if genre=='combat' else 2 if unsigned else 1
    def exchange(model,live,cs,ds):
        nonlocal inner_nodes,accepted
        f=rules.forecast(model,live,cs,ds,horizon=1,max_plans=inner_plans,partner_model='uncertain')
        inner_nodes+=f.audit['nodes']
        result,audit=select(cs,f,group=f'future-{genre}-{team}',outside=tuple(d['action_id'] for d in ds),policy=policy)
        accepted+=result is not None
        return result,audit
    for key,root in roots.items():
        outcomes=[[] for _ in actors];scores=[]
        for sample in range(samples):
            model=w;branch=OwnerBranch(actors,contexts,decisions,root,policy=policy)
            model_seed=int(digest(['signed-combat-model',contexts[0]['scope']['episode'],w.tick,sample])[:16],16)
            for depth in range(horizon):
                if terminal(model) if genre=='combat' else tp.terminal(model):break
                live=tuple(i for i in actors if i in alive(model,team)) if genre=='combat' else actors
                if depth==0:ours=dict(zip(actors,root))
                elif live:
                    ours,_=branch.choose(live,model,factory,exchange if continuation=='exchange' else None)
                else:ours={}
                for i in unsigned:
                    if genre=='combat' and i not in alive(model,team):continue
                    kind=sample//2%2 if genre=='combat' else sample
                    ours[i]=rules.outside_action(model,i,kind)
                theirs=opponent_intents(model,team,aggressive=sample%2==1) if genre=='combat' else tp.greedy(model,1-team)
                model,_=resolve(model,{**ours,**theirs},model_seed) if genre=='combat' else tp.resolve(model,{**ours,**theirs})
                nodes+=1
            value=mission(model,team) if genre=='combat' else tp.purpose(model,team);scores.append(value)
            for j,actor in enumerate(actors):
                e=(transition_effect(w,model,actor,root[j],contexts[j]['facts']['chosen_route'],survival_security=True,p=1/samples)
                    if genre=='combat' else tp.consequence(w,model,actor,root[j],p=1/samples))
                e['objective']=value;e['values']['achievement']=value;outcomes[j].append(e)
            future_steps+=branch.steps;branches.append(branch.receipt())
        purpose[key]=float(np.clip(np.mean(scores),-1,1))
        for c,es in zip(future,outcomes):c['actions'].append(action(key,*es,confidence=.75 if genre=='combat' else .8))
    return JointForecast(tuple(future),roots,purpose,horizon,.15,
        dict(nodes=nodes+inner_nodes,world_nodes=nodes,inner_nodes=inner_nodes,plans=len(roots),
            unsigned=list(unsigned),samples=samples,continuation=continuation,future_owner_steps=future_steps,
            future_exchanges_accepted=accepted,inner_plans=inner_plans,branch_receipts_sha256=digest(branches),
            model='same fixed owners, refreshed public needs/routes, detached hypothetical state; bounded one-turn future exchange'),
        target=bank.target)
