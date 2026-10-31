"""Public-goal retry termination, not the reserved world cohort.

H8 keeps the previously retained unfinished-cycle failure and two longer
seven-step preparation checks. Only retry termination proposals change.
Neither the current purpose-tail twenty executions nor stable/Rust are changed.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from reflex.core import character_key,digest
from reflex.decision_loop import DecisionLoop
from reflex._continuation_portfolio import portfolio_planner
from reflex.model_observer import resolve_feedback
from tools.owned_checkpoint import OwnedCheckpoint,encoded,sha
from tools.run_continuation_portfolio import game_for
from tools.run_owned_value import cumulative_basis
from tools.run_purpose_tail_preparation import audit


def specs():
    original = [dict(id=f'chain-{profile}-{kind}',mechanism='chain',profile=profile,
        length=5,payout=2. if kind=='opportunity' else 14.,turns=128,
        changes=([[73,2.]] if kind=='loss' else [[29,14.]] if kind=='opportunity' else []))
        for profile in (0,2) for kind in ('stable','loss','opportunity')]
    return original+[dict(id='chain-0-long-'+kind,mechanism='chain',profile=0,
        length=7,payout=payout,turns=128,changes=[]) for kind,payout in (('profitable',16.),('bad',7.))]


def run(folder,spec,enabled):
    folder.mkdir(parents=True,exist_ok=True)
    config=dict(spec=spec,retry=True,horizon=8,seeds=[701],retry_until_gain=enabled)
    store=OwnedCheckpoint(folder/'checkpoint.json',('gain-retry',spec['id']),
        'gain' if enabled else 'control',configuration=config,every=1)
    g=game_for(spec);world=g.start();actor=g.actor(g.context(world),learning='segmented')
    axes=character_key(g.context(world));rng=np.random.default_rng(0)
    trace=[];checks=dict(fixed_axes=0,owner_parity=0,restored=0);elapsed=0.
    if store.saved is not None:
        saved=store.saved;world=saved['world'];trace=saved['trace'];checks=saved['checks'];elapsed=saved['elapsed']
        actor=DecisionLoop.from_record(g.context(trace[-1]['before']),saved['owner'],predictor=actor.predictor)
        rng.bit_generator.state=saved['rng'];assert actor.record()==saved['owner']
    start=time.monotonic()
    while not g.terminal(world):
        context=g.context(world,actor.state);request=g.request(context)
        prior=actor.record();detached=copy.deepcopy(actor);captured={}
        planner=portfolio_planner(actor,request,world,observe=g.context,events=g.events,terminal=g.terminal,
            assess=lambda w:g.goal(w).record(),request_for=g.request,feedback=g.feedback,model_id=g.model_id,
            target=g.target,horizon=8,seeds=(701,),world_model='posterior',
            cumulative_preference=cumulative_basis(g,True),include_retry=True,captured=captured,
            purpose_estimator='conditional',tail_recovery=2,coherent_cumulative_mean=True,
            tail_purpose_reflex=True,retry_until_gain=enabled)
        result=DecisionLoop.decide_batch([(actor,request)],False,planner)[0]
        key=result['decision']['action_id'];rng.random();after,flow=g.transition(world,key,'0')
        for tick,payout in spec['changes']:
            if after['tick']==tick:after['payout']=payout
        feedback=g.feedback(world,after,key,context,flow)
        resolve_feedback(actor,result['ticket'],feedback)
        control=DecisionLoop.decide_batch([(detached,request)],False,_counterfactual_roots=(key,))[0]
        resolve_feedback(detached,control['ticket'],feedback)
        assert actor.record()==detached.record() and character_key(result['context'])==axes
        record=json.loads(encoded(actor.record()))
        actor=DecisionLoop.from_record(context,record,predictor=actor.predictor)
        assert actor.record()==record
        for name in checks:checks[name]+=1
        selection=result['deliberation'];forecast=captured['forecast']
        proposal=forecast.audit['plans'][selection['selected_plan']]['proposal']
        trace.append(dict(tick=world['tick'],before=world,after=after,action=key,
            before_owner=digest(prior),owner=digest(record),truth={'0':1.},response='0',
            selected_policy=proposal,selected_purpose=selection['selected_purpose'],
            predicted_actions=captured['modeled'][proposal][0]['actions'],
            simulated_steps=captured['simulated_steps']))
        world=after
        store.save(world,actor,rng,trace,checks,elapsed+time.monotonic()-start,terminal=g.terminal(world))
        if len(trace)%32==0:print(json.dumps(dict(case=spec['id'],gain=enabled,turn=len(trace))),flush=True)
    result=dict(configuration=config,complete=True,final=world,trace=trace,checks=checks,
        score=g.goal(world).value,seconds=elapsed+time.monotonic()-start)
    (folder/'result.json').write_bytes(encoded(result))
    return audit(folder,spec)


def main(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    source={str(p.relative_to(ROOT)):sha(p.read_bytes()) for folder in ('reflex','tools')
            for p in (ROOT/folder).rglob('*.py')}
    import os
    plan=dict(version=1,specs=specs(),arms={'control':False,'gain':True},horizon=8,seeds=[701],
        sources=source,arithmetic={k:os.environ.get(k) for k in ('NPY_DISABLE_CPU_FEATURES','OPENBLAS_CORETYPE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        scope='Known deterministic preparation mechanism; not independent confirmation or replacement of the current twenty executions.',
        criteria='All eight pairs and 2048 factual steps, every loss retained, every owner/world/RNG replayed. Nonnegative every pair, at least one positive delta; no adoption from this screen.',adoption=False)
    plan_path=root/'plan.json'
    if plan_path.exists():assert json.loads(plan_path.read_bytes())==plan
    else:plan_path.write_bytes(encoded(plan))
    pairs=[]
    for spec in plan['specs']:
        arms={}
        for arm,enabled in plan['arms'].items():
            assert all(sha((ROOT/n).read_bytes())==h for n,h in source.items())
            folder=root/spec['id']/arm
            proof=audit(folder,spec) if (folder/'result.json').exists() else run(folder,spec,enabled)
            (folder/'audit.json').write_bytes(encoded(proof));arms[arm]=proof
        pair=dict(id=spec['id'],delta=arms['gain']['score']-arms['control']['score'],arms=arms)
        pairs.append(pair);print(json.dumps(dict(pair=pair)),flush=True)
        (root/'status.json').write_bytes(encoded(dict(complete=False,pairs_complete=len(pairs),adoption=False)))
    assert all(sha((ROOT/n).read_bytes())==h for n,h in source.items())
    result=dict(complete=True,pairs=pairs,actual_steps=sum(a['actual_steps'] for p in pairs for a in p['arms'].values()),
        nonnegative_every_pair=all(p['delta']>=-1e-12 for p in pairs),strict_gain=any(p['delta']>1e-12 for p in pairs),
        plan_sha256=sha(plan_path.read_bytes()),adoption=False)
    assert result['actual_steps']==2048
    (root/'summary.json').write_bytes(encoded(result));(root/'status.json').write_bytes(encoded(dict(complete=True,pairs_complete=8,adoption=False)))
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
