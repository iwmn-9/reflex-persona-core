"""Long-play ablation: change learning while retaining immediate decisions.

This isolates the learner from H32 numerical integration. It does not replace
the registered H32 study, its adverse cases, or independent confirmation gates.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reflex.core import digest,character_key
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.run_adverse_recovery import game_for,actor_for,replay
from tools.run_owned_value import law,sample
from tools.owned_checkpoint import encoded,sha


def pair(spec):
    game=game_for(spec);out={};started=time.monotonic()
    for arm in ('stable','transfer'):
        world=game.start();actor=actor_for(game,game.context(world),arm)
        rng=np.random.default_rng(spec['seed']);trace=[];axes=character_key(game.context(world))
        while not game.terminal(world):
            before=digest(actor.record());context=game.context(world,actor.state)
            prediction=actor.predictor.predict(context)
            result=DecisionLoop.decide_batch([(actor,game.request(context))],False)[0]
            action=result['decision']['action_id'];truth=law(world['tick'],spec['turns'],spec['regime'])
            response=sample(truth,rng.random());after,flow=game.transition(world,action,response)
            resolve_feedback(actor,result['ticket'],game.feedback(world,after,action,result['context'],flow))
            saved=json.loads(encoded(actor.record()))
            actor=DecisionLoop.from_record(context,saved,predictor=actor.predictor)
            assert actor.record()==saved and character_key(game.context(after))==axes
            trace.append(dict(tick=world['tick'],before=world,after=after,before_owner=before,owner=digest(saved),
                action=action,response=response,truth=truth,response_prediction=prediction,
                conditional_increment=sum(q*(game.goal(game.transition(world,action,z)[0]).value-game.goal(world).value) for z,q in truth.items())))
            world=after
        checked=replay(game,spec,trace,actor.record(),rng.bit_generator.state,arm)
        assert checked==spec['turns']
        out[arm]=dict(score=game.goal(world).value,conditional=sum(r['conditional_increment'] for r in trace),
            trace=trace,final_owner=actor.record(),final_rng=rng.bit_generator.state,replayed_steps=checked)
    assert [(r['response'],r['truth']) for r in out['stable']['trace']]==[(r['response'],r['truth']) for r in out['transfer']['trace']]
    return dict(spec=spec,arms=out,delta={k:out['transfer'][k]-out['stable'][k] for k in ('score','conditional')},
        seconds=time.monotonic()-started)


def run(plan_path,output,workers):
    plan_path,output=Path(plan_path),Path(output)
    plan=json.loads(plan_path.read_bytes())
    for name,expected in plan['sources'].items():assert sha((ROOT/name).read_bytes())==expected
    output.mkdir(parents=True,exist_ok=True)
    contract=sha(plan_path.read_bytes());receipt=output/'contract.json'
    if receipt.exists():assert json.loads(receipt.read_bytes())['plan_sha256']==contract
    else:receipt.write_bytes(encoded(dict(plan_sha256=contract)))
    jobs=[];rows=[]
    for spec in plan['specs']:
        path=output/(spec['id']+'.json.gz')
        if path.exists():
            row=json.loads(gzip.decompress(path.read_bytes()))
            assert row['spec']==spec and all(r['replayed_steps']==spec['turns'] for r in row['arms'].values())
            rows.append(row)
        else:jobs.append(spec)
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed(pool.submit(pair,spec) for spec in jobs):
            row=future.result();path=output/(row['spec']['id']+'.json.gz')
            with path.open('xb') as stream:stream.write(gzip.compress(encoded(row),mtime=0))
            rows.append(row)
            status=dict(complete=len(rows)==len(plan['specs']),pairs=len(rows),registered=len(plan['specs']),
                actual_steps=sum(2*r['spec']['turns'] for r in rows),seconds=time.monotonic()-started,plan_sha256=contract)
            (output/'status.json').write_bytes(encoded(status))
            print(json.dumps(status),flush=True)
    assert len(rows)==len(plan['specs'])
    groups=[]
    for game in sorted({r['spec']['game'] for r in rows}):
        for regime in sorted({r['spec']['regime'] for r in rows}):
            selected=[r for r in rows if (r['spec']['game'],r['spec']['regime'])==(game,regime)]
            groups.append(dict(game=game,regime=regime,pairs=len(selected),
                mean_delta={k:sum(r['delta'][k] for r in selected)/len(selected) for k in ('score','conditional')},
                better=sum(r['delta']['score']>1e-12 for r in selected),worse=sum(r['delta']['score']< -1e-12 for r in selected)))
    result=dict(complete=True,pairs=len(rows),actual_steps=sum(2*r['spec']['turns'] for r in rows),groups=groups,
        plan_sha256=contract,adoption=False,controller='immediate only; H32 still requires its own full experiment',
        cases=[dict(spec=r['spec'],delta=r['delta']) for r in rows],
        artifacts={p.name:sha(p.read_bytes()) for p in output.glob('*.json.gz')})
    (output/'summary.json').write_bytes(encoded(result))
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','cases')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.plan,a.output,a.workers)
