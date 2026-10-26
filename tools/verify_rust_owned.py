"""Native factual-owner/selector fidelity. Python remains the H32 path generator.

This does not replace or rerun the registered intelligence/quality experiment.
Every factual transition is driven by the native chosen action; the reference
learner is an oracle only. Full H32 paths, when requested, use the frozen Python
generator and are then independently selected by the native owner.
"""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
import numpy as np
from reflex.core import FEATURES, NEEDS, compile_batch
from reflex.laboratory import PROFILES
from reflex.model_observer import resolve_feedback
from reflex.fixed_horizon import FixedHorizonOwner
from reflex.decision_loop import DecisionLoop
from tools.continuing_games import ContinuingGame
from tools.run_owned_value import cumulative_basis
from tools.verify_rust_immediate import request, encoded
from tools.verify_rust_runtime import Driver
from reflex.native_owned import NativeOwnedBoundary, exported_state


def packet(c):
    return request(c, 'personal', .5, stochastic=False)


def native_input(g, actor, world):
    c = g.context(world, actor.state); req = g.request(c)
    p = actor.predictor
    factors, legal, models = p._inputs(c)
    converted = packet(c)
    conditional = {}
    for action in c['actions']:
        rows = [dict(g.outcome(c, action['id'], z), p=1/len(legal)) for z in legal]
        temp = copy.deepcopy(c); temp['actions'] = [dict(action, outcomes=rows)]
        conditional[action['id']] = packet(temp)['observation']['actions'][0]['outcomes']
    return dict(observation=converted['observation'],
        bindings={k:dict(method=b.method,situation=b.situation,estimated=[FEATURES.index(f) for f in b.estimated]) for k,b in req.bindings.items()},
        responses=dict(context=list(factors),legal=list(legal),models=[[models[k][z] for z in legal] for k in p.learner.names]),
        conditional=conditional,purpose=asdict(req.purpose),threatened=req.threatened,maintains_advantage=req.maintains_advantage)


def native_state(actor):
    return exported_state(actor)


def native_feedback(fb):
    return dict(observed=None if fb.observed is None else fb.observed.tolist(),
        response=None if fb.event is None else fb.event['revealed_action'],
        needs=None if fb.needs is None else dict(progress=[fb.needs.get(k) for k in NEEDS],maintained=[k in fb.maintained for k in NEEDS],completed=[k in fb.completed for k in NEEDS]),
        purpose=None if fb.purpose is None else asdict(fb.purpose))


def compare_decision(actual, expected):
    for k in ('action_id','target','reading_used','request_reading','next_state'):
        assert actual[k] == expected[k], (k,actual[k],expected[k])


def compare_numeric(actual, result, before, req, forecast=None):
    permitted=before._prepare(req)['progress_allowed']
    kwargs={}
    if forecast is None:
        b=compile_batch([result['context']])
        allowed=np.array([[name in permitted for name in b.ids[0]]])
    else:
        b=compile_batch(forecast.contexts,outcome_limit=forecast.outcome_limit)
        supported=np.array([forecast.roots[k][0] in permitted for k in b.ids[0]])
        purpose=np.array([forecast.purpose[k] for k in b.ids[0]])
        allowed=(supported & (purpose>=max(purpose[supported])-forecast.max_regret-1e-12))[None,:]
        kwargs=dict(return_frame=forecast.return_frame,return_means=forecast.return_means)
    b=replace(b,legal=b.legal & allowed)
    expected=before.policy.decide(b,False,**kwargs)
    assert [r['id'] for r in actual['scores']] == list(b.ids[0])
    assert [r['eligible'] for r in actual['scores']] == expected.eligible[0].tolist()
    assert actual['rng'] == b.rng[0].tolist()
    error=max(float(np.max(abs(np.array([r['score'] for r in actual['scores']])-expected.scores[0]))),
        float(np.max(abs(np.array([r['features'] for r in actual['scores']])-expected.features[0]))))
    assert error<=2e-12,error
    return error


def compare_state(actual, expected, path='state'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (path, actual.keys(), expected.keys())
        return max((compare_state(actual[k],v,path+'.'+k) for k,v in expected.items()),default=0.)
    if isinstance(expected, list):
        assert len(actual) == len(expected), (path,len(actual),len(expected))
        return max((compare_state(a,b,path+f'[{i}]') for i,(a,b) in enumerate(zip(actual,expected))),default=0.)
    if type(expected) is float:
        error=abs(actual-expected)
        assert np.isfinite(actual) and error <= 2e-12, (path, actual, expected)
        return error
    assert actual == expected, (path,actual,expected)
    return 0.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=160)
    p.add_argument('--profiles',type=int,default=4)
    p.add_argument('--horizon-steps',type=int,default=0)
    p.add_argument('--horizon-at',type=int,default=0)
    p.add_argument('--case-index',type=int,choices=range(16))
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    binary=R/'rust/target/debug/examples/owned_verify.exe'
    if not binary.exists(): binary=binary.with_suffix('')
    sources=[*sorted((R/'rust').glob('reflex-*/src/*.rs')),*sorted((R/'rust').glob('reflex-*/Cargo.toml')),
        *sorted((R/'reflex').glob('*.py')),Path(__file__),R/'tools/continuing_games.py',R/'tools/run_owned_value.py',
        R/'rust/Cargo.lock',R/'rust/Cargo.toml',binary]
    source_hashes={f.relative_to(R).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    (args.output/'inputs.json').write_bytes(encoded(dict(sources=source_hashes,steps=args.steps,profiles=args.profiles,
        horizon_at=args.horizon_at,horizon_steps=args.horizon_steps,case_index=args.case_index,numpy=np.__version__,python=sys.version)))
    driver=Driver(binary); trace=[]; prediction_error=0.; state_error=0.; score_error=0.; started=time.time()
    try:
        for family_index,family in enumerate(('auction','resources','combat','gambling')):
            for profile in range(args.profiles):
                if args.case_index is not None and args.case_index != family_index*4+profile: continue
                g=ContinuingGame(family,PROFILES[profile],args.steps,profile%2)
                world=g.start(); actor=g.actor(g.context(world),learning='segmented')
                settings=dict(observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
                    request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,cumulative_preference=cumulative_basis(g,True))
                boundary=NativeOwnedBoundary(FixedHorizonOwner(g.actor(g.context(world),learning='segmented'),**settings),world)
                config=boundary.configuration
                driver.call(op='new',config=config)
                for tick in range(args.steps):
                    (args.output/'progress.json').write_bytes(encoded(dict(game=family,profile=profile,tick=tick,completed_decisions=len(trace))))
                    factual=driver.call(op='state')['state']
                    inp=boundary.input(world,factual)
                    before=copy.deepcopy(actor); req=g.request(g.context(world,actor.state)); frozen=None
                    compare_state(inp,native_input(g,actor,world),'input')
                    expected_probs=list(actor.predictor.predict(g.context(world,actor.state)).values())
                    got_probs=driver.call(op='predict',inputs=inp['responses'])['probabilities']
                    error=float(np.max(abs(np.asarray(got_probs)-expected_probs))); prediction_error=max(error,prediction_error)
                    assert error <= 2e-12, (family,profile,tick,'probability',error)
                    actual=driver.call(op='prepare',input=inp,stochastic=False)
                    long=args.horizon_at <= tick < args.horizon_at+args.horizon_steps
                    if long:
                        captured=[]; factory=boundary._template._with_actor
                        def capture(a):
                            controller=factory(a);captured.append(controller);return controller
                        with patch.object(boundary._template,'_with_actor',side_effect=capture):
                            forecast=boundary.forecast(world,factual)
                        # Translation parity: ONE immutable model forecast, two
                        # independent selectors. The generating future owner
                        # starts from native factual memory, not the oracle NPC.
                        frozen=captured[0].last['forecast']
                        expected=DecisionLoop.decide_batch([(actor,g.request(g.context(world,actor.state)))],False,
                            lambda contexts,decisions:frozen)[0]
                        actual=driver.call(op='forecast',forecast=forecast)
                    else: expected=actor.decide(g.request(g.context(world,actor.state)),False)
                    compare_decision(actual['decision'],expected['decision'])
                    score_error=max(score_error,compare_numeric(actual['decision'],expected,before,req,frozen))
                    action=actual['decision']['action_id']; ticket=driver.call(op='commit',action=action)['ticket']
                    # Fixed observed regime shift, including response absent from library peaks.
                    response=str((tick//23+profile)%3)
                    after,flow=g.transition(world,action,response)
                    fb=g.feedback(world,after,action,expected['context'],flow)
                    driver.call(op='resolve',ticket=ticket,feedback=native_feedback(fb)); resolve_feedback(actor,expected['ticket'],fb)
                    got_state=driver.call(op='state')['state']; wanted=native_state(actor)
                    state_error=max(state_error,compare_state(got_state,wanted))
                    world=after
                    saved=driver.call(op='save',world=world)['save']
                    if tick%31==30: driver.restart()
                    driver.call(op='restore',config=config,world=world,save=saved)
                    assert driver.call(op='state')['state'] == got_state
                    trace.append(dict(game=family,profile=profile,tick=tick,action=action,response=response,horizon=long))
                print(json.dumps(dict(game=family,profile=profile,steps=args.steps,complete=True)),flush=True)
                (args.output/f'{family}-{profile}.json').write_bytes(encoded(dict(complete=True,
                    trace=[r for r in trace if r['game']==family and r['profile']==profile],state=got_state,
                    world=world,max_prediction_error=prediction_error,max_state_error=state_error)))
        assert all(hashlib.sha256((R/name).read_bytes()).hexdigest()==sha for name,sha in source_hashes.items()), 'source changed during parity run'
        result=dict(complete=True,decisions=len(trace),saves=len(trace),max_prediction_error=prediction_error,max_state_error=state_error,max_score_error=score_error,seconds=time.time()-started,
            horizon_decisions=sum(r['horizon'] for r in trace),trace_sha256=hashlib.sha256(encoded(trace)).hexdigest())
        (args.output/'summary.json').write_bytes(encoded(result)); (args.output/'trace.json').write_bytes(encoded(trace))
    except BaseException:
        (args.output/'failure.json').write_bytes(encoded(dict(last=driver.last,trace=trace)))
        raise
    finally: driver.close()


if __name__=='__main__': main()
