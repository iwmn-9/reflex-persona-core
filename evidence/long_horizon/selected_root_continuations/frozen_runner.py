"""Evaluator-only forks: one sampled root choice, then the unchanged real owner.

Reuse completed controls, not hypothetical perfect play. Preserve the actual
prefix, world random stream, personality and learning. Every subsequent decision
uses the frozen H32 controller and its original model seeds. Never substitutes
these post-hoc correlated forks for independent confirmation or adoption.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def sha(raw):return hashlib.sha256(raw).hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.new');tmp.write_bytes(encoded(value));tmp.replace(path)


def initialize(source):
    sys.path.insert(0,str(source))
    if os.name=='nt':
        import ctypes
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.GetCurrentProcess.restype=ctypes.c_void_p
        k.SetPriorityClass.argtypes=(ctypes.c_void_p,ctypes.c_uint)
        if not k.SetPriorityClass(k.GetCurrentProcess(),0x4000):
            raise OSError(ctypes.get_last_error(),'cannot set diagnostic priority')


def source_run(root,case):
    raw=(root/case['checkpoint']).read_bytes();assert sha(raw)==case['checkpoint_sha256']
    envelope=json.loads(raw);saved=envelope['payload'];assert sha(encoded(saved))==envelope['sha256']
    raw=(root/case['result']).read_bytes();assert sha(raw)==case['result_sha256']
    original=json.loads(raw)
    assert original['complete'] and original['trace']==saved['trace'] and original['final']==saved['world']
    assert original['configuration']==saved['contract']['configuration']
    assert original['configuration']['retry'] and original['configuration']['seeds']==[701,702,703]
    assert original['configuration']['horizon']==32
    assert saved['trace'][case['tick']]['before_owner']==case['owner']
    return saved,original


def fork_spec(original,case,job):
    spec=copy.deepcopy(original['configuration']['spec'])
    spec['id']=job['id']
    spec['diagnostic_fork']=dict(case=case['id'],tick=case['tick'],action=job['action'],
        reference_checkpoint=case['checkpoint_sha256'],reference_result=case['result_sha256'],
        kind='same-root-control-replay' if job['control'] else 'one-sampled-root-then-original-controller')
    return spec


def prepare(root,case,job):
    import numpy as np
    from reflex.core import digest,character_key
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    from tools.run_continuation_portfolio import game_for
    from tools.run_owned_value import law,sample
    from tools.owned_checkpoint import OwnedCheckpoint,encoded as checkpoint_encoded
    saved,original=source_run(root,case);spec=fork_spec(original,case,job)
    directory=root/'runs'/job['id'];directory.mkdir(parents=True,exist_ok=True)
    configuration=copy.deepcopy(original['configuration']);configuration['spec']=spec
    store=OwnedCheckpoint(directory/'checkpoint.json',('continuation-portfolio',spec['id']),
        'portfolio',configuration=configuration,every=1)
    if store.saved is not None:return spec
    game=game_for(spec);world=game.start();actor=game.actor(game.context(world),learning='segmented')
    axes=character_key(game.context(world));rng=np.random.default_rng(spec['seed'])
    checks=dict(owner_parity=0,fixed_axes=0,restored=0);trace=[]
    for entry in original['trace'][:case['tick']+1]:
        tick=world['tick'];context=game.context(world,actor.state);before=actor.record()
        assert world==entry['before'] and digest(before)==entry['before_owner']
        action=job['action'] if tick==case['tick'] else entry['action']
        assert action in game.keys(world)
        selected=DecisionLoop.decide_batch([(actor,game.request(context))],False,
            _counterfactual_roots=(action,))[0]
        truth=law(tick,spec['turns'],spec['regime']);response=sample(truth,rng.random())
        assert truth==entry['truth'] and response==entry['response']
        after,flow=game.transition(world,action,response)
        resolve_feedback(actor,selected['ticket'],game.feedback(world,after,action,context,flow))
        assert character_key(selected['context'])==axes
        checks['owner_parity']+=1;checks['fixed_axes']+=1
        if tick%23==0:
            record=json.loads(checkpoint_encoded(actor.record()))
            actor=DecisionLoop.from_record(context,record,predictor=actor.predictor)
            assert actor.record()==record;checks['restored']+=1
        if action==entry['action']:
            assert after==entry['after'] and digest(actor.record())==entry['owner']
            trace.append(copy.deepcopy(entry))
        else:
            assert tick==case['tick'] and not job['control']
            trace.append(dict(tick=tick,before=world,after=after,action=action,
                before_owner=digest(before),owner=digest(actor.record()),
                selected_policy='diagnostic-root-intervention',selected_purpose=job['archived_purpose'],
                predicted_actions=[],simulated_steps=0,response=response,truth=truth,
                conditional_increment=sum(p*(game.goal(game.transition(world,action,z)[0]).value-
                    game.goal(world).value) for z,p in truth.items()),
                intervention=dict(source='archived sampled-root forecast; no forecast rerun here',
                    banks=job['banks'],purpose_is_archived=True,subsequent_model_seeds=[701,702,703])))
        world=after
    assert world['tick']==case['tick']+1
    store.save(world,actor,rng,trace,checks,0.,terminal=game.terminal(world))
    return spec


def audit(root,case,job,*,terminal):
    import numpy as np
    from reflex.core import digest,character_key
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    from tools.run_continuation_portfolio import game_for
    from tools.run_owned_value import law,sample
    saved,original=source_run(root,case);spec=fork_spec(original,case,job)
    path=root/'runs'/job['id']/'checkpoint.json';envelope=read(path);p=envelope['payload']
    assert sha(encoded(p))==envelope['sha256']
    assert p['contract']['configuration']['spec']==spec
    assert p['trace'][:case['tick']]==original['trace'][:case['tick']]
    assert p['trace'][case['tick']]['action']==job['action']
    game=game_for(spec);world=game.start();actor=game.actor(game.context(world),learning='segmented')
    axes=character_key(game.context(world));rng=np.random.default_rng(spec['seed'])
    for entry in p['trace']:
        assert world==entry['before'] and digest(actor.record())==entry['before_owner']
        context=game.context(world,actor.state)
        choice=DecisionLoop.decide_batch([(actor,game.request(context))],False,
            _counterfactual_roots=(entry['action'],))[0]
        assert character_key(choice['context'])==axes
        truth=law(world['tick'],spec['turns'],spec['regime']);response=sample(truth,rng.random())
        assert entry['truth']==truth and entry['response']==response
        after,flow=game.transition(world,entry['action'],response)
        increment=sum(q*(game.goal(game.transition(world,entry['action'],z)[0]).value-game.goal(world).value)
            for z,q in truth.items())
        assert increment==entry['conditional_increment']
        resolve_feedback(actor,choice['ticket'],game.feedback(world,after,entry['action'],context,flow))
        assert after==entry['after'] and digest(actor.record())==entry['owner']
        world=after
    assert world==p['world'] and actor.record()==p['owner'] and rng.bit_generator.state==p['rng']
    assert bool(game.terminal(world))==terminal
    result=dict(job=job['id'],case=case['id'],intervention_action=job['action'],
        actual_decisions_replayed=len(p['trace']),unchanged_prefix=case['tick'],
        checkpoint_sha256=sha(path.read_bytes()),factual_owner_replayed=True,world_rng_replayed=True,
        personality_preserved=True,complete=terminal,adoption=False)
    if terminal:
        row=read(path.parent/'result.json')
        assert row['complete'] and row['trace']==p['trace'] and row['final']==world
        assert row['configuration']==p['contract']['configuration']
        assert row['runtime']==original['runtime'] and row['score']==game.goal(world).value
        from tools.summarize_horizon_study import realized_account
        from reflex.laboratory import PROFILES
        accounts=[]
        for value in (original,row):
            accounts.append(realized_account(dict(game=spec['game'],profile=PROFILES[spec['profile']]['id'],
                turns=spec['turns'],seed=spec['seed'],regime=spec['regime'],trace=value['trace'],score=value['score'])))
        a,b=accounts
        result.update(score_delta=row['score']-original['score'],
            conditional_delta=b['conditional_increments']-a['conditional_increments'],
            residual_delta=b['residual']-a['residual'],scores=[original['score'],row['score']],
            new_full_planning_decisions=len(p['trace'])-case['tick']-1,
            result_sha256=sha((path.parent/'result.json').read_bytes()),accounts=accounts)
        assert abs(result['conditional_delta']+result['residual_delta']-result['score_delta'])<1e-10
        if 'bellman' in a:
            result.update(reference_choice_delta=b['bellman']['residual_removed']-a['bellman']['residual_removed'],
                reference_residual_delta=b['bellman']['residual']-a['bellman']['residual'])
    else:
        assert job['control'] and len(p['trace'])==case['tick']+2
        assert p['trace']==original['trace'][:len(p['trace'])]
        result.update(complete=True,scope='one real H32 continuation decision matches archived control exactly; original terminal control is reused',
            complete_new_world=False,full_forecast_decisions_reproduced=1)
    return result


def run_job(args):
    root,case,job=args
    from tools.run_continuation_portfolio import execution
    assert Path(sys.modules[execution.__module__].__file__).resolve().is_relative_to(root/'_source')
    target=root/'runs'/job['id'];proof=target/'verification.json'
    if proof.exists():
        receipt=read(proof);assert receipt['complete']
        assert receipt['checkpoint_sha256']==sha((target/'checkpoint.json').read_bytes())
        return receipt
    spec=prepare(root,case,job)
    current=read(target/'checkpoint.json')['payload']['world']['tick']
    if not job['control'] or current==case['tick']+1:
        result=execution(target,spec,True,max_decisions=1 if job['control'] else None)
        if job['control']:assert result is None
        else:assert result is not None and result['complete']
    receipt=audit(root,case,job,terminal=not job['control']);save(proof,receipt)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    root=Path(args.root).resolve();plan=read(root/'plan.json')
    assert sha(Path(__file__).read_bytes())==plan['runner_sha256']
    manifest=read(root/'source_snapshot.json');assert sha(encoded(manifest))==plan['source']
    for name,h in manifest.items():assert sha((root/'_source'/name).read_bytes())==h
    lock=root/'execution.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,encoded(dict(pid=os.getpid())));os.close(fd)
    state=dict(complete=False,adoption=False,pid=os.getpid(),controls_completed=0,
        branches_completed=0,registered_branches=len(plan['branches']),source=plan['source'])
    jobs=plan['controls']+plan['branches'];cases={c['id']:c for c in plan['cases']}
    def status(**values):
        state.update(values,updated=datetime.now(timezone.utc).isoformat())
        progress=[]
        for job in jobs:
            path=root/'runs'/job['id']/'checkpoint.json'
            if path.exists():
                raw=read(path);assert sha(encoded(raw['payload']))==raw['sha256']
                tick=raw['payload']['world']['tick'];case=cases[job['case']]
                progress.append(dict(id=job['id'],control=job['control'],tick=tick,
                    new_planned_decisions=max(0,tick-case['tick']-1),
                    result_exists=(path.parent/'result.json').exists()))
        state['progress']=progress;save(root/'status.json',state)
    results=[]
    try:
        with ProcessPoolExecutor(max_workers=plan['workers'],initializer=initialize,
                initargs=(root/'_source',)) as pool:
            for phase,group in (('verifying-original-resume',plan['controls']),('continuing-all-sampled-root-actions',plan['branches'])):
                pending={pool.submit(run_job,(root,cases[j['case']],j)) for j in group}
                status(phase=phase)
                while pending:
                    finished,pending=wait(pending,timeout=45,return_when=FIRST_COMPLETED)
                    for future in finished:
                        result=future.result();results.append(result)
                        counter='controls_completed' if phase=='verifying-original-resume' else 'branches_completed'
                        state[counter]+=1
                        save(root/'receipts.json',dict(complete=False,adoption=False,rows=results))
                        print(json.dumps({k:v for k,v in result.items() if k not in ('accounts',)}),flush=True)
                    status()
        assert state['controls_completed']==len(plan['controls']) and state['branches_completed']==len(plan['branches'])
        save(root/'summary.json',dict(complete=True,adoption=False,source=plan['source'],
            plan_sha256=sha((root/'plan.json').read_bytes()),rows=results,limits=plan['limits']))
        status(complete=True,phase='all-forks-audited-needs-review')
    except Exception as error:
        status(error=repr(error));raise
    finally:lock.unlink()


if __name__=='__main__':main()
