"""Reuse only already computed H32 decisions on an identical factual prefix.

Consensus evaluates fixed H32 as its last ballot on the same frozen paths. If
that ballot wins, the selected full forecast is also that exact H32 forecast.
Replaying its actual action and feedback yields the same next owner and world.
Stop at the FIRST disagreement; subsequent evidence is never copied across
diverging worlds. This is shared computation, not another independent trial.
"""
import argparse
import copy
import json
import platform
import sys
from pathlib import Path


def project_entry(entry):
    out=copy.deepcopy(entry);meta=out['horizon_consensus']
    assert meta['maximum_horizon']==32 and meta['minimum_horizon']==16
    assert meta['source_owner']==out['before_owner']
    last=meta['ballots'][-1]
    assert last['horizon']==max(b['horizon'] for b in meta['ballots'])
    if last['action']!=out['action']:
        return None
    # Equal actions must select the longest supporting forecast, not a shorter
    # forecast with an accidentally equal root but different subjective value.
    assert meta['selected']==last
    assert last['purpose']==out['selected_purpose']
    meta.update(minimum_horizon=32,ballots=[last],votes={last['action']:1})
    return out


def reuse(source,target,key):
    from tools.owned_checkpoint import OwnedCheckpoint,encoded,sha
    from tools.run_continuation_portfolio import game_for
    from tools.run_owned_value import law,sample
    from reflex.core import character_key,digest
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    import numpy as np

    manifests=[json.loads((p/'source_snapshot.json').read_bytes()) for p in (source,target)]
    assert manifests[0]==manifests[1]
    for folder in (source,target):
        for name,value in manifests[0].items():assert sha((folder/'_source'/name).read_bytes())==value
    plans=[json.loads((p/'plan.json').read_bytes()) for p in (source,target)]
    assert plans[0]['source']==plans[1]['source']==sha(encoded(manifests[0]))
    original,spec=[next(s for s in p['specs'] if s['id']==key) for p in plans]
    assert dict(original,controller=dict(original['controller'],minimum_horizon=32),
        reference=spec['reference'])==spec
    assert original['controller']['horizon_consensus'] and 'minimum_horizon' not in original['controller']
    data=(source/key/'candidate/checkpoint.json').read_bytes();envelope=json.loads(data);saved=envelope['payload']
    assert sha(encoded(saved))==envelope['sha256']
    assert saved['contract']['configuration']['spec']==original
    assert saved['contract']['configuration']['retry'] is True
    projected=[]
    for entry in saved['trace']:
        candidate=project_entry(entry)
        if candidate is None:break
        projected.append(candidate)
    assert projected, 'No identical prefix to reuse'
    destination=target/key/'candidate'
    assert not destination.exists(), 'Never overwrite actual H32 execution or its checkpoint'
    config=copy.deepcopy(saved['contract']['configuration']);config['spec']=spec
    stage=target/('prefix-replay-'+key);stage.mkdir(exist_ok=False)
    store=OwnedCheckpoint(stage/'checkpoint.json',('continuation-portfolio',key),'portfolio',configuration=config,every=1)
    expected=copy.deepcopy(saved['contract']);expected['configuration']=config
    assert store.contract==expected, 'Same frozen runtime and actor contract required'
    game=game_for(spec);world=game.start();actor=game.actor(game.context(world),learning='segmented')
    axes=character_key(game.context(world));rng=np.random.default_rng(spec.get('seed',0))
    checks=dict(owner_parity=0,fixed_axes=0,restored=0)
    for entry in projected:
        assert entry['before']==world and entry['before_owner']==digest(actor.record())
        context=game.context(world,actor.state);request=game.request(context)
        result=DecisionLoop.decide_batch([(actor,request)],False,_counterfactual_roots=(entry['action'],))[0]
        assert character_key(result['context'])==axes
        truth=law(world['tick'],spec['turns'],spec['regime']) if spec['mechanism']=='world' else {'0':1.}
        response=sample(truth,rng.random())
        assert response==entry['response'] and truth==entry['truth']
        after,flow=game.transition(world,entry['action'],response)
        for tick,payout in spec.get('changes',[]):
            if after['tick']==tick:after['payout']=payout
        resolve_feedback(actor,result['ticket'],game.feedback(world,after,entry['action'],context,flow))
        assert after==entry['after'] and digest(actor.record())==entry['owner']
        checks['owner_parity']+=1;checks['fixed_axes']+=1
        if world['tick']%23==0:
            record=json.loads(encoded(actor.record()))
            actor=DecisionLoop.from_record(context,record,predictor=actor.predictor)
            assert actor.record()==record;checks['restored']+=1
        world=after
    # Conservatively carry the entire source elapsed time. It is shared and,
    # for partial prefixes, an upper bound; never report it as H32-only timing.
    store.save(world,actor,rng,projected,checks,saved['elapsed'],terminal=game.terminal(world))
    if game.terminal(world):
        assert len(projected)==len(saved['trace'])
        assert world==saved['world'] and actor.record()==saved['owner'] and rng.bit_generator.state==saved['rng']
        original_result=json.loads((source/key/'candidate/result.json').read_bytes())
        assert original_result['complete'] and original_result['trace']==saved['trace']
        runtime=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform())
        assert original_result['runtime']==runtime
        row=dict(configuration=config,complete=True,final=world,score=game.goal(world).value,
            trace=projected,checks=checks,seconds=saved['elapsed'],physical_replay=len(projected),runtime=runtime)
        (stage/'result.json').write_bytes(encoded(row))
    proof=dict(id=key,replayed_decisions=len(projected),complete=game.terminal(world),
        source_checkpoint=str(source/key/'candidate/checkpoint.json'),source_checkpoint_sha256=sha(data),
        source=plans[0]['source'],target_plan_sha256=sha((target/'plan.json').read_bytes()),
        reused_last_ballot=True,exact_actual_owner_and_world_rng=True,
        first_unshared_tick=None if game.terminal(world) else world['tick'],
        timing='Shared original elapsed time carried as an upper bound; not standalone H32 performance.',
        independence='Known-condition comparison with shared exact prefix; not independent confirmation.')
    (stage/'prefix_reuse.json').write_bytes(encoded(proof))
    destination.parent.mkdir(exist_ok=True)
    stage.rename(destination)  # Fails rather than replacing a concurrent executor.
    return proof


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--target',required=True)
    p.add_argument('--key',required=True);args=p.parse_args()
    source,target=Path(args.source).resolve(),Path(args.target).resolve()
    sys.path.insert(0,str(target/'_source'))
    print(json.dumps(reuse(source,target,args.key)),flush=True)


if __name__=='__main__':main()
