"""Read-only replay and accounting of a registered complete subset, not adoption.

The true law and exact purpose reference are evaluator-only. Their opportunity
loss includes personality and information constraints; it is not pure stupidity.
Frozen-current-belief reference assumes the present law persists, unlike the
playing posterior forecast. No diagnostic information is given to an NPC.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def encoded(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    root=Path(parser.parse_args().root).resolve();plan_bytes=(root/'plan.json').read_bytes();plan=json.loads(plan_bytes)
    source=(root/plan['trial_root']).resolve();manifest=json.loads((source/'source_snapshot.json').read_bytes())
    assert sha((source/'source_snapshot.json').read_bytes())==plan['source_manifest_sha256']
    assert sha((source/'plan.json').read_bytes())==plan['trial_plan_sha256']
    assert sha(encoded(manifest))==plan['source']
    for name,expected in manifest.items():assert sha((source/'_source'/name).read_bytes())==expected,name
    def read(name):
        data=(source/name).read_bytes();assert sha(data)==plan['inputs'][name],name
        return json.loads(data)
    sys.path.insert(0,str(source/'_source'))
    import numpy as np
    from reflex.core import digest,character_key
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    from reflex.laboratory import PROFILES
    from tools.run_continuation_portfolio import game_for
    from tools.run_owned_value import law,sample
    from tools.summarize_horizon_study import realized_account
    from tools.diagnose_auction_continuity import reference
    pairs=[];replayed=0
    for spec in plan['cases']:
        rows={};runtimes=[]
        for arm,retry in (('baseline',False),('candidate',True)):
            prefix=spec['id']+'/'+arm+'/'
            result=read(prefix+'result.json');env=read(prefix+'checkpoint.json');saved=env['payload']
            assert sha(encoded(saved))==env['sha256']
            assert result['complete'] and result['trace']==saved['trace'] and result['final']==saved['world']
            assert result['configuration']==saved['contract']['configuration']
            assert result['configuration']['spec']==spec and result['configuration']['retry']==retry
            for name,h in result['configuration']['sources'].items():assert manifest['tools/'+name]==h
            game=game_for(spec);world=game.start();actor=game.actor(game.context(world),learning='segmented')
            axes=character_key(game.context(world));rng=np.random.default_rng(spec['seed']);steps=[]
            truth_q=(reference(spec['turns'],spec['seed']%2,
                lambda t:tuple(law(t,spec['turns'],spec['regime']).values())) if spec['game']=='auction' else None)
            for entry in result['trace']:
                assert world==entry['before'] and digest(actor.record())==entry['before_owner']
                context=game.context(world,actor.state);belief=actor.predictor.predict(context)
                assert digest(actor.record())==entry['before_owner']
                decision=DecisionLoop.decide_batch([(actor,game.request(context))],False,
                    _counterfactual_roots=(entry['action'],))[0]
                assert character_key(decision['context'])==axes
                truth=law(world['tick'],spec['turns'],spec['regime']);response=sample(truth,rng.random())
                assert response==entry['response'] and truth==entry['truth']
                after,flow=game.transition(world,entry['action'],response)
                conditional=sum(p*(game.goal(game.transition(world,entry['action'],z)[0]).value-game.goal(world).value) for z,p in truth.items())
                assert abs(conditional-entry['conditional_increment'])<1e-12
                resolve_feedback(actor,decision['ticket'],game.feedback(world,after,entry['action'],context,flow))
                assert after==entry['after'] and digest(actor.record())==entry['owner']
                row=dict(tick=world['tick'],action=entry['action'],cash=world['cash'],belief=belief,
                    truth=truth,conditional_increment=conditional,selected_policy=entry['selected_policy'],
                    predicted_actions=entry['predicted_actions'],selected_purpose=entry['selected_purpose'])
                if truth_q:
                    q=truth_q(world['tick'],int(world['cash']))
                    assumed=reference(spec['turns'],spec['seed']%2,lambda t,p=tuple(belief[str(z)] for z in range(3)):p)
                    assumed_q=assumed(world['tick'],int(world['cash']))
                    row.update(truth_options=q,belief_options=assumed_q,
                        truth_opportunity_loss=max(q.values())-q[entry['action']],
                        belief_opportunity_loss=max(assumed_q.values())-assumed_q[entry['action']])
                steps.append(row);world=after;replayed+=1
            assert game.terminal(world) and world==result['final'] and actor.record()==saved['owner']
            assert rng.bit_generator.state==saved['rng'] and result['score']==game.goal(world).value
            legacy=dict(game=spec['game'],profile=PROFILES[spec['profile']]['id'],turns=spec['turns'],
                seed=spec['seed'],regime=spec['regime'],trace=result['trace'],score=result['score'])
            account=realized_account(legacy);runtimes.append(result['runtime'])
            rows[arm]=dict(score=result['score'],final=result['final'],fixed_persona=axes,
                actual_owner=digest(actor.record()),account=account,steps=steps,actions=dict(Counter(x['action'] for x in steps)))
        assert runtimes[0]==runtimes[1] and rows['baseline']['fixed_persona']==rows['candidate']['fixed_persona']
        a=rows['baseline'];b=rows['candidate'];aa=a['account'];bb=b['account']
        delta=b['score']-a['score'];difference=dict(score=delta,conditional=bb['conditional_increments']-aa['conditional_increments'],residual=bb['residual']-aa['residual'])
        assert abs(difference['conditional']+difference['residual']-delta)<1e-10
        if 'bellman' in aa:
            assert abs(aa['bellman']['initial_perfect_information']-bb['bellman']['initial_perfect_information'])<1e-12
            difference.update(reference_choice=bb['bellman']['residual_removed']-aa['bellman']['residual_removed'],reference_residual=bb['bellman']['residual']-aa['bellman']['residual'])
            assert abs(difference['reference_choice']+difference['reference_residual']-delta)<1e-10
        comparisons=[]
        for x,y in zip(a['steps'],b['steps']):
            assert x['tick']==y['tick']
            row=dict(tick=x['tick'],actions=[x['action'],y['action']],cash=[x['cash'],y['cash']],belief_equal=x['belief']==y['belief'])
            if 'truth_opportunity_loss' in x:
                row.update(reference_extra_loss=y['truth_opportunity_loss']-x['truth_opportunity_loss'],belief_extra_loss=y['belief_opportunity_loss']-x['belief_opportunity_loss'])
            comparisons.append(row)
        pairs.append(dict(spec=spec,difference=difference,arms=rows,comparisons=comparisons))
        print(json.dumps(dict(id=spec['id'],difference=difference,replayed=replayed)),flush=True)
    for name,expected in plan['inputs'].items():assert sha((source/name).read_bytes())==expected
    result=dict(review_complete=True,whole_trial_complete=False,adoption=False,at=datetime.now(timezone.utc).isoformat(),
        reviewed_pairs=len(pairs),whole_trial_registered=plan['whole_trial_registered'],replayed_actual_decisions=replayed,
        plan_sha256=sha(plan_bytes),auditor_sha256=sha(Path(__file__).read_bytes()),source=plan['source'],
        source_verified=True,actual_owner_replayed=True,pairs=pairs,limits=__doc__)
    (root/'review.json').write_bytes(encoded(result))


if __name__=='__main__':main()
