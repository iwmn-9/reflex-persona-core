"""Explain all first H32/consensus disagreements on their common factual owner.

Replay the frozen learner and compute the same paths once, then inspect each
cutoff's complete personal selection. A modeled purpose sacrifice is not proof
of an intentional realized loss. No hidden future law enters the forecaster;
these known-state diagnostics never substitute for complete longitudinal play.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    root=Path(parser.parse_args().root).resolve()
    plan=json.loads((root/'plan.json').read_bytes())
    assert plan['diagnostic_sha256']==sha(Path(__file__).read_bytes())
    manifest=json.loads((root/'source_snapshot.json').read_bytes())
    for name,value in manifest.items():
        assert sha((root/'_source'/name).read_bytes())==value
    sys.path.insert(0,str(root/'_source'))
    from reflex.core import digest
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    from tools.continuation_portfolio import portfolio_planner
    from tools.horizon_consensus import prefix_forecast,select_support
    from tools.owned_checkpoint import encoded
    from tools.run_continuation_portfolio import game_for
    from tools.run_owned_value import cumulative_basis
    assert sha(encoded(manifest))==plan['source']
    assert Path(sys.modules[portfolio_planner.__module__].__file__).is_relative_to(root/'_source')
    rows=[]
    for case in plan['cases']:
        destination=root/(case['id']+'.json')
        if destination.exists():
            row=json.loads(destination.read_bytes());assert row['case']==case
            rows.append(row);continue
        start=time.monotonic();data=(root/case['input']).read_bytes()
        assert sha(data)==case['sha256']
        envelope=json.loads(data);saved=envelope['payload']
        assert sha(encoded(saved))==envelope['sha256']
        spec=saved['contract']['configuration']['spec'];game=game_for(spec)
        actor=game.actor(game.context(game.start()),learning='segmented')
        for entry in saved['trace'][:case['tick']]:
            assert digest(actor.record())==entry['before_owner']
            context=game.context(entry['before'],actor.state)
            result=DecisionLoop.decide_batch([(actor,game.request(context))],False,
                _counterfactual_roots=(entry['action'],))[0]
            after,flow=game.transition(entry['before'],entry['action'],entry['response'])
            resolve_feedback(actor,result['ticket'],game.feedback(entry['before'],after,entry['action'],context,flow))
            assert after==entry['after'] and digest(actor.record())==entry['owner']
        entry=saved['trace'][case['tick']];world=entry['before'];before=actor.record()
        assert digest(before)==entry['before_owner']
        request=game.request(game.context(world,actor.state));cap={};contexts=[]
        options=dict(spec['controller']);options.pop('horizon_consensus');options.pop('minimum_horizon',None)
        original=portfolio_planner(actor,request,world,observe=game.context,events=game.events,
            terminal=game.terminal,assess=lambda w:game.goal(w).record(),request_for=game.request,
            feedback=game.feedback,model_id=game.model_id,target=game.target,horizon=32,
            seeds=(701,702,703),cumulative_preference=cumulative_basis(game,True),include_retry=True,
            capture_prefixes=True,captured=cap,**options)
        def capture(cs,ds):
            contexts.extend(copy.deepcopy(cs));return original(cs,ds)
        DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,capture)
        assert len(contexts)==1
        cutoffs=sorted({min(h,max(len(r['actions']) for rs in cap['modeled'].values() for r in rs))
                       for h in range(16,33)})
        choices=[];ballots=[]
        for horizon in cutoffs:
            forecast,_,_=prefix_forecast(contexts[0],cap['paths'],cap['modeled'],horizon,
                actor=actor,target=game.target,origin=game.goal(world),cumulative_preference=cumulative_basis(game,True))
            result=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,lambda cs,ds:forecast)[0]
            audit=result['deliberation'];assert audit['adopted']
            ballot=dict(horizon=horizon,action=result['decision']['action_id'],purpose=audit['selected_purpose'])
            ballots.append(ballot);best={}
            for name,roots in forecast.roots.items():
                best[roots[0]]=max(best.get(roots[0],-1.),forecast.purpose[name])
            choices.append(dict(**ballot,best_purpose=audit['best_purpose'],max_regret=audit['max_regret'],
                selected_purpose_gap=audit['best_purpose']-audit['selected_purpose'],
                mode=result['decision']['next_state']['mode'],primary_need=result['decision']['next_state']['primary_need'],
                best_by_root=best,progress_excluded=audit['progress_excluded'],
                purpose_excluded=audit['purpose_excluded'],selected_plan=audit['selected_plan']))
        selected,votes=select_support(ballots)
        assert ballots==entry['horizon_consensus']['ballots']
        assert selected==entry['horizon_consensus']['selected']
        assert ballots[-1]==case['fixed32_ballot'] and selected['action']!=ballots[-1]['action']
        assert actor.record()==before
        row=dict(case=case,world=world,source_owner=digest(before),choices=choices,votes=votes,
            consensus_selected=selected,fixed32_selected=ballots[-1],full_saved_choices_reproduced=True,
            factual_owner_unchanged=True,seconds=time.monotonic()-start,limit=__doc__)
        destination.write_bytes(encoded(row));rows.append(row)
        (root/'status.json').write_bytes(encoded(dict(complete=False,completed=len(rows),registered=len(plan['cases']),
            updated=datetime.now(timezone.utc).isoformat())))
        print(json.dumps(dict(id=case['id'],tick=case['tick'],complete=True)),flush=True)
    summary=dict(complete=True,source=plan['source'],adoption=False,rows=rows,limit=__doc__)
    (root/'summary.json').write_bytes(encoded(summary))
    (root/'status.json').write_bytes(encoded(dict(complete=True,completed=len(rows),registered=len(plan['cases']),
        updated=datetime.now(timezone.utc).isoformat())))


if __name__=='__main__':
    main()
