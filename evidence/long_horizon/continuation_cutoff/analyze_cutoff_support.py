"""Evaluate cutoff agreement on previously saved complete hypothetical paths."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from reflex.intertemporal import Branch
from reflex.examples import effect
from tools.horizon_consensus import prefix_forecast,select_support
from tools.owned_checkpoint import encoded,sha
from tools.run_continuation_portfolio import game_for
from tools.run_owned_value import cumulative_basis


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root)
    plan=json.loads((root/'plan.json').read_bytes());summary=json.loads((root/'summary.json').read_bytes());results=[]
    assert summary['complete']
    for case in plan['cases']:
        data=(root/(case['id']+'.json')).read_bytes()
        assert sha(data)==next(r['sha256'] for r in summary['rows'] if r['case']==case['id'])
        original=json.loads(data);saved=json.loads((root/case['input']).read_bytes())['payload']
        spec=saved['contract']['configuration']['spec'];g=game_for(spec)
        actor=g.actor(g.context(g.start()),learning='segmented')
        for t in saved['trace'][:case['tick']]:
            assert digest(actor.record())==t['before_owner']
            c=g.context(t['before'],actor.state);d=DecisionLoop.decide_batch([(actor,g.request(c))],False,
                _counterfactual_roots=(t['action'],))[0]
            after,flow=g.transition(t['before'],t['action'],t['response'])
            resolve_feedback(actor,d['ticket'],g.feedback(t['before'],after,t['action'],c,flow))
            assert digest(actor.record())==t['owner']
        w=original['world'];source=digest(actor.record());assert source==original['factual_owner']
        request=g.request(g.context(w,actor.state));c=actor.preview(request)[0]
        # Root events are enumerated; every group of three independent model
        # seeds shares one root probability from the factual learner.
        from reflex.closed_loop import owned_rollout
        model=owned_rollout(actor,observe=g.context,events=g.events,terminal=g.terminal,
            assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,feedback=g.feedback,
            model_id=g.model_id,target=g.target,tail_steps=31,seeds=(701,702,703),world_model='posterior')
        paths={}
        for name,rows in original['modeled'].items():
            first=model.owner_transitions(w,rows[0]['actions'][0],actor,c)
            paths[name]=[]
            for i,row in enumerate(rows):
                state=w;flows=[]
                for key,prefix in zip(row['actions'],row['prefixes']):
                    after,flow=g.transition(state,key,prefix['world']['signal'])
                    assert after==prefix['world'];flows.append(flow);state=after
                flows.extend(effect() for _ in range(32-len(flows)))
                paths[name].append(Branch(first[i//3].probability/3,tuple(flows),(1.,)*32))
        ballots=[]
        for h in range(16,33):
            f,_,_=prefix_forecast(c,paths,original['modeled'],h,actor=actor,target=g.target,
                origin=g.goal(w),cumulative_preference=cumulative_basis(g,True))
            d=DecisionLoop.decide_batch([(copy.deepcopy(actor),request)],False,lambda cs,ds:f)[0]
            ballots.append(dict(horizon=h,action=d['decision']['action_id'],purpose=d['deliberation']['selected_purpose']))
        assert ballots[0]['action']==saved['trace'][case['tick']]['action']
        chosen,votes=select_support(ballots);assert digest(actor.record())==source
        results.append(dict(case=case['id'],original=ballots[0]['action'],single32=ballots[-1]['action'],
            selected=chosen,votes=votes,ballots=ballots,owner=source))
    result=dict(complete=True,cases=results,adoption=False,
        limit='Known-state diagnosis only. Single H32 is a distinct explanation to test longitudinally; no causal or generalization claim.')
    (root/'consensus_support.json').write_bytes(encoded(result));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
