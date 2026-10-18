"""Post-hoc decomposition under the benchmark's ACTION-INDEPENDENT event law.

Replay the forecast's declared learned continuation on the same revealed event
sequence, starting with the actual selected root. Future events enter only one
at a time after each counterfactual decision. This is an evaluator, never advice.
For action-dependent opponents, reusing their observed responses is NOT valid.

realized_persona_gain compares risk-adjusted scalar path scores at the root.
Each path is scored alone: this does NOT compare the strongest-value eligibility
tier across alternatives and is not a complete ordering of the person's choices.
"""
import argparse,copy,json,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import compile_batch,digest
from reflex.intertemporal import Branch
from reflex.purpose_plan import goal_forecast
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from reflex.laboratory import PROFILES
from reflex.closed_loop import owned_rollout
from tools.continuing_games import ContinuingGame


def realized_score(g,own,c,root,actions,flows,endpoint):
    c=copy.deepcopy(c);c['actions']=[a for a in c['actions'] if a['id']==root]
    f=goal_forecast(c,{root:[Branch(1.,tuple(flows),(1.,)*len(flows))]},
        {root:[dict(actions=actions,terminal=g.terminal(endpoint),assessment=g.goal(endpoint).record())]},
        horizon=len(flows),unit='owner-decisions',target=g.target,policy=own.policy,
        settlement_weight='absolute',objective_origin=g.goal(g.world(c)),opportunity_regret=True)
    b=compile_batch(f.contexts,outcome_limit=256)
    return float(own.policy.decide(b,False,return_frame=f.return_frame).scores[0,0])


def audit_run(r,*,ticks=None):
    if r['arm'] not in ('endpoint','endpoint_posterior','endpoint_segmented','endpoint_persistent','receding','receding_posterior','receding_corrected','receding_recovery'):
        raise ValueError('decomposition requires the registered learned continuation')
    profile=next(p for p in PROFILES if p['id']==r['profile'])
    g=ContinuingGame(r['game'],profile,r['turns'],r['seed']%2);cases=[];eligible=0;incomplete=0
    for index,t in enumerate(r['trace']):
        if (index%16 if ticks is None else index not in ticks) or 'checkpoint' not in t or 'forecast_end' not in t or not t['planned']:continue
        eligible+=1
        stop=t['forecast_end'];segment=r['trace'][index:stop]
        if len(segment)!=stop-index:incomplete+=1;continue
        w=copy.deepcopy(t['before']);raw=g.context(w)
        segmented=t['checkpoint']['predictor']['memory']['format']=='segmented-events-v1'
        own=DecisionLoop.from_record(raw,t['checkpoint'],predictor=g.actor(raw,learning='segmented' if segmented else 'windows').predictor)
        assert digest(own.record())==t['before_owner']
        effective=own.preview(g.request(raw))[0];start=copy.deepcopy(own)
        future=None;future_calls=0
        if r['arm']=='receding_recovery':
            continuation=owned_rollout(start,observe=g.context,events=g.events,terminal=g.terminal,
                assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
                feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=r['tail_steps'],
                world_model='posterior',purpose_estimator='conditional',tail_recovery=2)
            future=continuation.frontier.future_planner
        actual_actions=[];actual_flows=[];reflex_actions=[];reflex_flows=[]
        post_root=(r['trace'][index+1].get('checkpoint') if index+1<len(r['trace']) else None)
        if post_root is not None:
            # The factual root is common to both continuations. Start from its
            # ACTUAL saved learner, rather than redoing Linux floating-point
            # feedback on a different platform and accepting an approximate hash.
            assert digest(post_root)==t['owner']
        for j,s in enumerate(segment):
            actual_actions.append(s['action']);actual_flows.append(g.transition(s['before'],s['action'],s['response'])[1])
            if g.terminal(w):continue
            if j==0 and post_root is not None:
                w=copy.deepcopy(s['after'])
                own=DecisionLoop.from_record(g.context(w),post_root,
                    predictor=g.actor(g.context(w),learning='segmented' if segmented else 'windows').predictor)
                assert digest(own.record())==t['owner']
                reflex_actions.append(s['action']);reflex_flows.append(actual_flows[-1]);continue
            c=g.context(w,own.state);req=g.request(c)
            planner=None
            if j and future is not None and future_calls<continuation.frontier.future_planner_limit:
                planner=future(copy.deepcopy(own),copy.deepcopy(req),copy.deepcopy(w))
                if planner is not None:future_calls+=1
            result=DecisionLoop.decide_batch([(own,req)],False,planner,
                _counterfactual_roots=(t['action'],) if j==0 else None)[0]
            key=result['decision']['action_id'];after,flow=g.transition(w,key,s['response'])
            resolve_feedback(own,result['ticket'],g.feedback(w,after,key,result['context'],flow))
            if j==0:assert digest(own.record())==t['owner'],f"root owner replay differs at {r['game']} {r['seed']} {r['regime']} tick {index}"
            reflex_actions.append(key);reflex_flows.append(flow);w=after
        predicted=t['predicted'];reference=g.goal(w).value;actual=segment[-1]['goal']
        # Score complete realized paths with the SAME decision-time person.
        # These ex-post scores are diagnostics, not expected-utility guarantees.
        ours=realized_score(g,start,effective,t['action'],actual_actions,actual_flows,segment[-1]['after'])
        theirs=realized_score(g,start,effective,t['action'],reflex_actions,reflex_flows,w)
        cases.append(dict(tick=index,root_owner='actual-checkpoint' if post_root is not None else 'exact-feedback-replay',
            reference_policy='two-step-recovery-when-no-immediate-gain' if future else 'learned-reflex',future_calls=future_calls,
            predicted=predicted,reflex_replay=reference,executed=actual,
            world_and_sampling_residual=reference-predicted,control_contribution=actual-reference,
            total_gap=actual-predicted,realized_persona_gain=ours-theirs,
            actual_actions=actual_actions,reference_actions=reflex_actions))
        assert abs((reference-predicted)+(actual-reference)-(actual-predicted))<1e-12
    return dict(**{k:r[k] for k in ('game','profile','regime','seed','arm','interval')},cases=cases,
        windows=dict(eligible=eligible,evaluated=len(cases),incomplete_actual_window=incomplete),
        summary={k:float(np.mean([c[k] for c in cases])) for k in
            ('world_and_sampling_residual','control_contribution','total_gap','realized_persona_gain')} if cases else {})


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rows=[];sources=[]
    for source in a.sources:
        path=Path(source);sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for line in path.read_text().splitlines():
            result=audit_run(json.loads(line));rows.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
    Path(a.output).write_text(json.dumps(dict(semantics=__doc__,sources=sources,runs=rows),indent=2)+'\n')


if __name__=='__main__':main()
