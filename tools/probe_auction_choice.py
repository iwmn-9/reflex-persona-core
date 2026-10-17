"""Post-hoc selector diagnosis; it does not change any played candidate."""
import argparse,copy,json,hashlib,sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.closed_loop import owned_rollout
from reflex.decision_loop import DecisionLoop
from reflex.core import digest,compile_batch
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame

CASES=(('auction','care',401,'outside',14),('auction','care',548,'outside',14),
       ('resources','steady',401,'shift-return',21))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--sources',nargs='+',required=True)
    p.add_argument('--frozen',action='store_true');p.add_argument('--world-contrast',action='store_true')
    a=p.parse_args();root=Path(a.root);sources=[Path(s).resolve() for s in a.sources]
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        dispatch(root,Path(__file__).name,['--sources',*map(str,sources),*(['--world-contrast'] if a.world_contrast else [])]);return
    allrows=[r for source in sources for r in map(json.loads,source.read_text().splitlines())]
    output=[]
    for game,profile,seed,regime,tick in CASES:
        row=next(r for r in allrows if (r['game'],r['profile'],r['seed'],r['regime'],r['arm'])==(game,profile,seed,regime,'endpoint_posterior'))
        t=row['trace'][tick];g=ContinuingGame(game,next(p for p in PROFILES if p['id']==profile),row['turns'],seed%2)
        w=t['before'];c=g.context(w);own=DecisionLoop.from_record(c,t['checkpoint'],predictor=g.actor(c,learning='segmented').predictor)
        before=digest(own.record());rq=g.request(c);cap={}
        ctrl=owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),
            record=lambda w:w,request_for=g.request,feedback=g.feedback,model_id=g.model_id,target=g.target,
            tail_steps=15,world_model='posterior')
        clone=copy.deepcopy(own)
        result=DecisionLoop.decide_batch([(clone,rq)],False,ctrl.planner(own,rq,w,captured=cap))[0]
        assert result['decision']['action_id']==t['action']
        f=cap['node']['forecast'];b=compile_batch(f.contexts,outcome_limit=f.outcome_limit)
        scores=own.policy.decide(b,False,return_frame=f.return_frame)
        unconstrained=DecisionLoop.decide_batch([(copy.deepcopy(own),rq)],False,lambda cs,ds:replace(f,max_regret=2.))[0]
        assert digest(own.record())==before
        output.append(dict(game=game,profile=profile,seed=seed,regime=regime,tick=tick,observed_choice=t['action'],
            without_purpose_narrowing=unconstrained['decision']['action_id'],public_state=w,
            purpose={f.roots[k][0]:v for k,v in f.purpose.items()},
            persona_scores={f.roots[k][0]:float(v) for k,v in zip(b.ids[0],scores.scores[0])},
            purpose_allowance=f.max_regret,predictor=own.predictor.predict(c)))
        if a.world_contrast:
            output[-1]['world_contrast']={}
            for mode in ('frozen','persistent'):
                captured={}
                other=owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,
                    assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
                    feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=15,world_model=mode)
                decision=DecisionLoop.decide_batch([(copy.deepcopy(own),rq)],False,other.planner(own,rq,w,captured=captured))[0]
                forecast=captured['node']['forecast']
                output[-1]['world_contrast'][mode]=dict(choice=decision['decision']['action_id'],
                    purpose={forecast.roots[k][0]:v for k,v in forecast.purpose.items()})
            assert digest(own.record())==before
        print(json.dumps(output[-1]),flush=True)
    result=dict(cases=output,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        interpretation='Post-hoc same-forecast selector ablation and optional same-owner world-model contrast. This probe runs no alternate long trajectory, so a different choice does not establish better play. These cases were selected after observing losses.')
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
