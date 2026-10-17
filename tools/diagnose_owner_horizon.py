"""Read-only trajectory instrumentation for a saved owner's finite continuation."""
import argparse,copy,hashlib,json,sys,time
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.closed_loop import owned_rollout
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame


def diagnose(row,tick,tail,recovery_limit=16):
    t=row['trace'][tick];assert t['tick']==tick
    g=ContinuingGame(row['game'],next(p for p in PROFILES if p['id']==row['profile']),row['turns'],row['seed']%2)
    w=copy.deepcopy(t['before']);c=g.context(w)
    own=DecisionLoop.from_record(c,t['checkpoint'],predictor=g.actor(c,learning='segmented').predictor)
    before=digest(own.record())
    ctrl=owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,
        assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
        feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=tail,
        world_model='posterior',purpose_estimator='conditional',tail_recovery=2,tail_recovery_limit=recovery_limit)
    frontier=ctrl.frontier;distribute=frontier.distribution;plan=frontier.future_planner;advance=frontier.owner_advance
    traces=[];active=None;pending=None
    def watched_plan(person,request,state):
        nonlocal pending
        assert pending is None
        result=plan(person,request,state)
        pending='planned' if result is not None else 'declined'
        return result
    def watched_advance(state,key,seed,person,context):
        nonlocal pending
        after,flow=advance(state,key,seed,person,context)
        active['paths'].setdefault(str(seed),[]).append(dict(tick=state['tick'],action=key,
            controller=pending or 'budget-skipped',before_goal=g.goal(state).value,after_goal=g.goal(after).value))
        pending=None
        return after,flow
    def watched_distribution(world,person):
        nonlocal active
        assert active is None and pending is None
        active=dict(start=copy.deepcopy(world),paths={});traces.append(active)
        result=distribute(world,person)
        active['endpoint_estimates']=result.purpose_estimates
        active=None;assert pending is None
        return result
    frontier.future_planner=watched_plan;frontier.owner_advance=watched_advance;frontier.distribution=watched_distribution
    req=g.request(c);start=time.monotonic();capture={};detached=copy.deepcopy(own)
    selected=DecisionLoop.decide_batch([(detached,req)],False,ctrl.planner(detached,req,w,captured=capture))[0]['decision']['action_id']
    assert capture['audit']['complete']
    forecast=capture['node']['forecast']
    assert digest(own.record())==before and own.pending is None,'diagnostic mutated factual owner'
    counts=Counter(s['controller'] for d in traces for steps in d['paths'].values() for s in steps)
    paths=[steps for d in traces for steps in d['paths'].values()]
    return dict(tick=tick,saved_action=t['action'],selected_action=selected,tail_steps=tail,recovery_limit=recovery_limit,
        seconds=time.monotonic()-start,controller_counts=dict(counts),paths=len(paths),
        capped_paths=sum(any(s['controller']=='budget-skipped' for s in steps) for steps in paths),
        first_skipped_tick=min((s['tick'] for steps in paths for s in steps if s['controller']=='budget-skipped'),default=None),
        distributions=traces,forecast=forecast.audit)


def main():
    p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('--profile',required=True)
    p.add_argument('--regime',required=True);p.add_argument('--tick',type=int,required=True)
    p.add_argument('--tail',type=int,default=63);p.add_argument('--recovery-limit',type=int,default=16)
    p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.study)/'trajectories.jsonl'
    row=next(x for x in map(json.loads,source.read_text().splitlines())
        if x['arm']=='receding_recovery' and x['profile']==a.profile and x['regime']==a.regime)
    result=diagnose(row,a.tick,a.tail,a.recovery_limit)
    result.update(game=row['game'],profile=row['profile'],regime=row['regime'],seed=row['seed'],
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        core_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve().parents[1]/'reflex').glob('*.py')},
        limitation='Read-only model continuation under the saved owner, not a realized outcome.')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('forecast','distributions','core_sources')}),flush=True)


if __name__=='__main__':main()
