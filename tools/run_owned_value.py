"""One persistent world: learned owner + endpoint evaluation + real replanning.

The hidden response law is used ONLY by the real world and diagnostic scores.
All control parameters are shared across the four rule families.
"""
import argparse,copy,json,sys,time,platform
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest,character_key
from reflex.laboratory import PROFILES
from reflex.closed_loop import owned_rollout
from reflex.purpose_step import purpose_step
from reflex.receding_owner import RecedingOwner
from reflex.actor_model import ActorModel
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from reflex.flow_rollout import rollout
from reflex.intertemporal import forecast
from tools.continuing_games import ContinuingGame,GAMES


def sample(p,u):
    total=0.
    for k,v in p.items():
        total+=v
        if u<total:return k
    return list(p)[-1]


def law(t,turns,regime):
    stage=min(3,4*t//turns)
    if regime=='irregular':
        stage=sum(t>=int(turns*f) for f in (.17,.43,.69,.91))
        top=(0,2,1,0,2)[stage]
    elif regime in ('steady','outside','shift-return'):
        top=0 if regime=='steady' else (2 if regime=='outside' else (0,2,1,0)[stage])
    else:raise ValueError('registered response regime required')
    return {str(i):(.8 if i==top else .1) for i in range(3)}


def cumulative_basis(game,enabled):
    if type(enabled) is not bool:raise ValueError('explicit cumulative achievement mode required')
    if not enabled:return None
    from reflex.cumulative_preference import CumulativePreference
    return CumulativePreference(game.target,'public-score-per-decision',(('values.achievement',float(game.turns)),))


def run(job,arm,*,checkpoint=None,max_decisions=None,checkpoint_every=8,tail_recovery_limit=16,model_seeds=(701,702,703),sampling_standard_errors=None,cumulative_achievement=False,tail_goal_refinement=False):
    if type(tail_recovery_limit) is not int or not 1<=tail_recovery_limit<=256:raise ValueError('bounded recovery budget required')
    if tail_recovery_limit!=16 and arm!='receding_recovery':raise ValueError('recovery budget requires the recovery arm')
    if (not isinstance(model_seeds,tuple) or not 1<=len(model_seeds)<=32
        or any(type(s) is not int for s in model_seeds) or len(set(model_seeds))!=len(model_seeds)):
        raise ValueError('explicit independent model seeds required')
    if model_seeds!=(701,702,703) and arm not in ('receding','receding_posterior','receding_corrected','receding_recovery'):
        raise ValueError('model seed override requires a receding owner')
    if sampling_standard_errors is not None and arm not in ('receding_corrected','receding_recovery'):
        raise ValueError('sampling margin requires a conditional receding owner')
    if type(cumulative_achievement) is not bool or cumulative_achievement and arm!='receding_recovery':
        raise ValueError('explicit cumulative achievement on the registered recovery owner required')
    configuration={}
    if type(tail_goal_refinement) is not bool or tail_goal_refinement and arm!='receding_recovery':
        raise ValueError('explicit goal refinement on the recovery owner required')
    if tail_goal_refinement:configuration['tail_goal_refinement']=True
    if tail_recovery_limit!=16:configuration['tail_recovery_limit']=tail_recovery_limit
    if model_seeds!=(701,702,703):configuration['model_seeds']=list(model_seeds)
    if sampling_standard_errors is not None:configuration['sampling_standard_errors']=sampling_standard_errors
    if cumulative_achievement:configuration['cumulative_achievement']=True
    configuration=configuration or None
    game,pid,seed,regime,turns,interval,tail_steps=job[:7]
    offset=job[7] if len(job)>7 else 0
    is_reflex=arm in ('reflex','reflex_segmented')
    g=ContinuingGame(game,PROFILES[pid],turns,seed%2);w=g.start();c=g.context(w)
    cumulative=cumulative_basis(g,cumulative_achievement)
    own=g.actor(c,learned=arm!='endpoint_library',renewal=arm in ('endpoint_renewal','endpoint_both','endpoint_recovery'),
        learning='segmented' if arm in ('endpoint_segmented','endpoint_posterior','endpoint_persistent','endpoint_unpruned','reflex_segmented','competent','endpoint_competent','receding_posterior','receding_corrected','receding_recovery') else 'windows')
    if arm in ('receding','receding_posterior','receding_corrected','receding_recovery') and (interval!=1 or offset!=0):
        raise ValueError('receding owner requires every decision; no periodic substitution')
    axes=character_key(c);rng=np.random.default_rng(seed);trace=[];checks=Counter();started=time.monotonic()
    start_tick=0;storage=None
    if checkpoint is not None:
        from tools.owned_checkpoint import OwnedCheckpoint
        storage=OwnedCheckpoint(checkpoint,job,arm,max_decisions=max_decisions,every=checkpoint_every,configuration=configuration)
        if storage.saved is not None:
            saved=storage.saved;w=saved['world'];trace=saved['trace'];checks=Counter(saved['checks'])
            # The same post-decision owner restoration already used below.
            # The preceding live context also works at a terminal boundary.
            c=g.context(trace[-1]['before'])
            own=DecisionLoop.from_record(c,saved['owner'],predictor=own.predictor)
            assert own.record()==saved['owner']
            rng.bit_generator.state=saved['rng'];start_tick=w['tick']
            started=time.monotonic()-saved['elapsed']
    elif max_decisions is not None:raise ValueError('a decision slice requires a persistent checkpoint')
    for tick in range(start_tick,turns):
        if g.terminal(w):break
        c=g.context(w,own.state);req=g.request(c);before=digest(own.record())
        p=own.predictor.predict(c);actual=law(tick,turns,regime)
        due=tick%interval==offset
        cap={};planner=None;checkpoint=own.record() if not is_reflex and due else None
        if not is_reflex and due:
            if arm in ('endpoint','short_endpoint','endpoint_library','endpoint_renewal','endpoint_stratified','endpoint_both','endpoint_conditional','endpoint_recovery','endpoint_dense','endpoint_segmented','endpoint_posterior','endpoint_persistent','endpoint_unpruned','competent','endpoint_competent'):
                def future(person,request,state):
                    return purpose_step(person,state,events=g.events,terminal=g.terminal,
                        assess=lambda w:g.goal(w).record(),target=g.target)
                ctrl=owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,
                    assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
                    feedback=g.feedback,model_id=g.model_id,target=g.target,
                    tail_steps=0 if arm=='competent' else 3 if arm=='short_endpoint' else tail_steps,
                    future_planner=future if arm=='endpoint_competent' else None,
                    future_planner_id='purpose-persona-one-step' if arm=='endpoint_competent' else None,
                    future_planner_limit=16,
                    seeds=tuple(range(701,713)) if arm=='endpoint_dense' else (701,702,703),
                    sampling='stratified' if arm in ('endpoint_stratified','endpoint_both') else 'independent',
                    purpose_estimator='conditional' if arm in ('endpoint_conditional','endpoint_recovery') else 'sampled',
                    world_model='posterior' if arm in ('endpoint_posterior','endpoint_unpruned','endpoint_competent') else 'persistent' if arm=='endpoint_persistent' else 'frozen',
                    max_regret=1. if arm=='endpoint_unpruned' else .15)
                planner=ctrl.planner(own,req,w,captured=cap)
                if arm=='competent':
                    planner=purpose_step(own,w,events=g.events,terminal=g.terminal,
                        assess=lambda w:g.goal(w).record(),target=g.target,captured=cap)
            elif arm=='flow':
                def planner(cs,ds):
                    model=ActorModel(own,root_request=req,request_for=g.request,
                        feedback=lambda b,e,k,cc:g.feedback(b,e,k,cc,g.transition(b,k,e['signal'])[1]))
                    def advance(s,k,seed):
                        u=np.random.default_rng(np.random.SeedSequence([seed,s['tick']])).random()
                        return g.transition(s,k,sample(p,u))
                    paths,audit=rollout(cs[0],w,observe=g.context,advance=advance,terminal=g.terminal,
                        horizon=3,seeds=(701,702,703),observer=model,policy=own.policy)
                    cap['flow']=True
                    return forecast(cs[0],paths,horizon=3,unit='owner-decisions',target='incremental-purpose')
        mirror=ActorModel(own,root_request=req,request_for=g.request,
            feedback=lambda b,e,k,cc:g.feedback(b,e,k,cc,g.transition(b,k,e['signal'])[1]))
        prepared,mreq,_,_=mirror.prepare(w,c,prepared=True);mirror.preview()
        if arm in ('receding','receding_posterior','receding_corrected','receding_recovery'):
            live=RecedingOwner(own,observe=g.context,events=g.events,terminal=g.terminal,
                assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
                feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=tail_steps,
                world_model='frozen' if arm=='receding' else 'posterior',
                purpose_estimator='conditional' if arm in ('receding_corrected','receding_recovery') else 'sampled',
                tail_recovery=2 if arm=='receding_recovery' else 0,tail_recovery_limit=tail_recovery_limit,seeds=model_seeds,
                sampling_standard_errors=sampling_standard_errors,cumulative_preference=cumulative,tail_goal_refinement=tail_goal_refinement)
            result=live.decide(w);cap=live.last
        else:result=DecisionLoop.decide_batch([(own,req)],False,planner)[0]
        key=result['decision']['action_id']
        mirror.commit_choice(key);mirror.adopt_state(result['decision']['next_state'])
        end,flow=g.transition(w,key,sample(actual,rng.random()))
        resolve_feedback(own,result['ticket'],g.feedback(w,end,key,c,flow));mirror.advance(w,end,key,prepared,mreq)
        assert own.record()==mirror.actor.record();checks['owner_parity']+=1
        assert character_key(c)==axes;checks['fixed_axes']+=1
        if tick%23==0:
            record=json.loads(json.dumps(own.record()));own=DecisionLoop.from_record(c,record,predictor=own.predictor)
            assert own.record()==record;checks['restore']+=1
        diag=result.get('deliberation',{})
        trace.append(dict(tick=tick,before=w,after=end,action=key,response=end['signal'],
            goal=g.goal(end).value,law=p,truth=actual,mode=result['decision']['next_state']['mode'],
            brier=sum((p[z]-actual[z])**2 for z in p),owner=digest(own.record()),
            planned=bool(diag.get('adopted')),predicted=diag.get('selected_purpose'),
            before_owner=before,needs=own.pressure.debt,
            forecast={}))
        if checkpoint is not None:trace[-1]['checkpoint']=checkpoint
        if 'node' in cap:
            node=cap['node'];f=node['forecast']
            trace[-1]['forecast']={f.roots[k][0]:v for k,v in f.purpose.items()}
            trace[-1]['future_coverage']=[v['settled_fraction'] for v in node.get('frontiers',[])]
            trace[-1]['forecast_end']=min(turns,tick+f.horizon)
            if 'purpose_mean_estimator' in f.audit:trace[-1]['mean_estimator']=f.audit['purpose_mean_estimator']
            if 'purpose_sampling' in diag:trace[-1]['purpose_sampling']=diag['purpose_sampling']
            if 'cumulative_preference' in f.audit:trace[-1]['cumulative_preference']=f.audit['cumulative_preference']
        w=end
        if storage is not None:
            storage.save(w,own,rng,trace,checks,time.monotonic()-started,terminal=g.terminal(w))
            if storage.stopping(len(trace)) and not g.terminal(w):return None
    for t in trace:
        if t.get('predicted') is not None and 'forecast_end' in t:
            end=trace[min(t['forecast_end'],len(trace))-1]
            t['realized_at_forecast_end']=end['goal']
            t['forecast_error']=t['predicted']-end['goal']
    return dict(game=game,profile=PROFILES[pid]['id'],seed=seed,regime=regime,arm=arm,
        turns=turns,executed=len(trace),interval=interval,tail_steps=tail_steps,offset=offset,
        score=g.goal(w).value,final=w,checks=dict(checks),trace=trace,seconds=time.monotonic()-started,
        **(configuration or {}))


def evaluate(job,arms):return [run(job,arm) for arm in arms]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true')
    p.add_argument('--turns',type=int,default=128);p.add_argument('--interval',type=int,default=8)
    p.add_argument('--tail-steps',type=int,default=15);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--stage',choices=('development','confirmation','diagnostic','stress'),default='development')
    p.add_argument('--games',nargs='+',default=GAMES);p.add_argument('--profiles',nargs='+',type=int,default=[0,1,2,3])
    p.add_argument('--regimes',nargs='+',default=['steady','outside','shift-return'])
    p.add_argument('--seeds',nargs='+',type=int,default=[43]);p.add_argument('--arms',nargs='+',default=['reflex','flow','endpoint'])
    a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        args=[]
        for k in ('turns','interval','tail_steps','workers','offset','stage'):args+=['--'+k.replace('_','-'),str(getattr(a,k))]
        for k in ('games','profiles','regimes','seeds','arms'):args+=['--'+k,*map(str,getattr(a,k))]
        dispatch(root,Path(__file__).name,args);return
    root.mkdir(parents=True,exist_ok=True)
    (root/'environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,
        system=platform.system(),release=platform.release(),machine=platform.machine()),indent=2)+'\n',encoding='utf8')
    if not 0<=a.offset<a.interval:raise ValueError('planning offset must be within the interval')
    jobs=[(g,i,s,r,a.turns,a.interval,a.tail_steps,a.offset) for g in a.games for i in a.profiles for s in a.seeds for r in a.regimes]
    (root/'registration.json').write_text(json.dumps(dict(jobs=jobs,arms=a.arms,
        world_seeds=a.seeds,stage=a.stage,semantics=__doc__),indent=2),encoding='utf8')
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(root/'trajectories.jsonl').open('w',encoding='utf8') as f:
        fs={pool.submit(run,j,arm):(j,arm) for j in jobs for arm in a.arms}
        for fu in as_completed(fs):
            row=fu.result();rows.append(row);f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
            print(json.dumps(dict(done=len(rows),total=len(jobs)*len(a.arms),job=fs[fu],score=row['score'])),flush=True)
    (root/'summary.json').write_text(json.dumps([{k:v for k,v in row.items() if k!='trace'} for row in rows],indent=2),encoding='utf8')


if __name__=='__main__':main()
