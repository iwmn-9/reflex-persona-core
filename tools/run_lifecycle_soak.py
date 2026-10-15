"""Long-lived owned controllers across repeated bouts of four existing rules.

This tests lifecycle composition, NOT a new game-specific policy. A public
response remains visible when the immediate numerical effect is censored.
Both arms get the same observations and exogenous random uniforms. Legacy
coupling drops that response on censored turns; independent channels retain it.
Physical rounds reset; the owner, evidence, pressure and progress do not.
"""
import argparse,copy,json,sys,time
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import Policy,character_key,digest,TRAITS,NEEDS,FEATURES
from reflex.examples import context,action
from reflex.decision_loop import DecisionLoop,Request
from reflex.judgment import Binding,ReadControl
from reflex.loop_predictor import CategoricalReader
from reflex.actor_model import ActorModel
from reflex.model_observer import ModelFeedback,resolve_feedback
from reflex.pressure import NeedPressure
from reflex.progress import ProgressWatch,PurposeRequest,PurposeFeedback,Activity
from reflex.planning import vector
from reflex.flow_rollout import rollout
from reflex.intertemporal import forecast
from reflex.laboratory import PROFILES
from tools.open_model_games import OpenGame

GAMES=('auction','resources','combat','gambling')


def public_context(c):
    return json.loads(''.join(c['facts'][k] for k in sorted(c['facts']) if k.startswith('public_')))


class Adapter:
    def __init__(self,game,profile,variant):
        self.rules=OpenGame(game,profile,variant=variant)
        self.profile=profile;self.game=game
        self.scope=dict(game='lifecycle-'+game,episode='continuous-owner',npc='owner')

    def start(self,round_id=0,tick=0):
        return dict(world=self.rules.start(round_id).public,tick=tick,round=round_id)

    def terminal(self,s):return self.rules.terminal(SimpleNamespace(public=s['world']))
    def keys(self,s):return self.rules.keys(SimpleNamespace(public=s['world']))
    def goal(self,s):return self.rules.goal(SimpleNamespace(public=s['world'])).value
    def symbols(self):return tuple(str(i) for i in range(4 if self.game=='auction' else 3))

    def transition(self,s,key,symbol):
        w=s['world'];legal=self.rules.symbols(w)
        # The event is the actually revealed capped bid, never hidden intent.
        realized=str(min(int(symbol),len(legal)-1))
        event=next(e for e in self.rules.events(w,key) if e.symbol==realized)
        return dict(world=event.public,tick=s['tick']+1,round=s['round']),event.flow,realized

    def raw(self,s,state=None):
        rows=[]
        for k in self.keys(s):
            rows.append(action(k,*[dict(self.transition(s,k,z)[1],p=1/len(self.symbols())) for z in self.symbols()]))
        c=context('lifecycle',rows,{'growth':.25,'safety':.2},self.profile['values'],dict(zip(TRAITS,self.profile['traits'])),mode=None)
        encoded=json.dumps(s,sort_keys=True,separators=(',',':'))
        c.update(scope=copy.deepcopy(self.scope),tick=s['tick'],facts={f'public_{i:03d}':encoded[i:i+128] for i in range(0,len(encoded),128)})
        for k in NEEDS:
            if k not in ('growth','safety'):c['needs'][k]=dict(enabled=False,supported=False,deficit=None)
        if state is not None:c['state']=copy.deepcopy(state)
        return c

    def outcome(self,c,k,z):return self.transition(public_context(c),k,z)[1]

    def models(self,c):
        syms=self.symbols()
        return {z:{r:(.85 if z==r else .15/(len(syms)-1)) for r in syms} for z in syms}

    def readiness(self,s):
        w=s['world']
        if self.game=='combat':return w['hp']/4
        if self.game=='resources':return w['plant']/2
        return 0.

    def request(self,c):
        s=public_context(c);acts={}
        condition=digest({k:v for k,v in s['world'].items() if k not in ('tick','episode','previous','signal')})[:24]
        for k in self.keys(s):
            # Actual event labels differ even on a safe action; compare physical
            # effect, not the signal field, when declaring a deterministic means.
            deterministic=len({digest(self.transition(s,k,z)[1]) for z in self.symbols()})==1
            acts[k]=Activity('attempt' if deterministic else 'uncertain','public-effect',condition='physical-'+condition)
        return Request(c,{k:Binding(k,'public-'+condition,FEATURES) for k in acts},
            {k:() for k in acts},threatened=True,purpose=PurposeRequest(self.goal(s),self.readiness(s),acts))

    def feedback(self,before,after,key,c,arm,visibility):
        symbol=after['world']['signal'];row=self.transition(before,key,symbol)[1]
        visible=visibility=='complete' or before['tick']%4==0
        needs={k:float(row.get('needs',{}).get(k,0.)) for k in ('growth','safety')}
        return ModelFeedback(PurposeFeedback(self.goal(after),self.readiness(after),completed=self.terminal(after)),
            needs=needs,maintained=('safety',) if needs['safety']==0 else (),
            observed=vector(row) if visible else None,
            event={'revealed_action':symbol} if visible or arm=='channels' else None)

    def actor(self,c):
        reader=CategoricalReader(c['scope'],self.symbols(),self.models,self.outcome,known_conditionals=True)
        return DecisionLoop(c,policy=Policy(principle_priority='personal',commitments={character_key(c):.5}),
            predictor=reader,read_control=ReadControl(max_nodes=64),pressure=NeedPressure(c['scope']),progress=ProgressWatch(c['scope']))


def distribution(symbols,tick,turns,schedule):
    stage=min(3,tick*4//turns)
    main=0 if schedule=='steady' or stage in (0,3) else (len(symbols)-1 if stage==1 else 1)
    return {s:(.85 if int(s)==main else .15/(len(symbols)-1)) for s in symbols}


def sample(p,u):
    total=0.
    for k,v in p.items():
        total+=v
        if u<total:return k
    return list(p)[-1]


def run_arm(job,arm,*,capture=None,capture_ticks=(),adapter_factory=Adapter):
    game,pid,seed,schedule,turns,visibility,plan_every=job
    a=adapter_factory(game,PROFILES[pid],seed%2);s=a.start();c=a.raw(s);actor=a.actor(c)
    identity=character_key(c);initial_preferences=copy.deepcopy((c['personality'],c['values']))
    trace=[];scores=[];checks=Counter();true_observed=0;plan_decisions=0
    rng=np.random.default_rng(seed);uniforms=rng.random(turns)
    for tick in range(turns):
        if a.terminal(s):scores.append(a.goal(s));s=a.start(s['round']+1,tick)
        c=a.raw(s,actor.state);req=a.request(c);before_record=digest(actor.record())
        captured_owner=copy.deepcopy(actor) if capture is not None and tick in capture_ticks else None
        assert character_key(c)==identity and (c['personality'],c['values'])==initial_preferences
        fb=lambda b,e,k,cc:a.feedback(b,e,k,cc,arm,visibility)
        forecast_info={}
        def planner(cs,ds):
            nonlocal plan_decisions
            model=ActorModel(actor,root_request=req,request_for=a.request,feedback=fb)
            # Law is frozen from PUBLIC evidence at the root. Future owners
            # learn/reconsider within branches; no future real RNG is exposed.
            prior,candidate=actor.predictor.forecasts(c)
            law=prior
            def advance(w,k,model_seed):
                u=np.random.default_rng(np.random.SeedSequence([model_seed,w['tick']])).random()
                e,row,_=a.transition(w,k,sample(law,u));return e,row
            paths,audit=rollout(cs[0],s,observe=a.raw,advance=advance,terminal=a.terminal,
                horizon=3,seeds=(1801,2702,3603),observer=model,policy=actor.policy)
            assert digest(actor.record())==before_record,'planning contaminated real memory'
            checks['isolated_plans']+=1
            plan_decisions+=sum(len(t['actions']) for rows in audit.values() for t in rows)
            forecast_info['actions']={k:[r['actions'] for r in rows] for k,rows in audit.items()}
            return forecast(cs[0],paths,horizon=3,unit='decisions',target='incremental-purpose')
        # A detached execution of the realized root must resolve to the same
        # complete owner, even if the root was selected by a longer forecast.
        mirror=ActorModel(actor,root_request=req,request_for=a.request,feedback=fb)
        prepared,mreq,_,_=mirror.prepare(s,c,prepared=True);preview=mirror.preview()
        result=DecisionLoop.decide_batch([(actor,req)],False,planner if plan_every and tick%plan_every==0 else None)[0]
        key=result['decision']['action_id'];mirror.commit_choice(key)
        # Planning changes intent/age; retain the actually selected root state.
        mirror.adopt_state(result['decision']['next_state'])
        prior,candidate=actor.predictor.forecasts(c)
        dist=distribution(a.symbols(),tick,turns,schedule)
        after,row,symbol=a.transition(s,key,sample(dist,uniforms[tick]))
        feedback=fb(s,after,key,c)
        true_observed+=int(feedback.observed is not None)
        resolve_feedback(actor,result['ticket'],feedback)
        mirror.advance(s,after,key,prepared,mreq)
        assert digest(actor.record())==digest(mirror.actor.record()),'model/execution owner drift'
        checks['same_feedback_same_owner']+=1
        if tick%29==0:
            record=json.loads(json.dumps(actor.record()))
            restored=DecisionLoop.from_record(c,record,predictor=actor.predictor)
            assert digest(record)==digest(restored.record()),'checkpoint drift'
            actor=restored;checks['checkpoint_roundtrips']+=1
        obs=actor.predictor.tracker.observations
        assert obs==(tick+1 if visibility=='complete' or arm=='channels' else true_observed)
        # Bounded storage includes evictions; audit total on each update below.
        assert len(actor.memory.entries)<=actor.memory.capacity
        checks['preference_and_observation_contracts']+=1
        brier=lambda p:sum((v-float(k==symbol))**2 for k,v in p.items())/len(p)
        trace.append(dict(tick=tick,round=s['round'],step=s['world']['tick'],action=key,response=symbol,
            effect_visible=feedback.observed is not None,predictor_observations=obs,goal=a.goal(after),
            response_brier=brier(prior),candidate_brier=brier(candidate),need_debt=actor.pressure.debt,
            progress_stalls=actor.progress.stalls,mode=result['decision']['next_state']['mode'],
            learned_entries=len(actor.memory.entries),evidence={k:v['state'] for k,v in actor.evidence.entries.items()},
            owner_digest=digest(actor.record()),plan=forecast_info.get('actions')))
        if captured_owner is not None:capture(copy.deepcopy(s),copy.deepcopy(c),captured_owner,copy.deepcopy(result),copy.deepcopy(trace[-1]))
        s=after
    if a.terminal(s):scores.append(a.goal(s))
    return dict(game=game,profile=PROFILES[pid]['id'],seed=seed,schedule=schedule,arm=arm,visibility=visibility,
        turns=turns,completed_rounds=len(scores),score=float(np.mean(scores)) if scores else None,
        scores=scores,response_brier=float(np.mean([r['response_brier'] for r in trace])),
        checks=dict(checks),modeled_decisions=plan_decisions,trace=trace)


def run_job(job):
    left=run_arm(job,'coupled');right=run_arm(job,'channels')
    if job[5]=='complete':
        assert left['trace']==right['trace'],'complete observations must preserve behavior'
    return left,right


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['game'],r['profile'],r['seed'],r['schedule'],r['visibility'])].append(r)
    pairs=[]
    for key,p in groups.items():
        left=next(r for r in p if r['arm']=='coupled');right=next(r for r in p if r['arm']=='channels')
        delta=right['score']-left['score']
        pairs.append(dict(game=key[0],profile=key[1],seed=key[2],schedule=key[3],visibility=key[4],
            score_delta=delta,response_brier_delta=right['response_brier']-left['response_brier'],
            changes=sum(a['action']!=b['action'] for a,b in zip(left['trace'],right['trace']))))
    return dict(runs=len(rows),decisions=sum(r['turns'] for r in rows),
        modeled_decisions=sum(r['modeled_decisions'] for r in rows),pairs=pairs,
        checks={k:sum(r['checks'].get(k,0) for r in rows) for k in rows[0]['checks']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true')
    p.add_argument('--turns',type=int,default=240);p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--plan-every',type=int,default=12)
    p.add_argument('--pilot',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        args=['--turns',str(a.turns),'--seeds',str(a.seeds),'--workers',str(a.workers),'--plan-every',str(a.plan_every)]
        if a.pilot:args+=['--pilot']
        dispatch(root,Path(__file__).name,args);return
    root.mkdir(parents=True,exist_ok=True)
    jobs=[(g,p,s,mode,a.turns,visibility,a.plan_every) for g in GAMES for p in range(4) for s in range(a.seeds)
          for mode in ('steady','shift-return') for visibility in ('censored','complete')]
    if a.pilot:jobs=[(g,0,0,'shift-return',a.turns,'censored',a.plan_every) for g in GAMES]
    (root/'registration.json').write_text(json.dumps(dict(jobs=jobs,semantics=__doc__),indent=2)+'\n',encoding='utf-8')
    rows=[];started=time.monotonic()
    with (root/'trajectories.jsonl').open('w',encoding='utf-8') as f,ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(run_job,j):j for j in jobs}
        for i,future in enumerate(as_completed(futures)):
            result=future.result()
            for r in result:f.write(json.dumps(r,separators=(',',':'))+'\n');rows.append(r)
            f.flush();print(json.dumps(dict(done=i+1,total=len(jobs),job=futures[future],seconds=round(time.monotonic()-started,1))),flush=True)
    (root/'summary.json').write_text(json.dumps(summarize(rows),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
