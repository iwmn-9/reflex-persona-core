"""Fixed-candidate endpoint goals and observed preparation/release experiments.

Old flow evaluation, corrected goal evaluation, and goal+observed-release are
compared on matching public worlds. No hidden event schedule or actual RNG is
provided to a forecast. Real-only feedback is kept out of hypothetical memory.
"""
from dataclasses import replace,asdict
from pathlib import Path
import copy
import json
import subprocess
from .core import digest
from .payback_cycle import Probe
from .purpose_plan import Goal,goal_forecast,PersonaProgressWatch
from .flow_rollout import rollout
from .intertemporal import forecast
from .decision_loop import DecisionLoop,Request
from .judgment import Binding
from .pressure import NeedPressure,PressureConfig
from .progress import Activity,PurposeRequest,PurposeFeedback,ProgressConfig
from .laboratory import profiles
from .purpose_experiment import source_hashes


class GoalProbe(Probe):
    def __init__(self,spec,profile,corrected=True):
        self.corrected=corrected;self.route=spec.get('route','science')
        super().__init__(spec,profile)
        self.seeds=(820,821,822,823) if self.genre=='combat' else (0,)
        if self.genre=='resources' and self.route=='mixed':
            from .route_experiment import route_context
            from .routes import choose_route
            c,_=route_context(self.start(),profile,42,0)
            self.route=choose_route(c)[0].chosen
        self.target=self.genre+'-'+self.route if self.genre=='resources' else self.genre+'-goal'

    def start(self):
        w=super().start()
        if self.genre=='resources':
            from .resource_world import Empire
            if self.spec.get('opening')=='conversion':
                e={'science':Empire(stock=(8,8,5,9),buildings=(1,1,1,0,2,0),tech=2,science=14),
                   'culture':Empire(stock=(8,8,5,9),buildings=(1,1,1,0,0,2),monuments=1,culture=8),
                   'power':Empire(stock=(8,8,6,9),army=4,land=3)}[self.spec['route']]
                w=replace(w,empires=(e,)+w.empires[1:])
            w=replace(w,routes=tuple(('science','culture','power') if self.spec.get('route')=='mixed' else (self.spec['route'],)))
        if self.genre=='combat' and self.spec.get('distance')==4:
            w=replace(w,units=(replace(w.units[0],x=1),)+w.units[1:])
        return w

    def goal(self,w):
        if self.genre=='resources':
            from .resource_world import winners,terminal
            from .route_experiment import route_potential
            won=winners(w)
            if 0 in won:return Goal(self.target,'success',1.)
            if won:return Goal(self.target,'failure',-1.)
            if terminal(w):return Goal(self.target,'draw',0.)
            return Goal(self.target,'running',route_potential(w,0,self.route))
        if self.genre=='auction':
            value=w.scores[0]/sum(self.spec['prizes'])
            return Goal(self.target,'scored' if self.terminal(w) else 'running',value)
        from .combat import victory
        if 0 in victory(w):return Goal(self.target,'success',1.)
        if 1 in victory(w):return Goal(self.target,'failure',-1.)
        if self.terminal(w):return Goal(self.target,'draw',0.)
        return Goal(self.target,'running',.6*(1-w.units[3].hp/9))

    def correct(self,before,after,row):
        if self.corrected:
            row=copy.deepcopy(row);g=(self.goal(after).value-self.goal(before).value)/2
            row['objective']=g;row['values']['achievement']=g;row['needs']['growth']=g
        return row

    def advance(self,w,key,seed):
        after,row=super().advance(w,key,seed)
        return after,self.correct(w,after,row)

    def actual(self,w,key,seed):
        # Probe.actual dispatches through this advance for ordinary transitions;
        # rederive the objective once after any real public event as well.
        after,row=super().actual(w,key,seed)
        return after,self.correct(w,after,row)

    def observe(self,w,memory=None):
        c=super().observe(w,memory);c['objective']='本人に指定された '+self.target+' を期限までに達成する'
        if self.genre=='resources':c['facts']['chosen_route']=self.route
        return c

    def purpose(self,w,c,previous=None):
        if self.genre=='resources':
            from .progress_adapters import economy_request
            return economy_request(w,c,previous)
        if self.genre=='combat':
            from .progress_adapters import combat_request
            return combat_request(w,0,c)
        options={};future=bool(w.remaining and max(w.remaining)>w.prize)
        for a in c['actions']:
            root=a['id'];positive=any(r['objective']>0 for r in a['outcomes'])
            if root=='bid:0' and future:
                options[root]=Activity('wait','reserve-for-public-later-prize','public-auction-budget',min(2,len(w.remaining)),
                    'higher-valued-public-prize-arrives')
            else:options[root]=Activity('uncertain' if positive else 'idle','score-opportunity' if positive else 'no-current-scoring-effect')
        return PurposeRequest(self.goal(w).value,w.budgets[0]/18,options)

    def feedback(self,before,after,key):
        if self.genre=='resources':
            from .progress_adapters import economy_feedback
            return economy_feedback(before,after,0,key)
        if self.genre=='combat':
            from .progress_adapters import combat_feedback
            return combat_feedback(before,after,0,key)
        return PurposeFeedback(self.goal(after).value,after.budgets[0]/18,completed=self.terminal(after))


def run(spec,profile,variant,seed,horizon=6):
    watched=variant=='release';p=GoalProbe(spec,profile,variant!='old');w=p.start();c=p.observe(w)
    watch=PersonaProgressWatch(c['scope'],ProgressConfig(grace=2,repeat_limit=2)) if watched else None
    pressure=NeedPressure(c['scope'],PressureConfig(grace=2)) if watched else None
    loop=DecisionLoop(c,progress=watch,pressure=pressure);trace=[];previous=None
    while not p.terminal(w):
        c=p.observe(w,loop.state);pr=p.purpose(w,c,previous) if watched else None;captured={}
        def planner(cs,ds):
            paths,audit=rollout(cs[0],w,observe=p.observe,advance=p.advance,terminal=p.terminal,
                horizon=horizon,seeds=p.seeds,assess=lambda s:p.goal(s).record())
            captured.update(paths=paths,audit=audit)
            if variant=='old':return forecast(cs[0],paths,horizon=horizon,unit='public-turns',target=p.target)
            return goal_forecast(cs[0],paths,audit,horizon=horizon,unit='public-turns',target=p.target)
        req=Request(c,{k:Binding(k,'public',()) for k in p.keys(w)},{k:() for k in p.keys(w)},purpose=pr)
        r=DecisionLoop.decide_batch([(loop,req)],False,planner)[0];key=r['decision']['action_id']
        before=w;w,row=p.actual(w,key,seed);fb=p.feedback(before,w,key) if watched else None
        if watched:
            g=max(-1.,min(1.,p.goal(w).value-p.goal(before).value))
            loop.abandon(r['ticket'],need_progress={'growth':g},maintained=('growth',) if fb.maintained else (),
                completed=('growth',) if p.goal(w).status=='success' else (),purpose_feedback=fb)
        else:loop.abandon(r['ticket'])
        assert not loop.memory.entries and c['personality']==loop.personality and c['values']==loop.values
        trace.append(dict(before=asdict(before),after=asdict(w),root=key,flow=row,goal=p.goal(w).record(),
            purpose=None if pr is None else asdict(pr),feedback=None if fb is None else asdict(fb),
            progress=r.get('progress'),deliberation=r['deliberation'],state=copy.deepcopy(loop.state),
            watch=None if loop.progress is None else loop.progress.record(),
            pressure=None if loop.pressure is None else loop.pressure.record(),
            persona_hash=digest([c['personality'],c['values']]),
            endpoints=captured['audit'].get(key)))
        previous=before
    goals=p.goal(w);reversals=0
    if p.genre=='combat':
        for i in range(len(trace)-2):
            a=trace[i]['after']['units'][0];b=trace[i+2]['after']['units'][0]
            reversals+=int((a['x'],a['y'])==(b['x'],b['y']) and trace[i+1]['root'].startswith('move:') and trace[i+2]['root'].startswith('move:'))
    return dict(spec=spec,profile=profile['id'],variant=variant,seed=seed,route=p.route,target=p.target,
        status=goals.status,goal_value=goals.value,steps=len(trace),actions=[t['root'] for t in trace],
        surviving=w.units[0].hp>0 if p.genre=='combat' else None,
        shortages=w.empires[0].shortages if p.genre=='resources' else None,move_reversals=reversals,
        progress_applied=sum(bool((t['progress'] or {}).get('applied')) for t in trace),
        progress_unresolved=sum(bool((t['progress'] or {}).get('unresolved')) for t in trace),trace=trace)


def cases(split):
    for route in ('science','culture','power'):
        yield dict(genre='resources',route=route,opening='conversion',limit=5 if split=='control' else 6)
    yield dict(genre='resources',route='mixed',limit=8 if split=='control' else 10,famine=True)
    yield dict(genre='auction',prizes=[8,10] if split=='control' else [7,11],budget=6)
    yield dict(genre='auction',prizes=[8,10,6] if split=='control' else [9,12,7],budget=10,rival_shift=True)
    for hp in (3,6):
        yield dict(genre='combat',hp=hp,limit=8 if split=='control' else 9,distance=2 if split=='control' else 4)
    yield dict(genre='combat',hp=6,limit=1,distance=2 if split=='control' else 4)


def experiment(output,progress=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any((output/n).exists() for n in ('preregister.json','trajectories.jsonl','evaluation.json')):
        raise FileExistsError('frozen purpose/release evidence exists; use a fresh directory')
    fixed=source_hashes();ps=profiles();splits={s:list(cases(s)) for s in ('control','confirmation')};runs=[]
    try:base=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):base=None
    prereg=dict(format='purpose-recovery-v1',base=base,
        source_hashes=fixed,cases=splits,profiles=ps,variants=['old','goal','release'],horizon=6,
        model_combat_seeds=[820,821,822,823],actual_seeds={'control':[200,201],'confirmation':[210,211]},
        parameters=dict(max_regret=.15,grace=2,repeat_limit=2,pressure_grace=2),
        status='functional control fixtures explored in smoke tests; batch comparisons and confirmation predeclared and source-frozen; no result-driven tuning; not representative game strength')
    (output/'preregister.json').write_text(json.dumps(prereg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    with (output/'trajectories.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        for split,ss in splits.items():
            for spec in ss:
                for p in ps:
                    for seed in prereg['actual_seeds'][split] if spec['genre']=='combat' else prereg['actual_seeds'][split][:1]:
                        for v in prereg['variants']:
                            r=run(spec,p,v,seed);r['split']=split;stream.write(json.dumps(r,ensure_ascii=False)+'\n')
                            runs.append({k:v for k,v in r.items() if k!='trace'})
                if progress:progress(f'{split} {spec}: {len(runs)} complete episodes',flush=True)
    assert fixed==source_hashes();summary=[]
    for split in splits:
        for genre in ('resources','auction','combat'):
            for v in prereg['variants']:
                rs=[r for r in runs if r['split']==split and r['spec']['genre']==genre and r['variant']==v]
                summary.append(dict(split=split,genre=genre,variant=v,episodes=len(rs),
                    successes=sum(r['status']=='success' for r in rs),failures=sum(r['status']=='failure' for r in rs),
                    goal_sum=sum(r['goal_value'] for r in rs),surviving=sum(r['surviving'] is True for r in rs),
                    shortages=sum(r['shortages'] or 0 for r in rs),move_reversals=sum(r['move_reversals'] for r in rs),
                    progress_applied=sum(r['progress_applied'] for r in rs),unresolved=sum(r['progress_unresolved'] for r in rs)))
    result=dict(format=prereg['format'],episodes=len(runs),summary=summary,runs=runs,source_hashes=fixed,
        preregister_digest=digest(prereg),default_policy_changed=False,training_performed=False,cloud_runtime_used=False)
    (output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    return result
