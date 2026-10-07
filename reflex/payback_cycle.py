"""Closed-loop payback development probes on three existing exact rule engines.

Every real decision is recomputed. No forced commitment, hidden event schedule,
actual RNG, or delayed model payoff enters empirical learning. Public changes
are observed after they happen. This is not a representative strength benchmark.
"""
from dataclasses import replace, asdict
from pathlib import Path
import copy
import json
import subprocess
from .core import TRAITS, digest
from .examples import context, action, effect
from .intertemporal import forecast
from .flow_rollout import rollout
from .decision_loop import DecisionLoop, Request
from .judgment import Binding
from .laboratory import profiles
from .purpose_experiment import source_hashes


class Probe:
    """Game-owned evaluation and public transition adapters; no core rules."""
    def __init__(self,spec,profile):
        self.spec=spec;self.profile=copy.deepcopy(profile);self.genre=spec['genre']
        self.seeds=(800,801) if self.genre=='combat' else (0,)
        # Observation identity must not encode unrevealed event treatments.
        self.name=digest([self.genre,asdict(self.start()),profile['id']])[:16]

    def start(self):
        if self.genre=='resources':
            from .resource_world import World
            return World.start(limit=self.spec['limit'])
        if self.genre=='auction':
            from .contests import Public
            ps=self.spec['prizes'];b=self.spec['budget']
            return Public('auction',((),)*4,(0,)*4,(b,)*4,ps[0],'science',tuple(ps[1:]))
        from .combat import Battle,Unit
        return Battle((Unit(0,3,2,hp=self.spec['hp']),Unit(0,0,0,hp=0),Unit(0,0,4,hp=0),
            Unit(1,5,2),Unit(1,8,0,hp=0),Unit(1,8,4,hp=0)),goal='eliminate',limit=self.spec['limit'])

    def terminal(self,w):
        if self.genre=='resources':
            from .resource_world import terminal
            return terminal(w)
        if self.genre=='auction':return w.round>=len(self.spec['prizes'])
        from .combat import terminal
        return terminal(w) or w.units[0].hp<=0

    def keys(self,w):
        if self.genre=='resources':
            from .resource_world import legal
            return legal(w)
        if self.genre=='auction':return tuple('bid:'+str(b) for b in w.legal(0))
        from .combat import legal
        return legal(w,0)

    def score(self,w):
        if self.genre=='resources':
            e=w.empires[0];return (sum(e.stock)+e.science+e.culture)/100
        if self.genre=='auction':return w.scores[0]/30
        from .combat import potential
        return potential(w,0,'eliminate')/2

    def advance(self,w,key,seed):
        if self.genre=='resources':
            from .resource_world import step,terminal,consequence
            after=step(w,key)
            while not terminal(after) and after.turn!=0:after=step(after,'wait')
            row=consequence(w,after,0);g=self.score(after)-self.score(w)
            row['objective']=g;row['values']['achievement']=g
        elif self.genre=='auction':
            from .contests import settle
            # Revealed previous rival bids can update the public continuation;
            # no current pending bid/controller identity is supplied.
            rival=max((3,)+tuple(w.last[1:]))
            bids=(int(key.split(':')[1]),)+tuple(min(rival,b,8) for b in w.budgets[1:])
            after,_=settle(w,bids)
            if w.remaining:after=replace(after,prize=w.remaining[0],remaining=w.remaining[1:])
            g=self.score(after)-self.score(w);paid=(w.budgets[0]-after.budgets[0])/18
            row=effect(g,values={'achievement':g,'power':g},cost=.05*paid)
        else:
            from .combat import resolve,legal
            keys=legal(w,3);shots=[k for k in keys if k.startswith('shoot:')]
            enemy='guard' if w.tick==0 else shots[0] if shots else 'reload' if 'reload' in keys else 'guard'
            after,_=resolve(w,{0:key,3:enemy},seed)
            g=self.score(after)-self.score(w);health=(after.units[0].hp-w.units[0].hp)/9
            damage=(w.units[3].hp-after.units[3].hp)/9
            cost=.08 if key.startswith('heal:') else .015 if key.startswith('shoot:') else .005
            row=effect(g,needs={'physiology':health,'safety':health},
                values={'achievement':g,'power':damage,'security':-float(after.units[0].hp==0)},cost=cost)
        return after,row

    def observe(self,w,memory=None):
        choices=[]
        for key in self.keys(w):
            rows=[]
            for seed in self.seeds:
                _,row=self.advance(w,key,seed);row['p']=1/len(self.seeds);rows.append(row)
            choices.append(action(key,*rows))
        if self.genre=='resources':
            from .resource_world import deficits
            needs=deficits(w.empires[0],w)
        else:
            urgent=1-w.units[0].hp/9 if self.genre=='combat' else .1
            needs={'physiology':urgent,'safety':urgent,'growth':.3}
        c=context(self.name,choices,needs,self.profile['values'],dict(zip(TRAITS,self.profile['traits'])),mode=None)
        c['tick']=w.tick if self.genre=='combat' else w.round
        if memory is not None:c['state']=copy.deepcopy(memory)
        c['facts']['model']='public state; current public yields/prizes and revealed rival bids persist; no unrevealed events'
        return c

    def scripted(self,w,c):
        if self.genre=='resources':return 'wait'
        if self.genre=='auction':return 'bid:'+str(min(4,w.budgets[0]))
        keys=self.keys(w);shots=[k for k in keys if k.startswith('shoot:')]
        return shots[0] if shots else 'reload' if 'reload' in keys else 'guard'

    def actual(self,w,key,seed):
        # Events are owned by the actual experiment, outside advance/observe.
        # Only their resulting public state is visible on the following turn.
        if self.genre=='auction' and self.spec.get('rival_shift') and w.round>=1:
            from .contests import settle
            bids=(int(key.split(':')[1]),)+tuple(min(5,b,8) for b in w.budgets[1:])
            after,_=settle(w,bids)
            if w.remaining:after=replace(after,prize=w.remaining[0],remaining=w.remaining[1:])
            g=self.score(after)-self.score(w);paid=(w.budgets[0]-after.budgets[0])/18
            row=effect(g,values={'achievement':g,'power':g},cost=.05*paid)
        else:after,row=self.advance(w,key,seed)
        if self.genre=='resources' and self.spec.get('famine') and after.round==2:
            after=replace(after,food_yield=0)
        if self.genre=='auction' and self.spec.get('revised_prize') and after.round==1:
            after=replace(after,prize=3)
        return after,row


def run(spec,profile,variant,actual_seed=190,horizon=6):
    p=Probe(spec,profile);w=p.start();initial=p.score(w);c=p.observe(w);loop=DecisionLoop(c)
    trace=[];forecasts=0
    while not p.terminal(w):
        c=p.observe(w,loop.state);captured={}
        def planner(cs,ds):
            nonlocal forecasts
            paths,audit=rollout(cs[0],w,observe=p.observe,advance=p.advance,terminal=p.terminal,
                horizon=horizon,seeds=p.seeds,continuation=p.scripted if variant=='scripted' else None)
            captured.update(paths=paths,audit=audit);forecasts+=sum(len(b.effects) for bs in paths.values() for b in bs)
            return forecast(cs[0],paths,horizon=horizon,unit='rounds' if p.genre!='combat' else 'ticks',target='finite-payback')
        req=Request(c,{k:Binding(k,'public',()) for k in p.keys(w)},{k:() for k in p.keys(w)})
        result=DecisionLoop.decide_batch([(loop,req)],False,None if variant=='reflex' else planner)[0]
        key=result['decision']['action_id'];before=w;w,row=p.actual(w,key,actual_seed)
        # Deliberation has a different target. These probes deliberately do
        # not learn its hypothetical tail as an immediate observed outcome.
        loop.abandon(result['ticket'])
        assert not loop.memory.entries and c['personality']==dict(zip(TRAITS,profile['traits']))
        selected=captured.get('paths',{}).get(key,())
        expected=sum(b.probability*sum(r['objective'] for r in b.effects) for b in selected) if selected else None
        continuations=captured.get('audit',{}).get(key,[])
        trace.append(dict(before=asdict(before),after=asdict(w),root=key,flow=row,
            model_expected_undiscounted=expected,continuations=continuations,
            persona_hash=digest([c['personality'],c['values']]),decision_state=loop.state))
    mismatches=[]
    for i,t in enumerate(trace[:-1]):
        nexts=[r['actions'][1] for r in t['continuations'] if len(r['actions'])>1]
        if nexts:mismatches.append(sum(k!=trace[i+1]['root'] for k in nexts)/len(nexts))
    return dict(spec=spec,profile=profile['id'],variant=variant,actual_seed=actual_seed,horizon=horizon,
        score=p.score(w)-initial,steps=len(trace),actions=[r['root'] for r in trace],
        surviving=None if p.genre!='combat' else w.units[0].hp>0,
        mean_next_action_mismatch=sum(mismatches)/len(mismatches) if mismatches else None,
        model_flow_steps=forecasts,trace=trace)


def specs():
    for limit in (2,6,12):
        yield dict(genre='resources',limit=limit,famine=False)
    yield dict(genre='resources',limit=8,famine=True)
    yield dict(genre='auction',prizes=[8,10],budget=6)
    yield dict(genre='auction',prizes=[8,10,6],budget=10)
    yield dict(genre='auction',prizes=[8,10],budget=6,revised_prize=True)
    yield dict(genre='auction',prizes=[8,10,6],budget=10,rival_shift=True)
    for hp in (3,6):
        for limit in (1,8):yield dict(genre='combat',hp=hp,limit=limit)


def experiment(output,progress=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any((output/n).exists() for n in ('preregister.json','trajectories.jsonl','evaluation.json')):
        raise FileExistsError('frozen closed-loop evidence exists; use a fresh directory')
    ss=list(specs());ps=profiles();fixed=source_hashes();runs=[]
    # Include this helper explicitly: source_hashes uses all reflex Python files.
    try:base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):base=None
    prereg=dict(format='closed-loop-payback-v1',base=base,source_hashes=fixed,specs=ss,profiles=ps,
        variants=['reflex','scripted','persona'],horizon=6,model_combat_seeds=[800,801],actual_combat_seeds=[190,191],
        status='development probes; no tuning after outcome inspection; not unseen general strength evidence')
    (output/'preregister.json').write_text(json.dumps(prereg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (output/'trajectories.jsonl').open('w',encoding='utf-8') as stream:
        for s in ss:
            for p in ps:
                for seed in ((190,191) if s['genre']=='combat' else (190,)):
                    for variant in prereg['variants']:
                        r=run(s,p,variant,seed);stream.write(json.dumps(r,ensure_ascii=False)+'\n')
                        runs.append({k:v for k,v in r.items() if k!='trace'})
            if progress:progress(f'{s}: {len(runs)} complete episodes',flush=True)
    assert fixed==source_hashes()
    summary=[]
    for g in ('resources','auction','combat'):
        for reference in ('reflex','scripted'):
            rs=[r for r in runs if r['spec']['genre']==g];pairs=[]
            for r in rs:
                if r['variant']!='persona':continue
                b=next(b for b in rs if b['variant']==reference and b['spec']==r['spec'] and b['profile']==r['profile'] and b['actual_seed']==r['actual_seed'])
                pairs.append((r,b))
            summary.append(dict(genre=g,reference=reference,paired_episodes=len(pairs),
                better=sum(a['score']>b['score']+1e-9 for a,b in pairs),worse=sum(a['score']<b['score']-1e-9 for a,b in pairs),
                equal=sum(abs(a['score']-b['score'])<=1e-9 for a,b in pairs),
                changed=sum(a['actions']!=b['actions'] for a,b in pairs)))
    result=dict(format='closed-loop-payback-v1',episodes=len(runs),summary=summary,runs=runs,
        source_hashes=fixed,preregister_digest=digest(prereg),default_policy_changed=False,
        recursive_self_model=False,training_performed=False,cloud_runtime_used=False,license_granted=False)
    (output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result
