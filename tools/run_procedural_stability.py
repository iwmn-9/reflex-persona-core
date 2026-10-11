"""Unfitted procedural rule graphs, separate from the four authored games.

All graph rules and current node are public. Future chance results are sampled
independently of the actual trajectory seed. A game-owned binary terminal goal
is used; there is no fitted cutoff value or graph-specific action heuristic.
"""
import argparse,copy,hashlib,json,random,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
from reflex.core import Policy,TRAITS,digest
from reflex.examples import context,action,effect
from reflex.laboratory import profiles
from reflex.purpose_plan import Goal
from reflex.continuation_search import search_forecast
from reflex.plan_continuity import PlanIntention
from reflex.replan_check import ReplanCheck
from reflex.deliberation import select

MODES=('retained','repair','feedback')


class Trial:
    def __init__(self,graph,profile):
        self.profile=profile;self.graph=graph;self.length=5;self.width=4
        rng=random.Random(graph);self.edges={};self.effects={}
        for stage in range(self.length):
            for node in range(self.width):
                for key in ('option-u','option-v'):
                    self.edges[stage,node,key]=(rng.randrange(self.width),rng.randrange(self.width))
                    self.effects[stage,node,key]=rng.choice((-.15,0.,.15))
        self.winning=frozenset(rng.sample(range(self.width),2))
        self.name=digest([graph,profile['id']]);self.target='public-terminal-class'

    def terminal(self,state):return state[0]>=self.length
    def goal(self,state):
        if not self.terminal(state):return Goal(self.target,'running',0.)
        win=state[1] in self.winning;return Goal(self.target,'success' if win else 'failure',1. if win else -1.)
    def outcome(self,state,key,seed):
        # Chance is keyed to the public opportunity and an independent RNG
        # stream. The controller does not receive this stream identifier.
        token=json.dumps([seed,state,key],separators=(',',':')).encode()
        return hashlib.sha256(token).digest()[0]%2
    def advance(self,state,key,seed):
        after=(state[0]+1,self.edges[*state,key][self.outcome(state,key,seed)])
        row=effect(self.goal(after).value/2,values={'achievement':self.goal(after).value/2,
            'benevolence':self.effects[*state,key]},style={'openness':.15 if key=='option-u' else -.15})
        return after,row
    def observe(self,state,memory=None):
        rows=[]
        for key in ('option-u','option-v'):
            outcomes=[]
            for node in self.edges[*state,key]:
                after=(state[0]+1,node);v=self.goal(after).value/2
                outcomes.append(effect(v,values={'achievement':v,'benevolence':self.effects[*state,key]},
                    style={'openness':.15 if key=='option-u' else -.15},p=.5))
            rows.append(action(key,*outcomes))
        c=context(self.name,rows,needs={'physiology':0.,'safety':0.,'growth':.3},values=self.profile['values'],
            traits=dict(zip(TRAITS,self.profile['traits'])),mode=None)
        c['scope']['game']='generated-public-rule-graph';c['tick']=state[0]
        c['facts']['public_rules_sha256']=digest([sorted(self.edges.items()),sorted(self.winning)])
        c['objective']='Reach the publicly designated terminal node class.'
        if memory is not None:c['state']=copy.deepcopy(memory)
        return c


def run_game(job):
    graph,index,actual_seed,mode=job;p=Trial(graph,profiles()[index]);state=(0,0);c=p.observe(state)
    memory=c['state'];intention=PlanIntention(c);policy=Policy(principle_priority='finite');trace=[]
    def forecast(cc,world,observer,retained,guard):
        return search_forecast(cc,world,observe=p.observe,advance=p.advance,terminal=p.terminal,
            assess=lambda s:p.goal(s).record(),horizon=6,seeds=(100,101,102,103),validation_seeds=tuple(range(200,208)),
            target=p.target,unit='public-turns',policy=policy,retained=retained,repairs=0 if mode=='retained' else 24,
            repair_guard=guard,width=2,depth=2,settlement_weight='absolute')
    while not p.terminal(state):
        c=p.observe(state,memory);remaining=intention.offer(c,target=p.target,unit='public-turns',horizon=6)
        f=forecast(c,state,None,remaining,mode=='feedback')
        if mode=='feedback' and f.stability_reference is not None:
            f=replace(f,replan_check=ReplanCheck(state,p.observe,p.advance,p.terminal,lambda s:p.goal(s).record(),
                lambda cc,w,obs,tail:forecast(cc,w,obs,tail,True),(300,301,302,303)))
        saved=digest([c,intention.record()]);records,a=select([c],f,policy)
        assert saved==digest([c,intention.record()]);d=records[0];before=state
        state,row=p.advance(state,d['action_id'],actual_seed);memory=d['next_state']
        intention.remember(c,f,a['selected_plan'],d['action_id'])
        trace.append(dict(before=before,after=state,root=d['action_id'],flow=row,goal=p.goal(state).record(),
            deliberation=a,intention=intention.record(),state=memory,persona_hash=digest([c['personality'],c['values']])))
    return dict(graph=graph,profile_index=index,profile=profiles()[index]['id'],seed=actual_seed,mode=mode,
        status=p.goal(state).status,goal_value=p.goal(state).value,actions=[t['root'] for t in trace],trace=trace)


def audit_game(packet):
    r,job=packet;p=Trial(r['graph'],profiles()[r['profile_index']]);owner=None;turns=0;checks=0;rejected=0
    for t in r['trace']:
        before=tuple(t['before']);bit=hashlib.sha256(json.dumps([r['seed'],before,t['root']],separators=(',',':')).encode()).digest()[0]%2
        after=(before[0]+1,p.edges[*before,t['root']][bit]);assert list(after)==list(t['after'])
        assert p.goal(after).record()==t['goal'];i=t['intention'];identity=(i['owner'],i['preferences'],i['target'],i['unit'])
        if owner is None:owner=identity
        assert identity==owner and i['last_tick']==before[0];turns+=1
        report=t['deliberation']['metadata'].get('replan_check',{})
        checks+=report.get('checked',False);rejected+=report.get('accepted') is False
    assert digest(run_game(job))==digest(r),'whole frozen procedural episode replay'
    return dict(actual_turns=turns,full_replay_turns=turns,feedback_checks=checks,rejections=rejected)


def run(a):
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    jobs=[(g,p,s,m) for g in range(12100,12100+a.graphs) for p in range(4) for s in range(18000,18000+a.seeds) for m in MODES]
    if (root/'preregister.json').exists():raise FileExistsError('fresh study required')
    pre=dict(format='procedural-stability-v1',jobs=jobs,modes=MODES,
        primary='paired actual binary terminal goal, per graph/profile and seed; whole frozen replay',
        hypothesis='common protection/one-replan check remains useful on generated public rule graphs without tuning',
        constraints=['same five-stage graph in every arm','game-owned goal and public rules','no actual seed input to controllers',
            'same persona, six-step horizon, width2/depth2, eight validation branches',
            'graphs are not chosen by outcomes; no graph-specific action priorities'],
        limits=['tiny fully public authored generator, not unknown-rule understanding or a game-player benchmark',
            'only finite sampled model branches and one future replan','no actual feedback learning or team interaction'])
    (root/'preregister.json').write_text(json.dumps(pre,indent=2)+'\n',encoding='utf-8')
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(root/'trajectories.jsonl').open('w',encoding='utf-8') as stream:
        fs=[pool.submit(run_game,j) for j in jobs]
        for n,f in enumerate(as_completed(fs),1):
            r=f.result();stream.write(json.dumps(r,separators=(',',':'))+'\n');stream.flush()
            if n%12==0:print(f'{n}/{len(jobs)} generated',flush=True)
    (root/'completed.json').write_text(json.dumps(dict(games=len(jobs),complete=True))+'\n',encoding='utf-8')


def analyze(a):
    root=Path(a.root);pre=json.loads((root/'preregister.json').read_text());jobs={digest(j):j for j in pre['jobs']}
    rows=[json.loads(s) for s in (root/'trajectories.jsonl').read_text().splitlines()]
    indexed={digest([r['graph'],r['profile_index'],r['seed'],r['mode']]):r for r in rows};assert len(rows)==len(indexed)==len(jobs) and set(indexed)==set(jobs)
    checks=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(audit_game,(r,jobs[k])) for k,r in indexed.items()]
        for n,f in enumerate(as_completed(fs),1):
            checks.append(f.result())
            if n%12==0:print(f'{n}/{len(jobs)} independently verified',flush=True)
    pairs=[]
    for r in rows:
        if r['mode']=='retained':continue
        old=indexed[digest([r['graph'],r['profile_index'],r['seed'],'retained'])]
        pairs.append(dict(graph=r['graph'],profile=r['profile'],seed=r['seed'],mode=r['mode'],delta=r['goal_value']-old['goal_value'],actions_equal=r['actions']==old['actions']))
    result=dict(games=len(rows),full_replay=True,pairs=pairs,
        arms={m:dict(games=sum(r['mode']==m for r in rows),wins=sum(r['mode']==m and r['status']=='success' for r in rows)) for m in MODES},
        checks={k:sum(c[k] for c in checks) for k in checks[0]},
        trajectories_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest(),
        source_snapshot_sha256=hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest())
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result['arms']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--graphs',type=int,default=4)
    p.add_argument('--seeds',type=int,default=2);p.add_argument('--workers',type=int,default=2);p.add_argument('--analyze',action='store_true');p.add_argument('--frozen',action='store_true');a=p.parse_args()
    if a.analyze:analyze(a)
    elif a.frozen:run(a)
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--graphs',str(a.graphs),'--seeds',str(a.seeds),'--workers',str(a.workers)])
