"""Replay public facts independently; summarize paired completed campaigns."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from tools.open_model_games import OpenGame
from tools.run_open_model import truth


def audit(root):
    root=Path(root);read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    reg=read(root/'preregister.json');ev=read(root/'evaluation.json');snapshot=read(root/'source_snapshot.json')
    assert reg['source']==digest(snapshot)
    for path,h in snapshot.items():
        assert hashlib.sha256((root/'_source'/path).read_bytes()).hexdigest()==h
        if path.startswith('reflex/') or path in ('tools/open_model_games.py','tools/run_open_model.py'):
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,'replay rules/learner must match measured source'
    rows=[json.loads(l) for l in (root/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    assert not ev['errors'] and ev['complete']==ev['planned']==len(reg['jobs'])==len(rows)
    assert sorted(r['index'] for r in rows)==list(range(len(rows)))
    metrics=[];decisions=0;parity=0;openings={}
    for r in rows:
        game,scenario,profile,arm,seed,episodes,variant=reg['jobs'][r['index']]
        assert [r[k] for k in ('game','scenario','profile','arm','seed','episodes','variant')]==[game,scenario,profile['id'],arm,seed,episodes,variant]
        g=OpenGame(game,profile,arm='library' if arm=='library' else 'adaptive',variant=variant)
        memory=None;steps=iter(r['trace']);count=0
        for episode in range(episodes):
            s=g.start(episode,memory)
            while not g.terminal(s):
                t=next(steps);key=t['action'];decisions+=1;count+=1
                assert t['episode']==episode and t['tick']==s.tick and t['public']==s.public and key in g.keys(s)
                assert t['memory_before']==digest(g.kernel.learner.record(s.memory))
                events,_,_,_,p=g.kernel.forecast(s,key)
                assert set(p)==set(t['prediction']) and all(abs(p[k]-t['prediction'][k])<1e-12 for k in p)
                step=episode*g.length+s.public['tick'];actual=truth(g,s.public,scenario,step,episodes*g.length)
                assert actual==t['actual']
                draw=random.Random(int(digest(['opportunity-actual',seed,step])[:16],16)).random();mass=0.
                for event in events:
                    mass+=actual[event.symbol]
                    if draw<mass:break
                assert event.symbol==t['event'] and event.flow==t['flow']
                assert abs(t['log_loss']+math.log(p[event.symbol]))<1e-12
                assert key in t['root_purpose'] and abs(t['selected_purpose']-t['root_purpose'][key])<1e-12
                assert set(t['root_purpose'])<=set(g.keys(s)) and all(-1.0000000001<=x<=1.0000000001 for x in t['root_purpose'].values())
                if arm=='persona':assert t['table_action']==key;parity+=1
                if arm in ('short','library'):
                    assert t['table_action'] is None and t['graph_nodes']==0 and t['solution'] is None
                    for option,v in t['root_purpose'].items():
                        possible,_,_,_,prob=g.kernel.forecast(s,option)
                        expected=sum(prob[x.symbol]*g.goal(type(s)(x.public,s.memory,s.identity,s.tick+1)).value for x in possible)
                        assert abs(expected-v)<1e-12
                s=g.kernel.reveal(s,key,event.public,event.flow)
                assert t['memory_after']==digest(g.kernel.learner.record(s.memory))
            assert r['goals'][episode]==g.goal(s).value;memory=s.memory
        assert next(steps,None) is None and r['final_memory']==g.kernel.learner.record(memory)
        first=r['trace'][0];stamp=(game,profile['id'],arm,variant)
        opening={k:first[k] for k in ('public','action','prediction','root_purpose')}
        if stamp in openings:assert openings[stamp]==opening,'unobserved scenario/seed cannot change the cold opening'
        else:openings[stamp]=opening
        metrics.append(dict(game=game,scenario=scenario,profile=profile['id'],arm=arm,seed=seed,episodes=episodes,
            goal=statistics.fmean(r['goals']),late_goal=statistics.fmean(r['goals'][episodes//2:]),decisions=count,
            log_loss=statistics.fmean(t['log_loss'] for t in r['trace']),
            initial_terminal_mse=None if arm in ('short','library') else statistics.fmean((t['selected_purpose']-r['goals'][t['episode']])**2 for t in r['trace'] if t['tick']==0)))
    assert decisions==ev['decisions']
    return dict(stage=reg['stage'],baseline_only=reg.get('baseline_only',False),campaigns=len(rows),episodes=sum(r['episodes'] for r in rows),
        decisions=decisions,all_public_reveals_replayed=True,recorded_persona_choice_agreements=parity,
        cold_openings=[dict(game=k[0],profile=k[1],arm=k[2],variant=k[3],**v) for k,v in sorted(openings.items())],metrics=metrics)


def compare(audits):
    rows=[r for a in audits for r in a['metrics']]
    lookup={(r['game'],r['scenario'],r['profile'],r['seed'],r['arm']):r for r in rows};pairs=[]
    assert len(lookup)==len(rows)
    for r in rows:
        if r['arm']!='persona':continue
        for arm in ('short','purpose','library'):
            b=lookup.get((r['game'],r['scenario'],r['profile'],r['seed'],arm))
            if b is not None:pairs.append(dict(game=r['game'],scenario=r['scenario'],profile=r['profile'],seed=r['seed'],baseline=arm,
                goal=r['goal']-b['goal'],late_goal=r['late_goal']-b['late_goal']))
    groups=[]
    for game in sorted({r['game'] for r in pairs}):
        for arm in ('short','purpose','library'):
            rs=[r for r in pairs if r['game']==game and r['baseline']==arm]
            if not rs:continue
            groups.append(dict(game=game,baseline=arm,campaigns=len(rs),goal=statistics.fmean(r['goal'] for r in rs),
                better=sum(r['goal']>1e-9 for r in rs),same=sum(abs(r['goal'])<=1e-9 for r in rs),worse=sum(r['goal']< -1e-9 for r in rs)))
    arms=[]
    for game in sorted({r['game'] for r in rows}):
        for arm in ('library','short','purpose','persona'):
            rs=[r for r in rows if r['game']==game and r['arm']==arm]
            if rs:arms.append(dict(game=game,arm=arm,campaigns=len(rs),goal=statistics.fmean(r['goal'] for r in rs),
                late_goal=statistics.fmean(r['late_goal'] for r in rs),log_loss=statistics.fmean(r['log_loss'] for r in rs)))
    return dict(audits=audits,pairs=pairs,groups=groups,arms=arms,caveat='paired campaigns share rules/personas; episodes and repeated seeds are not independent game families')


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    audits=[]
    for root in a.roots:
        out=audit(root);audits.append(out);print(json.dumps({k:v for k,v in out.items() if k not in ('metrics','cold_openings')}),flush=True)
    Path(a.output).write_text(json.dumps(compare(audits),indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
