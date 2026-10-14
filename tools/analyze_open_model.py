"""Audit every registered public reveal and report paired campaign outcomes."""
import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from tools.open_model_games import OpenGame
from tools.run_open_model import truth


def mean(values):return statistics.fmean(values)


def audit(root):
    root=Path(root);reg=json.loads((root/'preregister.json').read_text());evaluation=json.loads((root/'evaluation.json').read_text())
    rows=[json.loads(line) for line in (root/'trajectories.jsonl').read_text().splitlines()]
    indices=[r['index'] for r in rows]
    assert not evaluation['errors'] and sorted(indices)==list(range(len(reg['jobs'])))
    decisions=0;metrics=[]
    for r in rows:
        game,scenario,profile,arm,seed,episodes,horizon,variant=reg['jobs'][r['index']]
        for name,value in (('game',game),('scenario',scenario),('profile',profile['id']),('arm',arm),('seed',seed),('episodes',episodes),('horizon',horizon),('variant',variant)):
            assert r[name]==value,(r['index'],name)
        g=OpenGame(game,profile,arm,length=6,variant=variant);memory=None;steps=iter(r['trace'])
        for episode in range(episodes):
            s=g.start(episode,memory)
            while not g.terminal(s):
                row=next(steps);decisions+=1
                assert row['public']==s.public and row['episode']==episode and row['tick']==s.tick
                assert row['action'] in g.keys(s)
                assert row['memory_before']==digest(g.kernel.learner.record(s.memory))
                events,_,_,_,forecast=g.kernel.forecast(s,row['action'])
                assert row['prediction']==forecast
                step=episode*g.length+s.public['tick'];actual=truth(g,s.public,scenario,step,episodes*g.length)
                assert row['actual']==actual
                draw=random.Random(int(digest(['open-world-actual',seed,step])[:16],16)).random();mass=0.
                for event in events:
                    mass+=actual[event.symbol]
                    if draw<mass:break
                assert event.symbol==row['event'] and event.flow==row['flow']
                values={key:sum(actual[e.symbol]*g.goal(type(s)(e.public,s.memory,s.identity,s.tick+1)).value for e in g.events(s.public,key)) for key in g.keys(s)}
                assert abs(row['regret']-(max(values.values())-values[row['action']]))<1e-12
                assert abs(row['log_loss']+math.log(forecast[event.symbol]))<1e-12
                assert row['diagnostics']==g.kernel.learner.diagnostics(s.memory)
                s=g.kernel.reveal(s,row['action'],event.public,event.flow)
                assert row['memory_after']==digest(g.kernel.learner.record(s.memory))
            assert r['goals'][episode]==g.goal(s).value
            memory=s.memory
        assert next(steps,None) is None and r['final_memory']==g.kernel.learner.record(memory)
        late=[t for t in r['trace'] if t['episode']>=episodes//2]
        metrics.append(dict(index=r['index'],game=game,scenario=scenario,profile=profile['id'],arm=arm,seed=seed,
            goal=mean(r['goals']),late_goal=mean(r['goals'][episodes//2:]),log_loss=mean(t['log_loss'] for t in r['trace']),
            late_log_loss=mean(t['log_loss'] for t in late),regret=mean(t['regret'] for t in r['trace']),
            late_regret=mean(t['regret'] for t in late),
            expected_log_loss=mean(-sum(q*math.log(t['prediction'][k]) for k,q in t['actual'].items()) for t in r['trace'])))
    lookup={(r['game'],r['scenario'],r['profile'],r['seed'],r['arm']):r for r in metrics};pairs=[]
    for r in metrics:
        if r['arm']!='adaptive':continue
        for baseline in ('library','marginal','long'):
            b=lookup.get((r['game'],r['scenario'],r['profile'],r['seed'],baseline))
            if b is None:continue
            pairs.append(dict(game=r['game'],scenario=r['scenario'],profile=r['profile'],seed=r['seed'],baseline=baseline,
                **{key:r[key]-b[key] for key in ('goal','late_goal','log_loss','late_log_loss','regret','late_regret','expected_log_loss')}))
    grouped=defaultdict(list)
    for r in pairs:grouped[(r['game'],r['scenario'],r['baseline'])].append(r)
    summary=[]
    for (game,scenario,baseline),group in sorted(grouped.items()):
        summary.append(dict(game=game,scenario=scenario,baseline=baseline,campaigns=len(group),
            **{key:mean(r[key] for r in group) for key in ('goal','late_goal','log_loss','late_log_loss','regret','late_regret','expected_log_loss')},
            better=sum(r['goal']>1e-9 for r in group),same=sum(abs(r['goal'])<=1e-9 for r in group),worse=sum(r['goal']< -1e-9 for r in group)))
    return dict(stage=reg['stage'],campaigns=len(rows),episodes=sum(len(r['goals']) for r in rows),decisions=decisions,
                all_public_reveals_replayed=True,metrics=metrics,pairs=pairs,groups=summary,
                caveat='paired campaigns share rules/personas; repeated episodes are not independent generalization cases; lower log loss/regret is better')


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=[]
    for root in a.roots:
        data=audit(root);result.append(data)
        print(json.dumps({k:v for k,v in data.items() if k not in ('metrics','pairs','groups')}),flush=True)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
