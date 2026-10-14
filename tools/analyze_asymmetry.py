"""Replay every public transition before summarising asymmetric campaigns."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.laboratory import profiles
from tools.run_asymmetry import entry, world, sample


def mean(xs):
    xs=[x for x in xs if x is not None]
    return None if not xs else math.fsum(xs)/len(xs)


def audit(row):
    profile=next(p for p in profiles() if p['id']==row['profile']);count=0
    for arm in row['arms']:
        memory=None
        for ep,end in enumerate(arm['ends']):
            g,prefix=entry(row['game'],row['case'],profile,row['seed'],row['variant'],ep)
            assert prefix==arm['prefixes'][ep]
            if memory is None:memory=g.kernel.learner.start()
            assert memory.observations==arm['entries'][ep]['observations_before']
            if row['case']=='late_observer':
                for item in prefix:
                    w=item['public'];key=item['action']
                    memory=g.kernel.learner.observe(memory,g.features(w,key),g.symbols(w),g.models(w,key),item['event'])
            s=g.start(ep,memory)
            assert s.public==arm['entries'][ep]['public']
            assert digest(g.kernel.learner.record(s.memory))==arm['entries'][ep]['memory']
            assert s.memory.observations==arm['entries'][ep]['observations_at_entry']
            for t in (t for t in arm['trace'] if t['episode']==ep):
                assert s.public==t['public'] and s.tick==t['tick'] and not g.terminal(s)
                assert t['action'] in g.keys(s)
                _,_,_,_,pred=g.kernel.forecast(s,t['action'])
                assert pred==t['prediction'] and world(g,s.public)==t['actual']
                assert digest(g.kernel.learner.record(s.memory))==t['before_memory']
                event=sample(g,s.public,t['action'],row['seed'],ep,'active')
                assert event.symbol==t['event']
                s=g.kernel.reveal(s,t['action'],event.public,event.flow)
                assert digest(g.kernel.learner.record(s.memory))==t['after_memory']
                assert t['root_opportunity_loss']>=-1e-10 and t['personality_gap']>=-1e-10
                if t['continuation_value'] is not None:
                    assert abs(t['best_possible']-t['continuation_value']-t['personality_gap']-t['prediction_gap'])<1e-10
                count+=1
            assert g.terminal(s) and s.public==end['public'] and g.goal(s).value==end['goal']
            assert list(g.quantities(s.public))==end['quantities']
            memory=s.memory
        assert g.kernel.learner.record(memory)==arm['final_memory']
    return count


def summarize(rows):
    groups=defaultdict(list);personas=defaultdict(list);roles=defaultdict(dict);adverse=[]
    for row in rows:
        groups[row['game']+'/'+row['case']].append(row)
        personas[(row['game'],row['case'],row['seed'])].append(row)
        roles[(row['game'],row['profile'],row['seed'])][row['case']]=row
    out={}
    for group,rr in groups.items():
        arms={}
        for name in ('live','reflex'):
            aa=[next(a for a in r['arms'] if a['arm']==name) for r in rr]
            ts=[t for a in aa for t in a['trace']]
            roots=[next(t for t in a['trace'] if t['episode']==ep) for a in aa for ep in range(len(a['ends']))]
            arms[name]=dict(campaigns=len(aa),games=sum(len(a['ends']) for a in aa),decisions=len(ts),
                goal=mean([e['goal'] for a in aa for e in a['ends']]),
                opportunity_loss=mean([t['root_opportunity_loss'] for t in ts]),
                attainable_at_entry=mean([t['best_possible'] for t in roots]),
                impossible_entries=sum(t['best_possible']<1e-12 for t in roots),
                initial_continuation_value=mean([t['continuation_value'] for t in roots]),
                initial_personality_gap=mean([t['personality_gap'] for t in roots]),
                initial_prediction_gap=mean([t['prediction_gap'] for t in roots]),
                actions=dict(sorted(__import__('collections').Counter(t['action'] for t in ts).items())))
        delta=[]
        for r in rr:
            values={a['arm']:mean([e['goal'] for e in a['ends']]) for a in r['arms']}
            d=values['live']-values['reflex'];delta.append(d)
            if d< -1e-10:adverse.append(dict(game=r['game'],case=r['case'],profile=r['profile'],seed=r['seed'],delta=d))
        out[group]=dict(arms=arms,delta=mean(delta),improved=sum(d>1e-10 for d in delta),
                       same=sum(abs(d)<=1e-10 for d in delta),worse=sum(d< -1e-10 for d in delta))
    personality={}
    for name in ('live','reflex'):
        valid=[rr for rr in personas.values() if len(rr)==4]
        n=sum(len({next(a for a in r['arms'] if a['arm']==name)['starts'][0] for r in rr})>1 for rr in valid)
        personality[name]=dict(matched_boards=len(valid),different_first_actions=n)
    role_changes=[]
    for (game,profile,seed),cases in roles.items():
        if 'balanced' not in cases:continue
        baseline=next(a for a in cases['balanced']['arms'] if a['arm']=='live')
        for case,row in cases.items():
            if case=='balanced':continue
            arm=next(a for a in row['arms'] if a['arm']=='live')
            role_changes.append(dict(game=game,profile=profile,seed=seed,case=case,
                baseline=baseline['starts'],actions=arm['starts'],
                changed_first=baseline['starts'][0]!=arm['starts'][0]))
    return dict(groups=out,personality=personality,role_changes=role_changes,adverse=adverse,
        caution='root opportunities assume optimal later moves under the referee law; continuation gaps condition on fixed current belief, not subsequent learning; case/profile/seed repetitions correlated')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    rows=[json.loads(s) for s in (root/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    ev=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    assert len(rows)==ev['planned']==ev['complete'] and not ev['errors']
    count=sum(audit(r) for r in rows);assert count==ev['decisions']
    result=summarize(rows);result.update(replayed_decisions=count,games=ev['games'])
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(replayed_decisions=count,games=ev['games'],personality=result['personality'],
        groups={k:{x:v[x] for x in ('delta','improved','same','worse')} for k,v in result['groups'].items()})),flush=True)


if __name__=='__main__':main()
