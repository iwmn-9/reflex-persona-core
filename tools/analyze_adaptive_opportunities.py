"""Paired campaign summaries and public trajectory/learning replay checks."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.laboratory import profiles
from tools.open_model_games import OpenGame


def mean(xs):return math.fsum(xs)/len(xs) if xs else None


def replay(row,arm):
    p=next(p for p in profiles() if p['id']==row['profile'])
    g=OpenGame(row['game'],p,arm='long' if arm['arm']=='long' else 'adaptive',length=3,variant=row['variant'])
    memory=g.kernel.learner.restore(arm['warm_memory'])
    if arm['arm']=='frozen':g.kernel.update=False
    seen=0
    for ep,goal in enumerate(arm['goals']):
        s=g.start(ep,memory)
        for t in (t for t in arm['trace'] if t['episode']==ep):
            assert not g.terminal(s) and s.public==t['public']
            assert t['action'] in g.keys(s)
            events,_,_,_,p=g.kernel.forecast(s,t['action'])
            assert p==t['prediction'] and digest(g.kernel.learner.record(s.memory))==t['memory_before']
            event=next(e for e in events if e.symbol==t['event'])
            s=g.kernel.reveal(s,t['action'],event.public,event.flow)
            assert digest(g.kernel.learner.record(s.memory))==t['memory_after'];seen+=1
        assert g.terminal(s) and g.goal(s).value==goal
        memory=s.memory
    assert g.kernel.learner.record(memory)==arm['final_memory']
    return seen


def summarize(rows):
    groups=defaultdict(list);adverse=[];replayed=0;persona=defaultdict(dict)
    for row in rows:
        arms={a['arm']:a for a in row['arms']}
        group=(row['game'],row['scenario']);groups[group].append(row)
        for a in row['arms']:
            replayed+=replay(row,a)
            persona[(row['game'],row['scenario'],row['seed'],a['arm'])][row['profile']]=a['starts'][0]
        for control in ('frozen','long','legacy'):
            delta=mean(arms['live']['goals'])-mean(arms[control]['goals'])
            if delta< -1e-10:adverse.append(dict(game=row['game'],scenario=row['scenario'],profile=row['profile'],seed=row['seed'],control=control,delta=delta))
    out={}
    for group,rr in sorted(groups.items()):
        entry={}
        for name in ('frozen','long','live','legacy'):
            aa=[next(a for a in r['arms'] if a['arm']==name) for r in rr]
            ts=[t for a in aa for t in a['trace']];revisions=[t['revision']['delta'] for t in ts if t['revision'] is not None]
            entry[name]=dict(campaigns=len(aa),mean_goal=mean([mean(a['goals']) for a in aa]),
                late_goal=mean([mean(a['goals'][6:]) for a in aa]),log_loss=mean([t['log_loss'] for t in ts]),
                early_probability_error=mean([t['probability_error'] for t in ts if t['episode']<2]),
                late_probability_error=mean([t['probability_error'] for t in ts if t['episode']>=8]),
                revisions=len(revisions),revision_effect_mean=mean(revisions),
                helpful_revisions=sum(x>1e-10 for x in revisions),harmful_revisions=sum(x< -1e-10 for x in revisions),
                neutral_revisions=sum(abs(x)<=1e-10 for x in revisions),
                parity_start_switches=sum(a['starts'][i]!=a['starts'][i-2] for a in aa for i in range(2,len(a['starts']))))
        paired={}
        for control in ('frozen','long','legacy'):
            deltas=[]
            for r in rr:
                aa={a['arm']:a for a in r['arms']}
                deltas.append(mean(aa['live']['goals'])-mean(aa[control]['goals']))
            paired[control]=dict(mean_delta=mean(deltas),improved=sum(d>1e-10 for d in deltas),same=sum(abs(d)<=1e-10 for d in deltas),worse=sum(d< -1e-10 for d in deltas))
        out['/'.join(group)]=dict(arms=entry,live_minus=paired)
    differences={}
    for name in ('frozen','long','live','legacy'):
        ps=[v for k,v in persona.items() if k[-1]==name and len(v)==4]
        differences[name]=dict(matched_initial_boards=len(ps),different_persona_actions=sum(len(set(p.values()))>1 for p in ps))
    by_case={(r['game'],r['scenario'],r['profile'],r['seed']):r for r in rows};outliers=defaultdict(list)
    for r in rows:
        if r['scenario']!='outlier':continue
        stable=by_case.get((r['game'],'stable',r['profile'],r['seed']))
        if stable is None:continue
        aa={a['arm']:a for a in r['arms']};bb={a['arm']:a for a in stable['arms']}
        # Skip episode zero here: distinguish unavoidable immediate damage
        # from behavior persisting in later fresh games with the same draws.
        effects={name:mean(aa[name]['goals'][1:])-mean(bb[name]['goals'][1:]) for name in aa}
        outliers[r['game']].append(dict(profile=r['profile'],seed=r['seed'],later_goal_effect=effects,
            live_minus_frozen_effect=effects['live']-effects['frozen'],
            first_event_changed=aa['live']['trace'][0]['event']!=bb['live']['trace'][0]['event']))
    return dict(campaigns=len(rows),games=sum(len(a['goals']) for r in rows for a in r['arms']),replayed_decisions=replayed,
        groups=out,personality= differences,adverse=sorted(adverse,key=lambda r:r['delta']),
        paired_outlier_followthrough=dict(outliers),
        interpretation='campaign-level paired goal changes; same-board revision effect under diagnostic world; no pooled cross-game unit, no human equivalence or universal stability claim')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output');a=p.parse_args();root=Path(a.root)
    rows=[json.loads(s) for s in (root/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    evaluation=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    assert evaluation['planned']==evaluation['complete']==len(rows) and not evaluation['errors']
    result=summarize(rows);dst=Path(a.output) if a.output else root/'summary.json'
    dst.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','adverse')},ensure_ascii=False))


if __name__=='__main__':main()
