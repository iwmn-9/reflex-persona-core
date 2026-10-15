"""Check all construction trajectories and voluntary signatures independently."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import gzip
import itertools
import json
from reflex import team_projects as game
from reflex.core import digest
from reflex.cooperation import Ballot,Offer,negotiate
from tools.audit_cooperation import close


def audit(root):
    root=Path(root);runs=json.loads((root/'results.json').read_text());count=Counter();totals={}
    for r in runs:
        name=f'{r["scenario"]}-{r["members"]}-{r["seed"]}-{int(r["communicate"])}.json.gz'
        with gzip.open(root/name,'rt',encoding='utf-8') as f:saved=json.load(f)
        close(saved['result'],r);w=game.Workshop.start(r['scenario'],members=r['members'])
        for t in saved['trace']:
            chosen={int(k):v for k,v in t['chosen'].items()};submitted={int(k):v for k,v in t['submitted'].items()}
            assert t['tick']==w.tick
            if r['communicate']:
                for team,receipt in enumerate(t['negotiations']):
                    people=game.members(w,team);offers=[]
                    for number,role in enumerate(itertools.product(*(game.legal(w,i) for i in people))):
                        offers.append(Offer(f'method-{number:03d}',tuple(sorted((f'worker-{i}',k) for i,k in zip(people,role)))))
                    scope=('team-projects-v1',f'projects-{r["seed"]}',w.tick,digest([w.stock,w.points,[p.energy for p in w.people]]))
                    ballots=[]
                    for i in people:
                        e=t['evaluations'][str(i)]
                        for key in e['accepted']:
                            assert e['scores'][key]>e['scores'][e['outside_option']]+1e-9
                        ballots.append(Ballot(f'worker-{i}',scope,submitted[i],tuple(e['accepted'])))
                    selection,actual=negotiate(offers,ballots,receipt['order'],scope=scope,
                        legal={f'worker-{i}':set(game.legal(w,i)) for i in people})
                    close(actual,receipt)
                    for i in people:assert chosen[i]==selection[f'worker-{i}']
                    count['agreements']+=len(receipt['agreements'])
            else:assert chosen==submitted
            after,resolution=game.resolve(w,chosen);close(resolution,t['resolution'])
            # Independent pre-state reservation ledger: concurrent gathers may
            # not finance builds in this same step; excess reservations all fail.
            conflicts=[]
            for team in (0,1):
                claims=[i for i in game.members(w,team) if chosen[i] in ('build','train')]
                if 2*len(claims)>w.stock[team]:conflicts.extend(claims)
            assert conflicts==t['resolution']['material_conflicts']
            close(list(after.points),t['points']);close(list(w.stock),t['stock'])
            w=after;count['world_ticks']+=1;count['decisions']+=len(w.people)
        assert game.terminal(w);assert list(w.points)==r['points'];count['worlds']+=1
        key=f'{r["scenario"]}|{r["members"]}|{r["communicate"]}'
        group=totals.setdefault(key,Counter());group['worlds']+=1;group['points']+=sum(r['points']);group['conflicts']+=r['conflicts'];group['agreements']+=r['agreements']
    lookup={(r['seed'],r['scenario'],r['members'],r['communicate']):r for r in runs};paired=Counter();conditions=defaultdict(Counter)
    for k,r in lookup.items():
        if not k[-1]:continue
        old=lookup[k[:-1]+(False,)];delta=sum(r['points'])-sum(old['points'])
        for c in (paired,conditions[f'{k[1]}|{k[2]}']):
            c['world_pairs']+=1;c['points_delta']+=delta;c['fewer_conflicts']+=old['conflicts']-r['conflicts']
            c['better' if delta>0 else 'worse' if delta<0 else 'same']+=1
    report=dict(verified=dict(count),groups={k:dict(v) for k,v in totals.items()},paired=dict(paired),by_condition={k:dict(v) for k,v in conditions.items()},
        limitations=['self-play efficiency comparison, not a win-rate advantage over a fixed rival','all members of an existing team invited; no freely selected partners here'])
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();audit(a.root)
