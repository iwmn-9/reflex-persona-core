"""Transfer the ordinal consent protocol to existing shared-stock construction.

All legal within-team joint roots are enumerated (2/3 actors); members privately
rank consequences. No extra reward for signing, gathering or reserving stock.
Opponent CURRENT intentions are never given to the forecast: the existing public
greedy model is a prediction only. Every team may refuse and use submitted roots.
"""
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import itertools
import json
import numpy as np
from reflex import team_projects as game
from reflex.core import Policy, digest, NEEDS
from reflex.cooperation import Offer, Ballot, evaluate, negotiate
from reflex.examples import action
from reflex.laboratory import PROFILES


def run(seed,scenario,members,communicate):
    w=game.Workshop.start(scenario,members=members)
    rng=np.random.default_rng(seed);profiles=[copy.deepcopy(PROFILES[int(i)]) for i in rng.integers(0,4,len(w.people))]
    states={};trace=[];policy=Policy(principle_priority='finite')
    while not game.terminal(w):
        cs=[game.make_context(w,i,profiles[i],seed,states.get(i)) for i in range(len(w.people))]
        decisions=[policy.choose(c,False) for c in cs]
        submitted={i:d['action_id'] for i,d in enumerate(decisions)}
        chosen=dict(submitted);negotiations=[];evaluations={}
        if communicate:
            for team in (0,1):
                people=game.members(w,team);owners=tuple(f'worker-{i}' for i in people)
                scope=('team-projects-v1',cs[0]['scope']['episode'],w.tick,digest([w.stock,w.points,[p.energy for p in w.people]]))
                offers=[];joint={};options={i:[] for i in people};roots={i:{} for i in people}
                # Only allied current intentions are explicitly communicated.
                outsider=game.greedy(w,1-team)
                outside_model,_=game.resolve(w,{**{i:submitted[i] for i in people},**outsider})
                for i in people:
                    options[i].append(action('outside',game.consequence(w,outside_model,i,submitted[i])))
                    roots[i]['outside']=submitted[i]
                for number,role in enumerate(itertools.product(*(game.legal(w,i) for i in people))):
                    key=f'method-{number:03d}';own_roots=dict(zip(people,role))
                    offer=Offer(key,tuple(sorted((f'worker-{i}',k) for i,k in own_roots.items())))
                    offers.append(offer);joint[key]=own_roots
                    after,_=game.resolve(w,{**own_roots,**outsider})
                    for i in people:
                        options[i].append(action(key,game.consequence(w,after,i,own_roots[i])))
                        roots[i][key]=own_roots[i]
                ballots=[]
                for i in people:
                    ranked,audit=evaluate(cs[i],options[i],roots[i],('outside',),policy=policy)
                    ballots.append(Ballot(f'worker-{i}',scope,submitted[i],ranked));evaluations[i]=audit
                offset=w.tick%members;order=owners[offset:]+owners[:offset]
                agreed,audit=negotiate(offers,ballots,order,scope=scope,
                    legal={f'worker-{i}':set(game.legal(w,i)) for i in people})
                chosen.update({i:agreed[f'worker-{i}'] for i in people});negotiations.append(audit)
        after,resolution=game.resolve(w,chosen)
        for i,d in enumerate(decisions):
            states[i]=d['next_state'];states[i]['intent_action']=chosen[i]
            if chosen[i]!=submitted[i]:states[i]['age']=0
        trace.append(dict(tick=w.tick,submitted=submitted,chosen=chosen,negotiations=negotiations,
            evaluations=evaluations,resolution=resolution,stock=w.stock,points=after.points))
        w=after
    return dict(seed=seed,scenario=scenario,members=members,communicate=communicate,
        points=list(w.points),conflicts=sum(len(t['resolution']['material_conflicts']) for t in trace),
        agreements=sum(len(n['agreements']) for t in trace for n in t['negotiations']),
        rejections=sum(bool(a['refused']) for t in trace for n in t['negotiations'] for a in n['attempts']),
        ticks=w.tick,profiles=[p['id'] for p in profiles]),trace


def experiment(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False);results=[]
    repo=Path(__file__).resolve().parents[1]
    names=[p.relative_to(repo).as_posix() for p in (repo/'reflex').glob('*.py')]+['tools/cooperation_projects.py']
    sources={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in names}
    for name in names:
        dest=root/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((repo/name).read_bytes())
    (root/'registration.json').write_text(json.dumps(dict(arguments=vars(args),
        sources=sources,
        criterion='same consent protocol, no new personality or game-specific score bonus; disclose conflicts/regressions',
        limitations=['one-step public opponent tactic','all team members invited, unlike free partner search in encounters']),indent=2)+'\n',encoding='utf-8')
    for seed,scenario,members,arm in itertools.product(range(args.seed_start,args.seed_start+args.seeds),
            ('balanced','exhausted','shock'),(2,3),(False,True)):
        result,trace=run(seed,scenario,members,arm);results.append(result)
        name=f'{scenario}-{members}-{seed}-{int(arm)}.json.gz'
        with gzip.open(root/name,'wt',encoding='utf-8') as f:json.dump(dict(result=result,trace=trace),f)
    (root/'results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(games=len(results),totals={str(arm):dict(points=sum(sum(r['points']) for r in results if r['communicate']==arm),
        conflicts=sum(r['conflicts'] for r in results if r['communicate']==arm)) for arm in (False,True)})))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed-start',type=int,default=92200);p.add_argument('--seeds',type=int,default=12)
    experiment(p.parse_args())
