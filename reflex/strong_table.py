"""One objective-focused CPU, three independently learning personality NPCs."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from .core import digest
from .laboratory import PROFILES
from .goofspiel import Position, make_context, referee
from .board_models import ThanksPosition, thanks_referee_score
from .tabletop_trials import competitive, score, thanks_observe
from .strong_search import (PublicMemory, STRONG, PERSONA, search, objective_choice,
    persona_context, reasonable_persona, random_stream, SearchBudget)
from .goal_progress import relative_progress, choose_with_progress
from .strong_search import _thanks_simulate

MODES=('reflex','planned','adaptive')


def decide(game,s,viewer,p,seed,encounter,tick,memory,state,mode,budget=PERSONA,*,variant='baseline'):
    if variant not in ('baseline','progress','continuation','combined'):raise ValueError('unknown controller variant')
    episode=f'series-{seed}-encounter-{encounter}'
    if mode=='reflex':
        if game=='goofspiel':c=competitive(make_context(s,viewer,p,'win_share',seed,tick,episode,state))
        else:c,_=thanks_observe(s,p,seed,tick,episode,state)
        d,_=score(c);return c,d,dict(method='reflex',action=d['action_id'])
    adaptive=mode=='adaptive';rng=random_stream(seed,game,encounter,tick,viewer,'npc-search')
    if game=='no_thanks' and variant in ('continuation','combined'):
        names=s.legal()
        scores,shares=_thanks_simulate(s,viewer,memory,adaptive,names,(0,)*len(names),budget.validate,rng,
                                      own_profile=p,own_state=state)
        stats=dict(sample_count=budget.validate,training_scenarios=0,validation_scenarios=budget.validate,
            continuation='owner finite reflex Policy with owner state; public rival hypotheses; NOT future root-search replanning',
            actions={name:dict(win_share=float(shares[i].mean()),
                standard_error=float(shares[i].std(ddof=1)/np.sqrt(budget.validate)),
                mean_score=float(scores[i,:,viewer].mean())) for i,name in enumerate(names)})
    else:names,scores,shares,stats=search(game,s,viewer,memory,adaptive,budget,rng)
    c=persona_context(game,s,viewer,p,seed,tick,episode,state,names,scores,shares)
    if variant in ('progress','combined'):
        progress=relative_progress(scores,viewer,direction=1 if game=='goofspiel' else -1,
                                   scale=max(s.prizes) if game=='goofspiel' else 35)
        c,d,guard=choose_with_progress(c,names,shares,progress)
        for i,name in enumerate(names):stats['actions'][name]['goal_progress']=float(progress[i].mean())
    else:d,guard=reasonable_persona(c,names,shares)
    if game=='no_thanks' and variant in ('continuation','combined'):
        c['facts']['continuation']=stats['continuation']
    stats.update(method=mode,guard=guard,action=d['action_id'],variant=variant)
    return c,d,stats


def play(game,seed,encounter,bench_seat,roster,mode,memories,*,strong=STRONG,persona=PERSONA,emit=None,variant='baseline'):
    rng=random_stream(seed,game,encounter,0,0,'world')
    if game=='goofspiel':
        s=Position.start(4,13,'ascending');order=tuple(map(int,rng.permutation(np.arange(1,14))))
        # Mixed known prize orders are public; unlike the previous two-order test.
        s=replace(s,prizes=order)
    else:
        deck=list(map(int,rng.permutation(np.arange(3,36))[:24]));s=ThanksPosition.start(4,deck.pop(0),encounter%4)
    states=[None]*4;tick=0;rows=[];failed=[0]*4;changes=[0]*4;guards=[0]*4
    while not (s.terminal if game=='goofspiel' else s.card is None and s.remaining==0):
        if game!='goofspiel' and s.card is None:s=s.draw(deck.pop(0));continue
        actors=range(4) if game=='goofspiel' else (s.turn,);moves={};details={}
        for actor in actors:
            if actor==bench_seat:
                names,sc,sh,stats=search(game,s,actor,memories[actor],True,strong,
                    random_stream(seed,game,encounter,tick,actor,'benchmark-search'))
                i=objective_choice(game,names,sc,sh,actor);chosen=names[i]
                details[actor]=dict(method='objective-search',action=chosen,search=stats)
            else:
                previous_state=states[actor]
                c,d,stats=decide(game,s,actor,roster[actor],seed,encounter,tick,memories[actor],previous_state,mode,persona,variant=variant)
                chosen=d['action_id'];states[actor]=d['next_state'];guards[actor]+=int(stats.get('guard',{}).get('guard_changed',False))
                # Counterfactual on the SAME present public state, same personality
                # and search random stream; compare read vs fixed-prior planning.
                if mode=='adaptive':
                    _,fd,fs=decide(game,s,actor,roster[actor],seed,encounter,tick,memories[actor],previous_state,'planned',persona,variant=variant)
                    changes[actor]+=int(fd['action_id']!=chosen)
                    stats['frozen_action']=fd['action_id'];stats['frozen_search']=fs
                details[actor]=dict(method=mode,action=chosen,search=stats,context=c,next_state=states[actor])
            moves[actor]=chosen
        # Every simultaneous decision has finished before any public reveal or
        # memory update. Neither player sees an already selected current bid.
        learns=[]
        if game=='goofspiel':
            bids=tuple(int(moves[a].split(':')[1]) for a in range(4));after=s.play(bids);referee(s,bids,after)
            high=max(bids)
            for a in range(4):failed[a]+=int(bids[a]==high and bids.count(high)>1)
            for observer in range(4):
                if mode!='adaptive' and observer!=bench_seat:continue
                for actor in range(4):
                    if actor!=observer:
                        rec=memories[observer].observe(s,actor,moves[actor],f'encounter-{encounter}-tick-{tick}-actor-{actor}')
                        learns.append(dict(observer=observer,actor=actor,**rec))
            before=dict(hands=s.hands,scores=s.scores,prizes=s.prizes,round=s.round,discarded=s.discarded)
            public_after=dict(hands=after.hands,scores=after.scores,round=after.round,discarded=after.discarded)
        else:
            actor=s.turn;after=s.play(moves[actor]);failed[actor]+=int(len(s.legal())==1)
            for observer in range(4):
                if observer!=actor and (mode=='adaptive' or observer==bench_seat):
                    rec=memories[observer].observe(s,actor,moves[actor],f'encounter-{encounter}-tick-{tick}-actor-{actor}')
                    learns.append(dict(observer=observer,actor=actor,**rec))
            before=asdict(s);public_after=asdict(after)
        row=dict(game=game,seed=seed,encounter=encounter,tick=tick,benchmark=bench_seat,mode=mode,
            before=before,moves=moves,decisions=details,after=public_after,learning=learns)
        if emit:emit(row)
        rows.append(row);s=after;tick+=1
        if tick>4096:raise RuntimeError('real match failed to terminate')
    scores=list(s.scores() if game!='goofspiel' else s.scores)
    if game!='goofspiel':assert scores==[thanks_referee_score(c,h) for c,h in zip(s.cards,s.chips)]
    top=(max if game=='goofspiel' else min)(scores);winners=[a for a,x in enumerate(scores) if x==top]
    credits=[1/len(winners) if a in winners else 0. for a in range(4)]
    return dict(game=game,seed=seed,encounter=encounter,mode=mode,benchmark=bench_seat,
        profiles={str(a):('benchmark' if a==bench_seat else roster[a]['id']) for a in range(4)},
        scores=scores,credits=credits,steps=tick,failures=failed,reading_changes=changes,guards=guards,
        final=asdict(s),beliefs=[m.record() for m in memories]),rows


def experiment(root,seeds,encounters=12,modes=MODES,strong=STRONG,persona=PERSONA,*,variant='baseline'):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    plan=dict(version='one-strong-three-personas-v3',variant=variant,seeds=list(seeds),encounters=encounters,
        games=['goofspiel','no_thanks'],modes=list(modes),strong=asdict(strong),persona=asdict(persona),
        benchmark='exactly one objective-focused public-information planner; not an optimal CPU claim',
        roster='seed rotates benchmark seat and missing profile; remaining 3 fixed personality axes',
        learning='persistent per observer/rival across encounters; public revealed actions only',
        worlds='same public prize orders/decks across modes; actual hidden future never provided to controllers',
        outcome='terminal winner credit primary; raw score, repeated collision/forced take, changed choices separate',
        continuation='shared root-specific own plans; rival hypotheses approximated; fresh validation scenarios',
        sources={str(p.relative_to(Path(__file__).parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(Path(__file__).parent.glob('*.py'))})
    (root/'preregister.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    results=[];started=time.monotonic()
    with (root/'trajectories.jsonl').open('w',encoding='utf-8') as f:
        def emit(row):f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        for game in plan['games']:
            for seed in seeds:
                bench=seed%4;others=[p for i,p in enumerate(PROFILES) if i!=(seed//4)%4]
                roster={a:p for a,p in zip([a for a in range(4) if a!=bench],others)}
                for mode in modes:
                    memories=[PublicMemory(game,a) for a in range(4)]
                    for encounter in range(encounters):
                        result,_=play(game,seed,encounter,bench,roster,mode,memories,strong=strong,persona=persona,emit=emit,variant=variant)
                        results.append(result)
                    print(f'{game} seed={seed} mode={mode} matches={len(results)} elapsed={time.monotonic()-started:.1f}s',flush=True)
                    (root/'partial.json').write_text(json.dumps(results,ensure_ascii=False),encoding='utf-8')
    payload=dict(plan=plan,matches=results,elapsed_seconds=time.monotonic()-started,
        trajectory_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest(),summary=summarize(results,encounters))
    (root/'evaluation.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return payload


def summarize(results,encounters):
    summary={}
    for game in ('goofspiel','no_thanks'):
        summary[game]={}
        for mode in MODES:
            rows=[r for r in results if r['game']==game and r['mode']==mode]
            if not rows:continue
            d=dict(matches=len(rows),benchmark_credit=sum(r['credits'][r['benchmark']] for r in rows),
                npc_credit=sum(sum(r['credits'])-r['credits'][r['benchmark']] for r in rows),
                npc_failures=sum(sum(x for a,x in enumerate(r['failures']) if a!=r['benchmark']) for r in rows),
                reading_changes=sum(sum(r['reading_changes']) for r in rows),guard_changes=sum(sum(r['guards']) for r in rows))
            for stage in ('early','late'):
                part=[r for r in rows if (r['encounter']<encounters//2)==(stage=='early')]
                d[stage]=dict(matches=len(part),benchmark_rate=(sum(r['credits'][r['benchmark']] for r in part)/len(part) if part else None))
            d['profiles']={p['id']:dict(appearances=sum(p['id'] in r['profiles'].values() for r in rows),
                credit=sum(sum(r['credits'][int(a)] for a,pid in r['profiles'].items() if pid==p['id']) for r in rows),
                failures=sum(sum(r['failures'][int(a)] for a,pid in r['profiles'].items() if pid==p['id']) for r in rows)) for p in PROFILES}
            summary[game][mode]=d
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--start',type=int,default=6100)
    parser.add_argument('--seeds',type=int,default=16);parser.add_argument('--encounters',type=int,default=12)
    parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    budget=SearchBudget(16,32,32,2) if args.quick else STRONG
    npc=SearchBudget(8,16,16,1) if args.quick else PERSONA
    experiment(args.root,range(args.start,args.start+args.seeds),args.encounters,strong=budget,persona=npc)
