"""Post-decision oracle diagnostic; NEVER a controller input or live reward.

For every preselected first-encounter intervention, replay the original root
and its alternative under the actual incumbent future controllers and world.
True rival profiles/future cards belong only to this evaluator.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from reflex.board_models import ThanksPosition
from reflex.laboratory import PROFILES
from reflex.strong_search import PublicMemory, STRONG, PERSONA, random_stream
from reflex.strong_table import PlaybackStart, play


def collect(source,seeds):
    source=Path(source);plan=json.loads((source/'preregister.json').read_text(encoding='utf-8'));cases=[]
    for seed in seeds:
        finals=json.loads((source/f'{seed}.json').read_text(encoding='utf-8'))
        for band in plan['bands']:
            memories=[PublicMemory('no_thanks',a) for a in range(4)];states=[None]*4
            final=next(r for r in finals if r['condition']==str(band) and r['encounter']==0)
            roster={int(a):next(p for p in PROFILES if p['id']==pid) for a,pid in final['profiles'].items() if pid!='benchmark'}
            for line in (source/f'{seed}-{band}.jsonl').open(encoding='utf-8'):
                t=json.loads(line)
                if t['encounter']!=0:break
                b=t['before'];actor=b['turn'];s=ThanksPosition(tuple(map(tuple,b['cards'])),tuple(b['chips']),actor,b['card'],b['pot'],tuple(b['seen']),b['remaining'],tuple(b['payments']))
                d=t['decisions'][str(actor)];rs=d['search'].get('policy_rollout')
                if rs:
                    deck=list(map(int,random_stream(seed,'no_thanks',0,0,0,'world').permutation(np.arange(3,36))[:24]))
                    cases.append(dict(seed=seed,band=band,tick=t['tick'],actor=actor,benchmark=t['benchmark'],position=s,
                        profile=roster[actor]['id'],roster=roster,memories=copy.deepcopy(memories),states=copy.deepcopy(states),
                        deck=tuple(c for c in deck if c not in s.seen),reference=final,chosen=d['action'],
                        incumbent=d['search']['incumbent_action'],forecast=rs))
                if actor!=t['benchmark']:states[actor]=copy.deepcopy(d['next_state'])
                for viewer in range(4):
                    if viewer!=actor:memories[viewer].observe(s,actor,t['moves'][str(actor)],f'encounter-0-tick-{t["tick"]}-actor-{actor}')
    return cases


def resolve(c):
    start=PlaybackStart(c['position'],c['deck'],tuple(c['states']),c['tick']);results={}
    for action in c['position'].legal():
        r,_=play('no_thanks',c['seed'],0,c['benchmark'],c['roster'],'adaptive',copy.deepcopy(c['memories']),
            strong=STRONG,persona=PERSONA,variant='certified_expiry',start=start,forced_root=(c['actor'],action))
        if action==c['chosen']:
            for key in ('scores','credits','final','beliefs'):assert json.loads(json.dumps(r[key]))==c['reference'][key],(c['seed'],c['band'],key)
        results[action]=dict(credit=r['credits'][c['actor']],score=r['scores'][c['actor']],scores=r['scores'])
    names=c['position'].legal();best=max(results[a]['credit'] for a in names)
    actual_gain=results[c['chosen']]['credit']-results[c['incumbent']]['credit']
    return dict(seed=c['seed'],band=c['band'],tick=c['tick'],profile=c['profile'],public_state=asdict(c['position']),
        chosen=c['chosen'],incumbent=c['incumbent'],forecast=c['forecast'],actual_world_oracle=results,
        actual_gain=actual_gain,chosen_credit_regret=best-results[c['chosen']]['credit'],
        incumbent_credit_regret=best-results[c['incumbent']]['credit'],original_root_exact_replay=True)


def run(root,source,seeds,workers):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);source=Path(source).resolve();base=Path(__file__).resolve().parents[1]
    plan=dict(version='actual-base-policy-oracle-diagnostic-v1',source=str(source),seeds=seeds,encounters=[0],workers=workers,
        selection='every first-encounter intervention in all predeclared bands for fixed seeds; no winner/action/result filtering',
        scope='post hoc deterministic actual-world oracle; future deck/rival private profiles never enter the controller; not expected action values',
        reference='chosen-root replay must exactly match complete scores, winner credits, final world and every public memory',
        limitations=['one actual future per state, not expected return','reuses primary games; no extra independent strength games','known opponent families only'],
        sources={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((base/'reflex').glob('*.py'))})
    path=root/'preregister.json'
    if path.exists():raise FileExistsError('fresh diagnostic required')
    path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    cases=collect(source,seeds);rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,row in enumerate(pool.map(resolve,cases),1):
            rows.append(row)
            if i%8==0:print('actual oracle cases',i,'/',len(cases),flush=True)
    summary=[]
    for band in sorted({r['band'] for r in rows}):
        rr=[r for r in rows if r['band']==band]
        summary.append(dict(band=band,cases=len(rr),changed=sum(r['chosen']!=r['incumbent'] for r in rr),
            mean_actual_gain=float(np.mean([r['actual_gain'] for r in rr])),
            chosen_credit_regret=float(np.mean([r['chosen_credit_regret'] for r in rr])),
            incumbent_credit_regret=float(np.mean([r['incumbent_credit_regret'] for r in rr])),
            changed_better=sum(r['actual_gain']>0 for r in rr),changed_worse=sum(r['actual_gain']<0 for r in rr)))
    result=dict(plan=plan,cases=rows,summary=summary,reference_replays=len(rows),alternative_branches=len(rows))
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True)
    p.add_argument('--start',type=int,required=True);p.add_argument('--seeds',type=int,default=24);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--frozen',action='store_true');a=p.parse_args()
    if a.frozen:run(a.root,a.source,list(range(a.start,a.start+a.seeds)),a.workers)
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--source',str(Path(a.source).resolve()),'--start',str(a.start),'--seeds',str(a.seeds),'--workers',str(a.workers)])
