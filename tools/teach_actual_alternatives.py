"""Matched last-card alternatives executed by the actual public controllers.

Only the teacher resolver receives rival profiles and memories. Learner features
contain public board and OWN persona/state. Original-root replay is mandatory.
Independent controller nonces supply labels, not the recorded future seed.
"""
import argparse,copy,hashlib,json,sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from reflex.board_models import ThanksPosition
from reflex.core import NEEDS
from reflex.strong_search import PublicMemory,STRONG,PERSONA
from reflex.strong_table import play,PlaybackStart
from reflex.goal_progress import relative_progress
from reflex.laboratory import PROFILES
from tools.probe_outcome_value import features


def public_features(b,actor,action,context):
    state=context['state'] or {};needs=context['needs']
    extra=[needs[n] for n in NEEDS]+[float(state.get('age',0))/30,
        float(state.get('mode')=='principle'),float(state.get('mode_urgency',0)),float(state.get('intent_action')==action)]
    extra += [float(state.get('primary_need')==n) for n in NEEDS]
    return np.r_[features('no_thanks',b,actor,action,context),extra]


def position(b):
    return ThanksPosition(tuple(tuple(h) for h in b['cards']),tuple(b['chips']),b['turn'],b['card'],b['pot'],tuple(b['seen']),b['remaining'],tuple(b['payments']))


def collect(folder):
    folder=Path(folder);evaluation=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    finals={(r['seed'],r['encounter']):r for r in evaluation['matches']};cases=[];series=None;episode=None
    for line in (folder/'trajectories.jsonl').open(encoding='utf-8'):
        t=json.loads(line);seed=t['seed'];enc=t['encounter'];s=position(t['before']);bench=t['benchmark']
        if seed!=series:series=seed;memories=[PublicMemory('no_thanks',a) for a in range(4)]
        if (seed,enc)!=episode:episode=(seed,enc);states=[None]*4
        actor=s.turn
        if s.remaining==0 and actor!=bench and len(s.legal())>1:
            final=finals[seed,enc];roster={int(a):next(p for p in PROFILES if p['id']==pid) for a,pid in final['profiles'].items() if pid!='benchmark'}
            d=t['decisions'][str(actor)]
            cases.append(dict(seed=seed,encounter=enc,tick=t['tick'],actor=actor,benchmark=bench,roster=roster,
                position=s,memories=copy.deepcopy(memories),states=copy.deepcopy(states),reference=final,
                source_action=d['action'],features=[public_features(t['before'],actor,a,d['context']).tolist() for a in s.legal()]))
        for a,d in t['decisions'].items():
            if int(a)!=bench:states[int(a)]=d['next_state']
        for observer in range(4):
            if observer!=actor:memories[observer].observe(s,actor,t['moves'][str(actor)],f'encounter-{enc}-tick-{t["tick"]}-actor-{actor}')
    return cases


def resolve(job):
    i,c,nonces=job;s=c['position'];actor=c['actor'];start=PlaybackStart(s,(),tuple(c['states']),c['tick'])
    common=dict(game='no_thanks',encounter=c['encounter'],bench_seat=c['benchmark'],roster=c['roster'],mode='adaptive',
                strong=STRONG,persona=PERSONA,variant='certified_expiry',start=start)
    reference,_=play(seed=c['seed'],memories=copy.deepcopy(c['memories']),forced_root=(actor,c['source_action']),**common)
    old=c['reference']
    for key in ('scores','credits','final','beliefs'):
        assert json.loads(json.dumps(reference[key]))==old[key],(c['seed'],c['encounter'],c['tick'],key)
    labels=[];runs=[]
    for action in s.legal():
        targets=[];details=[]
        for nonce in range(nonces):
            seed=903071+i*nonces+nonce
            r,_=play(seed=seed,memories=copy.deepcopy(c['memories']),forced_root=(actor,action),**common)
            final=r['final'];assert final['card'] is None and final['pot']==0 and sum(final['chips'])==44 and sum(map(len,final['cards']))==24
            score=[sum(card for card in h if card-1 not in h)-chips for h,chips in zip(final['cards'],final['chips'])]
            assert score==r['scores'];assert r['credits'][actor]==float(score[actor]==min(score))/score.count(min(score))
            progress=float(relative_progress(np.asarray(score)[None,None,:],actor,direction=-1,scale=35)[0,0])
            targets.append([r['credits'][actor],progress]);details.append(dict(nonce=seed,credit=r['credits'][actor],progress=progress,scores=score))
        labels.append(np.mean(targets,axis=0).tolist());runs.append(details)
    return dict(seed=c['seed'],encounter=c['encounter'],tick=c['tick'],actor=actor,profile=c['roster'][actor]['id'],
        public_state=asdict(s),actions=list(s.legal()),source_action=c['source_action'],features=c['features'],targets=labels,
        independent_nonce_branches=runs,exact_reference=True,labels_are_simulated=True)


def run(root,source,nonces=3,workers=4):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);source=Path(source).resolve();base=Path(__file__).resolve().parents[1]
    plan=dict(version='matched-actual-terminal-controller-teachers-v1',source=str(source.name),nonces=nonces,workers=workers,
        selection='every nonforced NPC last-card turn in every global series/encounter; no winner or candidate-result selection',
        targets=['terminal own winner credit','terminal own relative progress'],
        input='public board and OWN traits, values, needs and state; no rival profile, private memory or future controller nonce',
        reference='original root with original nonce must match complete terminal scores, credits, world and every public memory',
        label_futures='all four actual baseline controllers replan and learn after each public reveal, independent future controller nonces',
        limitations=['same known game and controller families','simulation targets, not human observed experience',
                     'terminal-only exposure','matched labels do not guarantee changed-policy future values'],
        strong=asdict(STRONG),persona=asdict(PERSONA),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((base/'reflex').glob('*.py'))})
    path=root/'preregister.json';assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    cases=[]
    for shard in range(4):cases.extend(collect(source/f'shard-{shard}/global'))
    assert cases
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=[]
        for row in pool.map(resolve,[(i,c,nonces) for i,c in enumerate(cases)],chunksize=1):
            rows.append(row)
            if len(rows)%16==0:print('paired cases',len(rows),'/',len(cases),flush=True)
    result=dict(plan=plan,cases=rows,summary=dict(cases=len(rows),matched_alternative_branches=len(rows)*2*nonces,
        original_root_reference_replays=len(rows),independent_series=len({r['seed'] for r in rows}),
        informative_credit_cases=sum(np.ptp(np.asarray(r['targets'])[:,0])>0 for r in rows),
        informative_progress_cases=sum(np.ptp(np.asarray(r['targets'])[:,1])>1e-12 for r in rows)))
    # np comparisons produce scalar booleans; cast for portable JSON.
    result['summary']={k:int(v) for k,v in result['summary'].items()}
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(result['summary'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True)
    p.add_argument('--nonces',type=int,default=3);p.add_argument('--workers',type=int,default=4);p.add_argument('--frozen',action='store_true');a=p.parse_args()
    if a.frozen:run(a.root,a.source,a.nonces,a.workers)
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--source',str(Path(a.source).resolve()),'--nonces',str(a.nonces),'--workers',str(a.workers)])
