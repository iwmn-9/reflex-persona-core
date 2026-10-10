"""Fresh paired real games: one designated NPC improves one actual decision.

Every later actor uses its actual incumbent controller. This isolates a root
intervention instead of confusing base-policy value with a new recursive policy.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.laboratory import PROFILES
from reflex.strong_search import PublicMemory, STRONG, PERSONA
from reflex.strong_table import play, decide as incumbent
from reflex.thanks_policy_rollout import decide as improved
from reflex.monte_carlo import RolloutBudget


def resolve(job):
    root,seed,encounters,bands,samples,selection,future_seeds,rival_policies=job;root=Path(root)
    bench=seed%4;others=[p for i,p in enumerate(PROFILES) if i!=(seed//4)%4]
    roster={a:p for a,p in zip([a for a in range(4) if a!=bench],others)}
    result=[]
    def label(b,kind,rival):
        if len(rival_policies)>1:return f'{b}-{kind}-{rival}'
        return str(b) if len(future_seeds)==1 else f'{b}-{kind}'
    arms=[(label(b,kind,rival),b,kind,rival) for b in bands for kind in future_seeds for rival in rival_policies]
    for label,band,kind,rival in [('baseline',None,None,None),*arms]:
        memories=[PublicMemory('no_thanks',a) for a in range(4)]
        started=time.monotonic()
        path=root/f'{seed}-{label}.jsonl'
        with path.open('w',encoding='utf-8') as f:
            def emit(row):f.write(json.dumps(row,separators=(',',':'))+'\n')
            for enc in range(encounters):
                target=[a for a in range(4) if a!=bench][enc%3]
                used=False;intervention=None
                def controller(game,s,viewer,p,nonce,encounter,tick,memory,state,mode,budget,**kw):
                    nonlocal used,intervention
                    # The diagnostic planned counterfactual must not consume the
                    # single REAL intervention or redo its expensive model.
                    if mode!='adaptive' or band is None or used or viewer!=target or s.remaining>band or len(s.legal())<2:
                        return incumbent(game,s,viewer,p,nonce,encounter,tick,memory,state,mode,budget,**kw)
                    used=True
                    c,d,stats=improved(game,s,viewer,p,nonce,encounter,tick,memory,state,mode,budget,
                        rollout=RolloutBudget(samples=samples,min_samples=samples,max_nodes=100000,max_steps=2048,rollout_policy='persona'),
                        selection=selection,future_seed=kind,rival_policy=rival,**kw)
                    intervention=dict(tick=tick,remaining=s.remaining,actor=viewer,profile=p['id'],
                        public_state=asdict(s),stats=stats['policy_rollout'],
                        baseline_action=stats['incumbent_action'],action=d['action_id'])
                    return c,d,stats
                r,_=play('no_thanks',seed,enc,bench,roster,'adaptive',memories,strong=STRONG,persona=PERSONA,
                    variant='certified_expiry',emit=emit,controller=controller)
                r.update(condition=label,target=target,target_profile=roster[target]['id'],intervention=intervention)
                result.append(r)
        print(f'seed {seed} condition {label} games {encounters} seconds {time.monotonic()-started:.1f}',flush=True)
    (root/f'{seed}.json').write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf-8')
    return seed


def run(root,start,seeds,encounters,bands,samples,selection,workers,future_seeds,rival_policies):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    base=Path(__file__).resolve().parents[1]
    plan=dict(version='actual-base-policy-root-v1',seeds=list(range(start,start+seeds)),encounters=encounters,
        bands=bands,samples=samples,selection=selection,workers=workers,future_seeds=future_seeds,
        arms=[dict(label=(f'{b}-{kind}-{rival}' if len(rival_policies)>1 else str(b) if len(future_seeds)==1 else f'{b}-{kind}'),
            band=b,future_seed=kind,rival_policy=rival) for b in bands for kind in future_seeds for rival in rival_policies],
        design='baseline and one actual intervention per designated NPC/encounter; first nonforced turn at remaining<=band; target rotates across three NPC seats',
        primary='same-owner rollout versus baseline designated NPC terminal winner credit, paired complete learning-series mean; no NPC team objective',
        secondary=['resampled-owner versus baseline and same-owner versus resampled-owner','own game score','per fixed personality result','root changes and forecast regret','complete public learning and rules'],
        controls='same actual world seed, actual strong CPU budget, fixed personality and incumbent future controller; each condition learns its own public observations',
        model='uniform public unseen deck, persistent latent public rival hypothesis, actual incumbent owner re-searches with same budget after every hypothetical future decision',
        limits=['known No Thanks only; no human-level claim','one intervention not continuous upgraded policy','rival hypotheses may be wrong; Monte Carlo SE does not cover that error','8/16 paired worlds are not exact values'],
        strong=asdict(STRONG),persona=asdict(PERSONA),
        sources={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((base/'reflex').glob('*.py'))},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=root/'preregister.json'
    if path.exists():raise FileExistsError('fresh study directory required')
    path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(resolve,(str(root),s,encounters,bands,samples,selection,future_seeds,rival_policies)) for s in plan['seeds']]
        for i,f in enumerate(as_completed(futures),1):print('complete series',i,'/',seeds,'seed',f.result(),flush=True)
    matches=[]
    for seed in plan['seeds']:matches.extend(json.loads((root/f'{seed}.json').read_text(encoding='utf-8')))
    (root/'evaluation.json').write_text(json.dumps(dict(plan=plan,matches=matches),separators=(',',':'))+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--seeds',type=int,default=24);p.add_argument('--encounters',type=int,default=4)
    p.add_argument('--bands',type=int,nargs='+',default=[6,12,23]);p.add_argument('--samples',type=int,default=8)
    p.add_argument('--selection',choices=['direct','paired_guard'],default='direct');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--future-seeds',nargs='+',choices=['same_owner','resampled'],default=['same_owner'])
    p.add_argument('--rival-policies',nargs='+',choices=['reactive','searched'],default=['reactive'])
    p.add_argument('--frozen',action='store_true');a=p.parse_args()
    if a.frozen:run(a.root,a.start,a.seeds,a.encounters,a.bands,a.samples,a.selection,a.workers,a.future_seeds,a.rival_policies)
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--start',str(a.start),'--seeds',str(a.seeds),'--encounters',str(a.encounters),
            '--bands',*[str(b) for b in a.bands],'--samples',str(a.samples),'--selection',a.selection,'--workers',str(a.workers),
            '--future-seeds',*a.future_seeds,'--rival-policies',*a.rival_policies])
