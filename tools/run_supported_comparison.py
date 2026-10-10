"""Registered matched-world supported-learning comparison, fixed strong CPU."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.strong_search import STRONG, PERSONA
from reflex.strong_table import experiment


def run(root,seeds,encounters=8):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);seeds=list(seeds)
    plan=dict(version='supported-learning-comparison-v1',seeds=seeds,encounters=encounters,
        variants=['global','supported'],controller_variant='progress',games=['goofspiel','no_thanks'],
        primary='individual NPC winner credit and paired per-series differences; NPC sum descriptive only, not a training reward',
        secondary=['per-profile scores/credit','public pre-reveal prediction log loss','forced takes/collisions','prediction vs actual outcome'],
        benchmark='same one high-budget objective CPU with global public memory in both conditions',
        support='finite declared public opportunity classes; No Thanks future draws/current card only; Goof one-bank negative control',
        interval='10000 paired series bootstrap samples; fixed seed 72190; no multiplicity adjustment',
        worlds='same ordered decks/public prize orders; different simulated draws when two banks require more sampled opponent kinds',
        strong=asdict(STRONG),persona=asdict(PERSONA),
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((Path(__file__).resolve().parents[1]/'reflex').glob('*.py'))})
    path=root/'analysis_preregister.json'
    if path.exists():raise FileExistsError('new output directory required')
    path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    for kind in plan['variants']:
        experiment(root/kind,seeds,encounters,modes=('adaptive',),variant='progress',memory_kind=kind)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--seeds',type=int,default=4);p.add_argument('--encounters',type=int,default=8)
    p.add_argument('--frozen',action='store_true')
    a=p.parse_args()
    if a.frozen:run(a.root,range(a.start,a.start+a.seeds),a.encounters)
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--start',str(a.start),'--seeds',str(a.seeds),'--encounters',str(a.encounters)])
