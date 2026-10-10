"""Registered fresh-series comparison of identical supported-memory controllers.

Only public final-card continuation changes. Earlier decisions and the high-
budget objective benchmark retain their existing algorithms and budgets.
"""
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
    plan=dict(version='finite-settlement-comparison-v1',seeds=seeds,encounters=encounters,
        variants=['supported','settlement'],controller_variants={'supported':'certified_expiry','settlement':'settlement'},games=['no_thanks'],
        primary='individual NPC winner credit and paired per-series differences',
        secondary=['per-profile score/credit','same-history first root divergences','prediction/actual continuation mismatch'],
        intervention='all public current-card settlement branches; future owner uses same fixed-persona selector and state transition; no sampling error added to finite-model guard',
        benchmark='same one high-budget objective CPU with global public memory',
        interval='10000 paired series bootstrap samples; seed 72190; no multiplicity adjustment',
        strong=asdict(STRONG),persona=asdict(PERSONA),
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((Path(__file__).resolve().parents[1]/'reflex').glob('*.py'))})
    path=root/'analysis_preregister.json';assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    for condition in plan['variants']:
        experiment(root/condition,seeds,encounters,modes=('adaptive',),variant=plan['controller_variants'][condition],memory_kind='supported',games=('no_thanks',))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--seeds',type=int,default=4);p.add_argument('--encounters',type=int,default=8)
    a=p.parse_args();run(a.root,range(a.start,a.start+a.seeds),a.encounters)
