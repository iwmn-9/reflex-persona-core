"""Fresh matched series for real resource-proxy expiry (not search truncation)."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.strong_table import experiment
from reflex.strong_search import STRONG, PERSONA


def run(root,start,seeds=4,certified=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    plan=dict(version='real-expiry-comparison-v1',seeds=list(range(start,start+seeds)),encounters=4,
        variants=['progress','certified_expiry' if certified else 'horizon_progress'],games=['no_thanks'],mode='adaptive',
        strong=asdict(STRONG),persona=asdict(PERSONA),
        intervention=('Real remaining==0 AND next seat chips==0 proves settlement before any owner redecision' if certified else 'Only real remaining==0 expires safety need/security/neuroticism future-resource proxies; fixed axes untouched'),
        primary='NPC fractional winner credit, paired by independent learning series',
        secondary=['score leader deficit','per-profile credit','same-state different-persona choices','forced takes'],
        selection=('Refinement after broad expiry regression on 6800..6815; fresh balanced seeds 6900..6915' if certified else 'Known exact last-card 17-point error; fresh balanced seeds 6800..6815, not used for fitting'),
        bootstrap_seed=90214 if certified else 90213,
        uncertainty='10,000 paired series bootstrap samples',
        limitations=['16 series, 128 games total, not 128 independent samples','known rules/controllers, no human trials'],
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((Path(__file__).resolve().parents[1]/'reflex').glob('*.py'))})
    path=root/'analysis_preregister.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    for variant in plan['variants']:
        experiment(root/variant,plan['seeds'],plan['encounters'],modes=('adaptive',),variant=variant,games=('no_thanks',))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--certified',action='store_true');a=p.parse_args();run(a.root,a.start,certified=a.certified)
