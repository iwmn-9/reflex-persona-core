"""Fixed-CPU component comparison; registration is written before any match.

Four variants have identical rules, worlds, roster and benchmark algorithm.
Seeds define independent learning series. The benchmark's observations change
when NPCs change, but its implementation and budgets do not.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.strong_table import experiment
from reflex.strong_search import STRONG, PERSONA

VARIANTS=('baseline','progress','continuation','combined')


def run(root,seeds,encounters=8,variants=VARIANTS):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    plan=dict(version='goal-progress-component-comparison-v1',seeds=list(seeds),encounters=encounters,
        variants=list(variants),games=['goofspiel','no_thanks'],mode='adaptive',
        strong=asdict(STRONG),persona=asdict(PERSONA),
        interventions=dict(progress='only all-zero terminal success activates game-supplied relative progress',
                           continuation='No Thanks owner finite reflex Policy replaces eight optimized thresholds; Goof unchanged'),
        fixed_benchmark='previous objective-search controller in every variant, public memory only, no identity or hidden future',
        primary='NPC fractional winner credit; pair by series, not by turn',
        secondary=['per-profile credits','pairwise NPC vs benchmark score','relative leader deficit','forced takes/highest ties',
                   'constant-zero decisions and fallback rate','same-context persona differences'],
        uncertainty='10,000 paired series bootstrap resamples, fixed seed 90212; sampling interval only',
        limitations=['no humans measured','future real root-search replanning is still approximated',
                     'continuation component changes policy family and removes training; equal validation count, not equal total work',
                     'different sample streams after changed training are not common random outcomes'],
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((Path(__file__).resolve().parents[1]/'reflex').glob('*.py'))})
    registered=root/'analysis_preregister.json'
    if registered.exists():raise FileExistsError('use a new root for each registered comparison')
    registered.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for variant in variants:
        experiment(root/variant,seeds,encounters,modes=('adaptive',),variant=variant)
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--start',type=int,default=6700)
    p.add_argument('--seeds',type=int,default=4);p.add_argument('--encounters',type=int,default=8)
    a=p.parse_args();run(a.root,range(a.start,a.start+a.seeds),a.encounters)
