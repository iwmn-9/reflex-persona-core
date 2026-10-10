"""Fresh paired trials of incumbent-guarded learning and its finite continuation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.strong_search import STRONG,PERSONA
from reflex.strong_table import experiment


def run(root,seeds,encounters=8):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);seeds=list(seeds)
    controllers={'global':'certified_expiry','guarded':'certified_expiry','integrated':'settlement'}
    memories={'global':'global','guarded':'guarded','integrated':'guarded'}
    plan=dict(version='incumbent-guarded-integrated-v1',seeds=seeds,encounters=encounters,
        variants=list(controllers),controller_variants=controllers,memory_kinds=memories,games=['no_thanks'],
        primary='integrated minus global individual NPC winner credit, paired learning series; no NPC team objective',
        secondary=['guarded minus global; integrated minus guarded','per-profile score/credit',
            'same global paths, pre-reveal prediction loss by public opportunity','adoption and revocation counts'],
        interval='10000 paired series bootstrap samples, seed 72190; no multiplicity correction',
        design_origin='strict separation and local-first borrowing both worsened sparse terminal predictions; retain established shared incumbent until specialization has local comparative proof',
        intervention='unchanged EvidenceGate controls; shared fallback versus local challenger; optional SAME-owner finite public final-card continuation',
        benchmark='unchanged high-budget objective CPU with global public memory in every condition',
        strong=asdict(STRONG),persona=asdict(PERSONA),
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((Path(__file__).resolve().parents[1]/'reflex').glob('*.py'))})
    path=root/'analysis_preregister.json';assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    for kind in plan['variants']:
        experiment(root/kind,seeds,encounters,modes=('adaptive',),variant=controllers[kind],memory_kind=memories[kind],games=('no_thanks',))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--start',required=True,type=int)
    p.add_argument('--seeds',default=4,type=int);p.add_argument('--encounters',default=8,type=int);p.add_argument('--frozen',action='store_true')
    a=p.parse_args()
    if a.frozen:run(a.root,range(a.start,a.start+a.seeds),a.encounters)
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--start',str(a.start),'--seeds',str(a.seeds),'--encounters',str(a.encounters)])
