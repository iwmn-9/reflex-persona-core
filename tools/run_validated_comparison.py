"""Fresh-series conditional transfer vs unchanged global-memory benchmark."""
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
    plan=dict(version='validated-support-transfer-v1',seeds=seeds,encounters=encounters,variants=['global','validated'],
        controller_variants={'global':'certified_expiry','validated':'certified_expiry'},games=['no_thanks'],
        primary='individual NPC winner credit, paired learning series; no NPC team reward',
        secondary=['per-profile raw score/credit','pre-reveal matched-path response log loss by opportunity','transfer activation and revocation'],
        interval='10000 paired series bootstrap samples, seed 72190, no multiplicity correction',
        intervention='local support remains intact; existing EvidenceGate borrows shared predictions only after >=4 informative local comparisons and revokes after comparative failures',
        benchmark='unchanged high-budget objective CPU with global public memory in every condition',
        design_origin='strict support separation hurt matched-path natural final-card prediction; fresh seeds were defined after that diagnosis',
        strong=asdict(STRONG),persona=asdict(PERSONA),
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((Path(__file__).resolve().parents[1]/'reflex').glob('*.py'))})
    path=root/'analysis_preregister.json';assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    for kind in plan['variants']:experiment(root/kind,seeds,encounters,modes=('adaptive',),variant='certified_expiry',memory_kind=kind,games=('no_thanks',))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--start',required=True,type=int)
    p.add_argument('--seeds',default=4,type=int);p.add_argument('--encounters',default=8,type=int);p.add_argument('--frozen',action='store_true')
    a=p.parse_args()
    if a.frozen:run(a.root,range(a.start,a.start+a.seeds),a.encounters)
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--start',str(a.start),'--seeds',str(a.seeds),'--encounters',str(a.encounters)])
