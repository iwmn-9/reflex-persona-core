"""Known-root sampling diagnosis; never full-game or adoption evidence.

Keep the actual public world and owner fixed and cross three declared model
banks with both samplers. These are not replacement seeds for the long study.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reflex.core import digest,character_key
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.run_adverse_recovery import decide,game_for
from tools.owned_checkpoint import encoded,sha


def probe(data,tick,arm,seeds):
    spec=data['spec'];g=game_for(spec);w=g.start();a=g.actor(g.context(w),learning='segmented')
    for t in data['result']['trace'][:tick]:
        assert digest(a.record())==t['before_owner'] and w==t['before']
        c=g.context(w,a.state)
        r=DecisionLoop.decide_batch([(a,g.request(c))],False,_counterfactual_roots=(t['action'],))[0]
        after,flow=g.transition(w,t['action'],t['response'])
        resolve_feedback(a,r['ticket'],g.feedback(w,after,t['action'],c,flow))
        assert digest(a.record())==t['owner'];w=after
    assert w==data['result']['trace'][tick]['before']
    owner=digest(a.record());character=character_key(g.context(w));started=time.monotonic()
    r,cap=decide(a,g,w,arm,seeds=seeds)
    assert character_key(r['context'])==character
    f=cap['forecast'];best={}
    for name,roots in f.roots.items():best[roots[0]]=max(best.get(roots[0],-1.),f.purpose[name])
    return dict(id=spec['id'],tick=tick,arm=arm,seeds=list(seeds),
        factual_owner=owner,action=r['decision']['action_id'],selected_purpose=r['deliberation']['selected_purpose'],
        estimated_purpose_by_root=best,simulated_steps=cap['simulated_steps'],seconds=time.monotonic()-started)


def main(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if (root/'registration.json').exists():raise FileExistsError('fresh diagnostic output required')
    source=ROOT/'evidence/long_horizon/continuation_selected_confirmation/runs.jsonl.gz'
    cases=[('auction-1-10061202-shift-return',63),('auction-1-10061202-shift-return',87),
           ('combat-1-10061202-shift-return',65),('combat-2-10061202-shift-return',32)]
    banks=[(701,702,703),(9101,9102,9103),(9201,9202,9203)]
    files=list((ROOT/'reflex').glob('*.py'))+[Path(__file__),ROOT/'tools/run_adverse_recovery.py',ROOT/'tools/run_owned_value.py',ROOT/'tools/continuing_games.py']
    fingerprints={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in files}
    reg=dict(at=datetime.now(timezone.utc).isoformat(),cases=cases,banks=banks,arms=['stable','stratified'],
        evidence_sha256=sha(source.read_bytes()),sources=fingerprints,
        limitation='Known adverse roots and only three banks. No population variance guarantee, no game-quality gate, no replacement of registered long trajectories.')
    (root/'registration.json').write_bytes(encoded(reg))
    with gzip.open(source,'rt') as f:rows=[json.loads(s) for s in f]
    jobs=[(next(r for r in rows if r['retry'] and r['spec']['id']==name),tick,arm,bank) for name,tick in cases for arm in reg['arms'] for bank in banks]
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed(pool.submit(probe,*args) for args in jobs):
            row=future.result();results.append(row)
            with (root/'rows.jsonl').open('ab') as stream:stream.write(encoded(row)+b'\n')
            print(json.dumps({k:row[k] for k in ('id','tick','arm','seeds','action','seconds')}),flush=True)
    assert all(sha((ROOT/n).read_bytes())==h for n,h in fingerprints.items())
    assert len(results)==24
    for name,tick in cases:
        group=[r for r in results if r['id']==name and r['tick']==tick]
        assert len({r['factual_owner'] for r in group})==1
    (root/'summary.json').write_bytes(encoded(dict(complete=True,evaluations=len(results),adoption=False,source_unchanged=True)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
