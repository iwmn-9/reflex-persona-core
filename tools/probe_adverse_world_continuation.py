"""Known-state intervention on future regime changes, not adoption evidence.

Keep the observed owner, every personality parameter, future owner learning,
candidate bank and sample seeds. Only vary the existing outer world model.
Previously verified posterior forecasts are reused when the factual root and
configuration match; no favorable seed bank or true world law enters planning.
"""
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reflex.core import digest, character_key
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from reflex._horizon_consensus import consensus_planner
from tools.run_adverse_recovery import game_for
from tools.run_owned_value import cumulative_basis
from tools.owned_checkpoint import encoded,sha


def probe(data,tick,sampling,world_model):
    spec=data['spec'];g=game_for(spec);w=g.start();a=g.actor(g.context(w),learning='segmented')
    for t in data['result']['trace'][:tick]:
        assert w==t['before'] and digest(a.record())==t['before_owner']
        c=g.context(w,a.state)
        r=DecisionLoop.decide_batch([(a,g.request(c))],False,_counterfactual_roots=(t['action'],))[0]
        after,flow=g.transition(w,t['action'],t['response'])
        resolve_feedback(a,r['ticket'],g.feedback(w,after,t['action'],c,flow))
        assert digest(a.record())==t['owner'];w=after
    factual=data['result']['trace'][tick]
    assert w==factual['before'] and digest(a.record())==factual['before_owner']
    owner=digest(a.record());axes=character_key(g.context(w));request=g.request(g.context(w,a.state))
    cap={};started=time.monotonic()
    planner=consensus_planner(a,request,w,observe=g.context,events=g.events,terminal=g.terminal,
        assess=lambda w:g.goal(w).record(),request_for=g.request,feedback=g.feedback,
        model_id=g.model_id,target=g.target,horizon=32,minimum_horizon=32,seeds=(701,702,703),
        world_model=world_model,purpose_estimator='conditional',tail_recovery=2,recovery_cache=64,
        coherent_cumulative_mean=True,cumulative_preference=cumulative_basis(g,True),
        sampling=sampling,captured=cap)
    r=DecisionLoop.decide_batch([(copy.deepcopy(a),request)],False,planner)[0]
    assert r['deliberation']['adopted'] and character_key(r['context'])==axes
    assert digest(a.record())==owner
    f=cap['forecast'];best={}
    for name,roots in f.roots.items():best[roots[0]]=max(best.get(roots[0],-1.),f.purpose[name])
    return dict(id=spec['id'],tick=tick,sampling=sampling,world_model=world_model,
        factual_owner=owner,world=w,action=r['decision']['action_id'],
        selected_purpose=r['deliberation']['selected_purpose'],estimated_purpose_by_root=best,
        simulated_steps=cap['simulated_steps'],seconds=time.monotonic()-started,adoption=False)


def main(output,live_root):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'registration.json').exists():raise FileExistsError('fresh diagnostic required')
    original=ROOT/'evidence/long_horizon/continuation_selected_confirmation/runs.jsonl.gz'
    controls=ROOT/'evidence/long_horizon/adverse_recovery/forecast_probe/rows.jsonl'
    live=Path(live_root)/'adverse-gambling-1-10061201-outside-stable/result.json'
    with gzip.open(original,'rt') as f:history=[json.loads(s) for s in f]
    previous=[json.loads(s) for s in controls.read_bytes().splitlines()]
    cases=[('auction-1-10061202-shift-return',63),('auction-1-10061202-shift-return',87),
        ('combat-1-10061202-shift-return',65),('combat-2-10061202-shift-return',32),
        ('gambling-1-10061201-outside',6)]
    live_result=json.loads(live.read_bytes())
    assert live_result['complete'] and len(live_result['trace'])==128
    data=[next(r for r in history if r['retry'] and r['spec']['id']==name) for name,_ in cases[:-1]]
    data.append(dict(spec=live_result['configuration']['spec'],result=live_result))
    # Bind all external inputs before execution. The actual trace, including
    # the loss-triggering observation history, cannot be swapped afterwards.
    (out/'inputs.json.gz').write_bytes(gzip.compress(encoded(data),mtime=0))
    files=list((ROOT/'reflex').glob('*.py'))+list((ROOT/'tools').glob('*.py'))
    sources={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in files}
    reg=dict(at=datetime.now(timezone.utc).isoformat(),cases=cases,seeds=[701,702,703],horizon=32,
        world_models=['posterior','persistent'],sampling=['independent','stratified'],sources=sources,
        inputs_sha256=sha((out/'inputs.json.gz').read_bytes()),
        original_sha256=sha(original.read_bytes()),previous_forecasts_sha256=sha(controls.read_bytes()),
        live_result_sha256=sha(live.read_bytes()),limitation=__doc__)
    (out/'registration.json').write_bytes(encoded(reg))
    jobs=[];rows=[]
    for item,(name,tick) in zip(data,cases):
        for sampling in reg['sampling']:
            arm='stable' if sampling=='independent' else 'stratified'
            old=next((r for r in previous if r['id']==name and r['tick']==tick and r['arm']==arm and r['seeds']==reg['seeds']),None)
            if old is not None:
                assert old['factual_owner']==item['result']['trace'][tick]['before_owner']
                rows.append(dict(old,world_model='posterior',sampling=sampling,reused_source=str(controls.relative_to(ROOT))))
            else:jobs.append((item,tick,sampling,'posterior'))
            jobs.append((item,tick,sampling,'persistent'))
    with (out/'rows.jsonl').open('wb') as f:
        for row in rows:f.write(encoded(row)+b'\n')
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed(pool.submit(probe,*job) for job in jobs):
            row=future.result();rows.append(row)
            with (out/'rows.jsonl').open('ab') as f:f.write(encoded(row)+b'\n')
            print(json.dumps({k:row[k] for k in ('id','tick','sampling','world_model','action','seconds')}),flush=True)
    assert len(rows)==20
    assert all(sha((ROOT/n).read_bytes())==h for n,h in sources.items())
    for name,tick in cases:
        group=[r for r in rows if r['id']==name and r['tick']==tick]
        assert len(group)==4 and len({r['factual_owner'] for r in group})==1
    (out/'summary.json').write_bytes(encoded(dict(complete=True,roots=5,forecasts=20,
        reused=8,new=12,source_unchanged=True,adoption=False,limitation=__doc__)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--live-root',required=True)
    a=p.parse_args();main(a.output,a.live_root)
