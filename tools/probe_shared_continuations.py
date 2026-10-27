"""Known-root coverage diagnosis with a shared continuation bank.

No adoption evidence: fresh full-game and independent comparisons remain
mandatory. Same owner, H32, three seeds, learning and personality evaluator.
Original diagonal proposals are projected from the identical expanded paths.
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
from reflex._continuation_portfolio import portfolio_planner
from reflex._horizon_consensus import prefix_forecast
from tools.run_adverse_recovery import game_for
from tools.run_owned_value import cumulative_basis
from tools.owned_checkpoint import encoded,sha


def probe(data,tick,*,horizon=32):
    spec=data['spec'];g=game_for(spec);w=g.start();a=g.actor(g.context(w),learning='segmented')
    for t in data['result']['trace'][:tick]:
        assert digest(a.record())==t['before_owner'] and w==t['before']
        c=g.context(w,a.state)
        r=DecisionLoop.decide_batch([(a,g.request(c))],False,_counterfactual_roots=(t['action'],))[0]
        after,flow=g.transition(w,t['action'],t['response'])
        resolve_feedback(a,r['ticket'],g.feedback(w,after,t['action'],c,flow))
        assert digest(a.record())==t['owner'];w=after
    assert w==data['result']['trace'][tick]['before']
    owner=digest(a.record());axes=character_key(g.context(w));request=g.request(g.context(w,a.state))
    cap={};basis=cumulative_basis(g,True);started=time.monotonic()
    planner=portfolio_planner(a,request,w,observe=g.context,events=g.events,terminal=g.terminal,
        assess=lambda w:g.goal(w).record(),request_for=g.request,feedback=g.feedback,
        model_id=g.model_id,target=g.target,horizon=horizon,seeds=(701,702,703),
        world_model='posterior',purpose_estimator='conditional',tail_recovery=2,
        recovery_cache=64,coherent_cumulative_mean=True,cumulative_preference=basis,
        continuation_scope='shared',capture_prefixes=True,captured=cap)
    # Evaluate projected plans identically to the public FixedHorizonOwner.
    prepared={}
    def capture(contexts,decisions):
        # The public decision path adds response-learning facts after _prepare.
        # Keep that exact context instead of bypassing observation integration.
        prepared['context']=copy.deepcopy(contexts[0])
        return planner(contexts,decisions)
    DecisionLoop.decide_batch([(copy.deepcopy(a),request)],False,capture)
    context=prepared['context']
    policies=cap['forecast'].audit['continuation_portfolio']['policies']
    output={}
    for scope in ('root','shared'):
        names=[n for n,rows in cap['modeled'].items() if scope=='shared' or
            policies[n]['preferred'] in (None,rows[0]['actions'][0])]
        paths={n:cap['paths'][n] for n in names};rows={n:cap['modeled'][n] for n in names}
        forecast,_,_=prefix_forecast(context,paths,rows,horizon,actor=a,target=g.target,
            origin=g.goal(w),cumulative_preference=basis)
        result=DecisionLoop.decide_batch([(copy.deepcopy(a),request)],False,lambda cs,ds:forecast)[0]
        assert result['deliberation']['adopted'] and character_key(result['context'])==axes
        by_root={};plans=[]
        for key,root in forecast.roots.items():
            proposal=forecast.audit['plans'][key]['proposal']
            by_root[root[0]]=max(by_root.get(root[0],-1.),forecast.purpose[key])
            plans.append(dict(root=root[0],policy=policies[proposal],purpose=forecast.purpose[key]))
        selected=forecast.audit['plans'][result['deliberation']['selected_plan']]['proposal']
        output[scope]=dict(action=result['decision']['action_id'],selected_purpose=result['deliberation']['selected_purpose'],
            selected_policy=policies[selected],estimated_purpose_by_root=by_root,plans=plans)
    assert digest(a.record())==owner
    return dict(id=spec['id'],tick=tick,factual_owner=owner,world=w,forecasts=output,
        simulated_steps=cap['simulated_steps'],seconds=time.monotonic()-started,adoption=False)


def main(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if (root/'registration.json').exists():raise FileExistsError('fresh diagnostic output required')
    source=ROOT/'evidence/long_horizon/continuation_selected_confirmation/runs.jsonl.gz'
    cases=[('auction-1-10061202-shift-return',63),('auction-1-10061202-shift-return',87),
           ('combat-1-10061202-shift-return',65),('combat-2-10061202-shift-return',32),
           ('gambling-1-10061201-outside',64),('resources-1-10061202-shift-return',64)]
    files=list((ROOT/'reflex').glob('*.py'))+[Path(__file__),ROOT/'tools/run_adverse_recovery.py',
        ROOT/'tools/run_owned_value.py',ROOT/'tools/continuing_games.py']
    fingerprints={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in files}
    reg=dict(at=datetime.now(timezone.utc).isoformat(),cases=cases,seeds=[701,702,703],horizon=32,
        scopes=['root','shared'],evidence_sha256=sha(source.read_bytes()),sources=fingerprints,
        limitation=__doc__)
    (root/'registration.json').write_bytes(encoded(reg))
    with gzip.open(source,'rt') as f:data=[json.loads(s) for s in f]
    jobs=[(next(r for r in data if r['retry'] and r['spec']['id']==name),tick) for name,tick in cases]
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed(pool.submit(probe,*args) for args in jobs):
            r=future.result();rows.append(r)
            with (root/'rows.jsonl').open('ab') as f:f.write(encoded(r)+b'\n')
            print(json.dumps(dict(id=r['id'],tick=r['tick'],choices={k:v['action'] for k,v in r['forecasts'].items()},seconds=r['seconds'])),flush=True)
    assert len(rows)==len(cases)
    assert all(sha((ROOT/n).read_bytes())==h for n,h in fingerprints.items())
    original=ROOT/'evidence/long_horizon/adverse_recovery/forecast_probe/rows.jsonl'
    controls=[json.loads(s) for s in original.read_bytes().splitlines()]
    verified=0
    for row in rows:
        old=next((r for r in controls if r['id']==row['id'] and r['tick']==row['tick'] and r['arm']=='stable' and r['seeds']==[701,702,703]),None)
        if old is None:continue
        new=row['forecasts']['root']
        assert old['factual_owner']==row['factual_owner'] and old['action']==new['action']
        assert all(abs(v-new['estimated_purpose_by_root'][k])<1e-12 for k,v in old['estimated_purpose_by_root'].items())
        verified+=1
    assert verified==4
    (root/'summary.json').write_bytes(encoded(dict(complete=True,roots=len(rows),original_known_roots_reproduced=verified,
        source_unchanged=True,adoption=False,limit=__doc__)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
