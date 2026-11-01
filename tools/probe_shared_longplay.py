"""Existing preparation mechanism: test whether broader tails cause deferral.

Known deterministic diagnostic, not the independent gaming confirmation. The
single path suffices because the adapter has one possible event. Both arms use
H32, unchanged personality and actual learning over 128 resolved decisions.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reflex.core import digest,character_key
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.run_continuation_portfolio import execution,game_for
from tools.owned_checkpoint import encoded,sha


def run(root,spec):
    result=execution(root,spec,True)
    assert result['complete']
    game=game_for(spec);world=game.start();actor=game.actor(game.context(world),learning='segmented')
    axes=character_key(game.context(world))
    for t in result['trace']:
        assert world==t['before'] and digest(actor.record())==t['before_owner']
        context=game.context(world,actor.state)
        r=DecisionLoop.decide_batch([(actor,game.request(context))],False,_counterfactual_roots=(t['action'],))[0]
        after,flow=game.transition(world,t['action'],t['response'])
        resolve_feedback(actor,r['ticket'],game.feedback(world,after,t['action'],context,flow))
        assert digest(actor.record())==t['owner'] and character_key(game.context(after))==axes
        world=after
    checkpoint=json.loads((root/'checkpoint.json').read_bytes())['payload']
    assert actor.record()==checkpoint['owner'] and world==checkpoint['world']
    return dict(id=spec['id'],scope=spec['controller']['continuation_scope'],profile=spec['profile'],
        score=result['score'],final=world,steps=len(result['trace']),
        actual_owner_hash_replay=True,result_sha256=sha((root/'result.json').read_bytes()))


def main(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():raise FileExistsError('fresh complete diagnostic required')
    jobs=[]
    for profile in (0,2):
        for scope in ('root','shared'):
            jobs.append(dict(id=f'chain-{profile}-{scope}',mechanism='chain',length=5,payout=14.,
                changes=[],turns=128,profile=profile,controller=dict(horizon_consensus=True,
                    minimum_horizon=32,world_model='posterior',purpose_estimator='conditional',
                    tail_recovery=2,recovery_cache=64,coherent_cumulative_mean=True,continuation_scope=scope)))
    files=list((ROOT/'reflex').glob('*.py'))+[Path(__file__),ROOT/'tools/run_continuation_portfolio.py',
        ROOT/'tools/preparation_world.py',ROOT/'tools/continuing_games.py',ROOT/'tools/run_owned_value.py']
    sources={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in files}
    (root/'plan.json').write_bytes(encoded(dict(at=datetime.now(timezone.utc).isoformat(),specs=jobs,sources=sources,limit=__doc__)))
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed(pool.submit(run,root/s['id'],s) for s in jobs):
            r=future.result();rows.append(r);print(json.dumps(r),flush=True)
            with (root/'completed.jsonl').open('ab') as f:f.write(encoded(r)+b'\n')
    assert len(rows)==4 and all(sha((ROOT/n).read_bytes())==h for n,h in sources.items())
    (root/'summary.json').write_bytes(encoded(dict(complete=True,rows=rows,adoption=False,
        source_unchanged=True,actual_steps=sum(r['steps'] for r in rows),limit=__doc__)))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
