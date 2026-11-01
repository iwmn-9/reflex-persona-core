"""Additional cross-platform replay diagnostic, never a relaxed worker audit.

Original worker replay still requires exact owner hashes on its pinned runtime.
This records, rather than erases, deviations when replayed on another runtime.
Only factual forced actions are replayed; planning decisions are not recomputed.
"""
import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reflex.core import character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.owned_checkpoint import encoded, sha
from tools.run_adverse_recovery import game_for
from tools.run_owned_value import law, sample


def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return [d for k in a for d in differences(a[k], b[k], path+'/'+str(k))]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for k, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, path+'/'+str(k))]
    return [] if a == b else [dict(path=path, local=a, saved=b)]


def probe(root):
    rows=[]
    for folder in sorted(Path(root).glob('adverse-*')):
        path=folder/'checkpoint.json'
        if not path.exists():continue
        raw=path.read_bytes();envelope=json.loads(raw);p=envelope['payload']
        assert sha(encoded(p))==envelope['sha256']
        cfg=p['contract']['configuration'];spec=cfg['spec'];g=game_for(spec)
        w=g.start();actor=g.actor(g.context(w),learning='segmented')
        axes=character_key(g.context(w));rng=np.random.default_rng(spec['seed'])
        mismatches=[]
        for t in p['trace']:
            assert w==t['before']
            c=g.context(w,actor.state)
            r=DecisionLoop.decide_batch([(actor,g.request(c))],False,_counterfactual_roots=(t['action'],))[0]
            truth=law(w['tick'],spec['turns'],spec['regime'])
            assert truth==t['truth'] and sample(truth,rng.random())==t['response']
            after,flow=g.transition(w,t['action'],t['response'])
            resolve_feedback(actor,r['ticket'],g.feedback(w,after,t['action'],c,flow))
            assert after==t['after'] and character_key(g.context(after))==axes
            if digest(actor.record())!=t['owner']:mismatches.append(t['tick'])
            w=after
        assert w==p['world'] and rng.bit_generator.state==p['rng']
        delta=differences(actor.record(),p['owner'])
        floats=[d for d in delta if type(d['local']) is float and type(d['saved']) is float]
        rows.append(dict(id=spec['id'],arm=cfg['arm'],steps=len(p['trace']),
            checkpoint_sha256=sha(raw),source_runtime=p['contract']['runtime'],
            physical_rng_personality_replay=True,owner_hash_mismatch_ticks=mismatches,
            final_owner_differences=delta,non_float_differences=len(delta)-len(floats),
            max_abs_float_difference=max((abs(d['local']-d['saved']) for d in floats),default=0.)))
    return dict(runtime=dict(system=platform.system(),python=sys.version,numpy=np.__version__),
        cases=rows,actual_steps=sum(r['steps'] for r in rows),
        max_abs_float_difference=max((r['max_abs_float_difference'] for r in rows),default=0.),
        non_float_differences=sum(r['non_float_differences'] for r in rows),
        semantics='Additional diagnostic only. Exact pinned-runtime worker hashes remain mandatory. No stored history, tolerance, adoption gate or controller was changed. Forced factual actions, not planner parity.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=probe(args.root)
    Path(args.output).write_bytes(encoded(result))
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
