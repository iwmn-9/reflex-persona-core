"""Compare two implementations on all 80 stored learning histories in Colab.

Actions are fixed from the existing study. This is implementation equivalence,
not new games, a replacement Windows audit, or an improvement in decision quality.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import verify_colab_optimized_root as contract


def worker(source, history, output):
    source=source.resolve()
    contract.require(all(os.environ.get(k)==v for k,v in contract.ARITHMETIC.items()),
                     'registered arithmetic environment required')
    sys.path.insert(0,str(source))
    import numpy as np
    contract.require(platform.python_version()=='3.12.14' and np.__version__=='2.3.5',
                     'Python 3.12.14 and NumPy 2.3.5 required')
    from reflex.core import character_key,digest
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    from tools.run_continuation_portfolio import game_for
    from tools.run_owned_value import law,sample
    contract.check_imports(source)
    before=contract.sources(source);raw=history.read_bytes()
    rows=[json.loads(line) for line in gzip.decompress(raw).splitlines()]
    assert len(rows)==80 and len({(r['spec']['id'],r['retry']) for r in rows})==80
    cases=[];started=time.process_time()
    for row in rows:
        spec=row['spec'];g=game_for(spec);world=g.start()
        actor=g.actor(g.context(world),learning='segmented')
        axes=character_key(g.context(world));rng=np.random.default_rng(spec['seed'])
        trace=row['result']['trace'];assert row['result']['complete'] and len(trace)==spec['turns']==128
        digests=[];original_mismatches=[]
        for entry in trace:
            assert world==entry['before']
            context=g.context(world,actor.state)
            result=DecisionLoop.decide_batch([(actor,g.request(context))],False,
                _counterfactual_roots=(entry['action'],))[0]
            truth=law(world['tick'],spec['turns'],spec['regime'])
            assert truth==entry['truth'] and sample(truth,rng.random())==entry['response']
            after,flow=g.transition(world,entry['action'],entry['response'])
            observed=resolve_feedback(actor,result['ticket'],g.feedback(world,after,entry['action'],context,flow))
            assert after==entry['after'] and character_key(result['context'])==axes
            saved=json.loads(contract.encoded(actor.record()))
            # Every actual update crosses the supported persistence boundary.
            actor=DecisionLoop.from_record(context,saved,predictor=actor.predictor)
            assert actor.record()==saved
            digests.append(contract.sha(contract.encoded(contract.plain(
                dict(decision=result,observation=observed,owner=saved,world=after),np))))
            if digest(saved)!=entry['owner']:original_mismatches.append(entry['tick'])
            world=after
        assert g.terminal(world) and world==row['result']['final']
        cases.append(dict(id=spec['id'],retry=row['retry'],steps=len(trace),step_sha256=digests,
            original_windows_hash_mismatch_ticks=original_mismatches))
        print(json.dumps(dict(completed_histories=len(cases),id=spec['id'],retry=row['retry'])),flush=True)
    contract.check_imports(source)
    assert contract.sources(source)==before and history.read_bytes()==raw
    result=dict(complete=True,adoption=False,histories=80,actual_steps=10240,
        cases=cases,history_sha256=contract.sha(raw),sources=before,cpu_seconds=time.process_time()-started,
        runtime=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
                     arithmetic={k:os.environ[k] for k in contract.ARITHMETIC}),
        tool_sha256=contract.sha(Path(__file__).read_bytes()),
        source_unchanged=True,input_unchanged=True,
        scope='Stored actual actions, same Linux source/runtime comparison. Historical Windows hash mismatches remain failures; not new quality evidence.')
    with output.open('xb') as stream:stream.write(contract.encoded(result))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--history',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();worker(a.source,a.history,a.output)


if __name__=='__main__':main()
