"""Recheck the original twelve preparation mechanisms with the single-model reader.

Fixed H32, deterministic one-path diagnostic and original payout schedules.
No world-game comparison or independent confirmation is replaced by this check.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reflex.core import character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.current_model_reader import CurrentModelReader
from tools.owned_checkpoint import encoded, sha
from tools.run_continuation_portfolio import execution, game_for


def audit(folder, spec, mode):
    folder=Path(folder)
    envelope=json.loads((folder/'checkpoint.json').read_bytes());saved=envelope['payload']
    assert sha(encoded(saved))==envelope['sha256']
    result=json.loads((folder/'result.json').read_bytes())
    assert result['complete'] and result['configuration']==saved['contract']['configuration']
    assert result['configuration']['spec']==spec and result['configuration'].get('reader_selection','advice')==mode
    assert result['trace']==saved['trace'] and len(result['trace'])==spec['turns']
    game=game_for(spec);world=game.start();context=game.context(world)
    actor=game.actor(context,learning='segmented');axes=character_key(context)
    if mode=='current_model':actor.predictor=CurrentModelReader.from_reader(actor.predictor,context['scope'])
    rng=np.random.default_rng(spec.get('seed',0))
    for row in result['trace']:
        assert row['before']==world and row['before_owner']==digest(actor.record())
        context=game.context(world,actor.state)
        chosen=DecisionLoop.decide_batch([(actor,game.request(context))],False,_counterfactual_roots=(row['action'],))[0]
        rng.random();assert row['truth']=={'0':1.} and row['response']=='0'
        after,flow=game.transition(world,row['action'],'0')
        for tick,payout in spec['changes']:
            if after['tick']==tick:after['payout']=payout
        resolve_feedback(actor,chosen['ticket'],game.feedback(world,after,row['action'],context,flow))
        assert after==row['after'] and digest(actor.record())==row['owner']
        assert character_key(game.context(after))==axes
        world=after
    assert game.terminal(world) and world==saved['world']==result['final']
    assert actor.record()==saved['owner'] and rng.bit_generator.state==saved['rng']
    assert result['score']==game.goal(world).value
    return dict(complete=True,actual_steps=len(result['trace']),score=result['score'],
        source_runtime=saved['contract']['runtime'],owner_replayed=True,persona=axes,
        files={n:sha((folder/n).read_bytes()) for n in ('checkpoint.json','result.json')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--frozen',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve()
    if not args.frozen:
        original=ROOT/'evidence/long_horizon/continuation_fixed32_screen/plan.json'
        parent=json.loads(original.read_bytes());assert len(parent['specs'])==12
        root.mkdir(parents=True,exist_ok=True)
        if (root/'plan.json').exists():raise FileExistsError('fresh diagnostic registration required')
        (root/'original_plan.json').write_bytes(original.read_bytes())
        plan=dict(original_plan_sha256=sha(original.read_bytes()),specs=parent['specs'],
            modes=['advice','current_model'],criterion='All twelve complete; no candidate final-score loss under the unchanged 1e-12 boundary. Audit every actual owner and unchanged persona. No future payout schedule revealed; shock prefixes must match.',
            arithmetic={k:os.environ.get(k) for k in ('OPENBLAS_CORETYPE','NPY_DISABLE_CPU_FEATURES','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            runtime=dict(python=sys.version,numpy=np.__version__),
            adoption=False,source_relation='Same reflex runtime and current-model reader as a14d164; only an existing deterministic diagnostic runner exposes reader selection.')
        (root/'plan.json').write_bytes(encoded(plan))
        from tools.freeze_experiment import dispatch
        dispatch(root,Path(__file__).name,[])
        return
    plan=json.loads((root/'plan.json').read_bytes())
    assert sha((root/'original_plan.json').read_bytes())==plan['original_plan_sha256']
    assert plan['specs']==json.loads((root/'original_plan.json').read_bytes())['specs']
    assert plan['runtime']==dict(python=sys.version,numpy=np.__version__)
    assert all(os.environ.get(k)==v for k,v in plan['arithmetic'].items())
    sources=json.loads((root/'source_snapshot.json').read_bytes())
    for name,checksum in sources.items():assert sha((ROOT/name).read_bytes())==checksum
    completed=[];outputs={}
    for spec in plan['specs']:
        pair=[]
        for mode in plan['modes']:
            folder=root/spec['id']/mode
            (root/'status.json').write_bytes(encoded(dict(complete=False,state='running',pid=os.getpid(),case=spec['id'],mode=mode,pairs_complete=len(completed),checkpoint=str(folder/'checkpoint.json'),expected_turns=spec['turns'])))
            if not (folder/'result.json').exists():execution(folder,spec,True,reader_selection=mode)
            receipt=audit(folder,spec,mode);(folder/'audit.json').write_bytes(encoded(receipt))
            pair.append(receipt);outputs[(spec['id'],mode)]=json.loads((folder/'result.json').read_bytes())
        assert pair[0]['source_runtime']==pair[1]['source_runtime']
        completed.append(dict(id=spec['id'],delta=pair[1]['score']-pair[0]['score'],arms=pair))
        print(encoded(completed[-1]).decode(),flush=True)
    prefixes=[]
    for spec in plan['specs']:
        if not spec['id'].endswith('-loss'):continue
        normal=spec['id'].removesuffix('-loss')+'-stable';shock=spec['changes'][0][0]
        for mode in plan['modes']:
            left=outputs[(normal,mode)]['trace'];right=outputs[(spec['id'],mode)]['trace']
            assert all(all(a[k]==b[k] for k in ('before','before_owner','action')) for a,b in zip(left[:shock],right[:shock]))
            prefixes.append(dict(id=spec['id'],mode=mode,identical_pre_reveal_choices=shock))
    result=dict(complete=True,pairs=completed,preparation_no_loss=all(p['delta']>=-1e-12 for p in completed),
        prefixes=prefixes,actual_steps=sum(a['actual_steps'] for p in completed for a in p['arms']),
        source_snapshot_sha256=sha((root/'source_snapshot.json').read_bytes()),plan_sha256=sha((root/'plan.json').read_bytes()),adoption=False)
    (root/'summary.json').write_bytes(encoded(result))
    (root/'status.json').write_bytes(encoded(dict(complete=True,pairs_complete=12,actual_steps=result['actual_steps'],adoption=False)))


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        if '--root' in sys.argv:
            root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
            if root.exists():
                status=root/'status.json'
                previous=json.loads(status.read_bytes()) if status.exists() else {}
                if not previous.get('error'):
                    previous.update(complete=False,state='failed',error=repr(error))
                    status.write_bytes(encoded(previous))
        raise
