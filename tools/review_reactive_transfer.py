"""Complete-only physical/RNG/source audit and persona-stratified ablation review."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reflex.core import character_key,digest
from tools.run_adverse_recovery import game_for
from tools.run_owned_value import law,sample
from tools.owned_checkpoint import encoded,sha


def review(plan_path,input_root,output):
    plan_path,input_root,output=map(Path,(plan_path,input_root,output))
    plan=json.loads(plan_path.read_bytes());summary=json.loads((input_root/'summary.json').read_bytes())
    assert summary['complete'] and summary['pairs']==len(plan['specs'])
    assert summary['plan_sha256']==sha(plan_path.read_bytes())
    for name,checksum in plan['sources'].items():assert sha((ROOT/name).read_bytes())==checksum
    if output.exists():raise FileExistsError('keep the original complete audit')
    rows=[];receipts=[];streams={};action_sequences={}
    for spec in plan['specs']:
        filename=spec['id']+'.json.gz';raw=(input_root/filename).read_bytes()
        assert sha(raw)==summary['artifacts'][filename]
        row=json.loads(gzip.decompress(raw));assert row['spec']==spec
        paired=[]
        for arm,execution in row['arms'].items():
            trace=execution['trace'];assert len(trace)==spec['turns']==execution['replayed_steps']
            game=game_for(spec);world=game.start();rng=np.random.default_rng(spec['seed'])
            axes=character_key(game.context(world));prior=None;increments=[]
            for t in trace:
                assert world==t['before'] and world['tick']==t['tick']
                if prior is not None:assert prior==t['before_owner']
                truth=law(world['tick'],spec['turns'],spec['regime']);assert truth==t['truth']
                response=sample(truth,rng.random());assert response==t['response']
                inc=sum(p*(game.goal(game.transition(world,t['action'],z)[0]).value-game.goal(world).value) for z,p in truth.items())
                assert inc==t['conditional_increment'];increments.append(inc)
                world,_=game.transition(world,t['action'],response)
                assert world==t['after'] and character_key(game.context(world))==axes
                prior=t['owner']
            assert game.terminal(world) and game.goal(world).value==execution['score']
            assert sum(increments)==execution['conditional'] and rng.bit_generator.state==execution['final_rng']
            assert digest(execution['final_owner'])==prior
            sequence=[(t['response'],t['response_prediction']) for t in trace]
            key=(spec['seed'],spec['regime'],arm)
            if key in streams:assert sequence==streams[key]
            else:streams[key]=sequence
            paired.append([t['response'] for t in trace])
            action_sequences[(spec['game'],spec['seed'],spec['regime'],arm,spec['profile'])]=[t['action'] for t in trace]
        assert paired[0]==paired[1]
        for metric in ('score','conditional'):
            assert row['delta'][metric]==row['arms']['transfer'][metric]-row['arms']['stable'][metric]
        rows.append(row);receipts.append(dict(id=spec['id'],sha256=sha(raw),steps=2*spec['turns'],source_bound=True,physical_and_rng_replayed=True,worker_exact_owner_replayed=True))
    def group(selected):
        return dict(pairs=len(selected),better=sum(r['delta']['score']>1e-12 for r in selected),
            same=sum(abs(r['delta']['score'])<=1e-12 for r in selected),worse=sum(r['delta']['score']< -1e-12 for r in selected),
            mean_delta={m:sum(r['delta'][m] for r in selected)/len(selected) for m in ('score','conditional')})
    families={g:group([r for r in rows if r['spec']['game']==g]) for g in sorted({r['spec']['game'] for r in rows})}
    persona={g:{str(p):group([r for r in rows if r['spec']['game']==g and r['spec']['profile']==p]) for p in range(4)} for g in families}
    diversity={arm:Counter() for arm in ('stable','transfer')}
    for game,seed,regime in sorted({k[:3] for k in action_sequences}):
        for arm in diversity:
            variants={tuple(action_sequences[(game,seed,regime,arm,p)]) for p in range(4)}
            diversity[arm][len(variants)]+=1
    report=dict(complete=True,pairs=len(rows),actual_steps=sum(r['steps'] for r in receipts),
        families=families,persona=persona,groups=summary['groups'],
        adverse=[dict(spec=r['spec'],delta=r['delta']) for r in rows if any(v< -1e-12 for v in r['delta'].values())],
        trajectory_diversity={arm:dict(counts) for arm,counts in diversity.items()},
        evidence_unit='256 paired configurations reuse 16 response streams across rule families/personas; they are not 256 independent worlds.',
        diversity_scope='Observed trajectory differences, not controlled same-state personality quality or a human-likeness score.',
        adoption=False,h32_confirmation='separate required experiment; no promotion from this immediate-policy ablation')
    output.mkdir(parents=True)
    (output/'review.json').write_bytes(encoded(report));(output/'plan.json').write_bytes(plan_path.read_bytes())
    (output/'runs.jsonl.gz').write_bytes(gzip.compress(b''.join(encoded(row)+b'\n' for row in rows),mtime=0))
    with zipfile.ZipFile(output/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in plan['sources']:z.write(ROOT/name,name)
        z.write(Path(__file__),'tools/review_reactive_transfer.py')
    (output/'verification.json').write_bytes(encoded(dict(complete=True,plan_sha256=sha(plan_path.read_bytes()),
        receipts=receipts,adoption=False,auditor_sha256=sha(Path(__file__).read_bytes()),
        artifacts={p.name:sha(p.read_bytes()) for p in output.iterdir()})))
    print(json.dumps({k:report[k] for k in ('complete','pairs','actual_steps','families','trajectory_diversity','adoption')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();review(a.plan,a.input,a.output)
