"""Replay actual feedback, using the frozen actor, before publishing a screen."""
import argparse
import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--publish');args=parser.parse_args();root=Path(args.root).resolve()
    manifest=json.loads((root/'source_snapshot.json').read_bytes())
    for name,expected in manifest.items():
        assert hashlib.sha256((root/'_source'/name).read_bytes()).hexdigest()==expected,name
    sys.path.insert(0,str(root/'_source'))
    from reflex.core import digest,character_key
    from reflex.decision_loop import DecisionLoop
    from reflex.model_observer import resolve_feedback
    from tools.owned_checkpoint import encoded,sha
    from tools.run_continuation_portfolio import game_for
    from tools.run_owned_value import law,sample
    import numpy as np
    plan=json.loads((root/'plan.json').read_bytes())
    assert plan['source']==sha(encoded(manifest))
    summary=json.loads((root/'summary.json').read_bytes())
    assert summary['complete'] and summary['completed_pairs']==summary['registered_pairs']==len(plan['specs'])
    runs=[];decisions=0;prefixes=0
    for spec in plan['specs']:
        for retry in (False,True):
            if not retry and 'reference' in spec:
                data=(root/spec['reference']['archive']).read_bytes()
                assert sha(data)==spec['reference']['sha256']
                runs.append(dict(reference=spec['reference'],result=json.loads(data)))
                continue
            directory=root/spec['id']/('candidate' if retry else 'baseline')
            row=json.loads((directory/'result.json').read_bytes())
            envelope=json.loads((directory/'checkpoint.json').read_bytes());saved=envelope['payload']
            assert sha(encoded(saved))==envelope['sha256']
            assert row['complete'] and row['trace']==saved['trace'] and row['final']==saved['world']
            assert row['configuration']==saved['contract']['configuration']
            assert row['configuration']['spec']==spec and row['configuration']['retry']==retry
            for name,expected in row['configuration']['sources'].items():assert manifest['tools/'+name]==expected
            g=game_for(spec);w=g.start();actor=g.actor(g.context(w),learning='segmented')
            axes=character_key(g.context(w));rng=np.random.default_rng(spec.get('seed',0))
            for entry in row['trace']:
                assert entry['before']==w and entry['before_owner']==digest(actor.record())
                context=g.context(w,actor.state);request=g.request(context)
                result=DecisionLoop.decide_batch([(actor,request)],False,_counterfactual_roots=(entry['action'],))[0]
                assert character_key(result['context'])==axes
                truth=law(w['tick'],spec['turns'],spec['regime']) if spec['mechanism']=='world' else {'0':1.}
                response=sample(truth,rng.random()) if 'response' in entry else '0'
                after,flow=g.transition(w,entry['action'],response)
                for tick,payout in spec.get('changes',[]):
                    if after['tick']==tick:after['payout']=payout
                resolve_feedback(actor,result['ticket'],g.feedback(w,after,entry['action'],context,flow))
                assert after==entry['after'] and digest(actor.record())==entry['owner']
                if 'response' in entry:assert entry['response']==response and entry['truth']==truth
                w=after;decisions+=1
            assert w==row['final'] and g.terminal(w) and actor.record()==saved['owner']
            assert rng.bit_generator.state==saved['rng']
            assert row['score']==g.goal(w).value
            runs.append(dict(spec=spec,retry=retry,result=row))
    # Stable and later-deteriorating worlds must be indistinguishable until the
    # announced change is actually observed. No predicted future event leaks.
    indexed={(r['spec']['id'],r['retry']):r['result'] for r in runs if 'spec' in r}
    for spec in plan['specs']:
        if not spec['id'].endswith('-loss'):continue
        stable_id=spec['id'].removesuffix('-loss')+'-stable'
        for retry in (False,True):
            left=indexed[(spec['id'],retry)]['trace'];right=indexed[(stable_id,retry)]['trace']
            for a,b in zip(left[:spec['changes'][0][0]],right):
                for key in ('before','action','before_owner','selected_purpose','predicted_actions'):assert a[key]==b[key]
            prefixes+=1
    proof=dict(complete=True,pairs=len(plan['specs']),runs=len(runs),replayed_actual_decisions=decisions,
        matched_pre_change_prefixes=prefixes,source_verified=True,actual_owner_replayed=True,
        reference_reuse='Prior complete, audited controls retain their original source, SHA and traces; not new independent evidence.',
        limit='Full behavioral screen, not human-level intelligence, production adoption or completion of registered 40/8/8.')
    (root/'verification.json').write_bytes(encoded(proof))
    if args.publish:
        dest=Path(args.publish);dest.mkdir(parents=True,exist_ok=True)
        for name in ('plan.json','summary.json','source_snapshot.json'):(dest/name).write_bytes((root/name).read_bytes())
        (dest/'runs.jsonl.gz').write_bytes(gzip.compress(b'\n'.join(encoded(r) for r in runs)+b'\n',mtime=0))
        with zipfile.ZipFile(dest/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(manifest):archive.write(root/'_source'/name,name)
        proof['artifacts']={p.name:sha(p.read_bytes()) for p in dest.iterdir() if p.is_file() and p.name!='verification.json'}
        (dest/'verification.json').write_bytes(encoded(proof))
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
