"""Read-only phase accounting; frozen-current-belief is a reference, not an NPC."""
import argparse,gzip,hashlib,json,sys,tempfile,zipfile
from pathlib import Path
from datetime import datetime,timezone

folder=Path(__file__).resolve().parent
base=folder.parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
output=Path(parser.parse_args().output).resolve()
if output.exists():raise FileExistsError('Use a new output file')
reviews=[]
input_hashes={}
for data_folder in (folder.with_name('selected_confirmation_review_18'),folder):
    item_proof=json.loads((data_folder/'verification.json').read_bytes())
    for name,h in item_proof['artifacts'].items():
        assert hashlib.sha256((data_folder/name).read_bytes()).hexdigest()==h
    raw=(data_folder/'review.json.gz').read_bytes()
    reviews.append((data_folder,json.loads(gzip.decompress(raw))))
    input_hashes[data_folder.name]=hashlib.sha256(raw).hexdigest()
proof=json.loads((folder/'verification.json').read_bytes())
assert len({r['source'] for _,r in reviews})==1
source_proof=json.loads((folder/'replay_verification.json').read_bytes())
source_zip=base/source_proof['source_zip']
assert hashlib.sha256(source_zip.read_bytes()).hexdigest()==source_proof['source_zip_sha256']
with tempfile.TemporaryDirectory(prefix='combat-reference-audit-') as temp:
    source=Path(temp).resolve()
    with zipfile.ZipFile(source_zip) as bundle:
        for name in bundle.namelist():assert (source/name).resolve().is_relative_to(source)
        bundle.extractall(source)
    with zipfile.ZipFile(folder/'inputs.zip') as bundle:
        manifest=json.loads(bundle.read('source_snapshot.json'))
    encoded=lambda d:json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    assert hashlib.sha256(encoded(manifest)).hexdigest()==proof['source']
    for name,h in manifest.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h
    sys.path.insert(0,str(source))
    from tools.combat_reference import reference
    from tools.run_continuation_portfolio import game_for

    out=[]
    for data_folder,review in reviews:
        with zipfile.ZipFile(data_folder/'inputs.zip') as archive:
         for pair in review['pairs']:
            spec=pair['spec']
            if spec['game']!='combat':continue
            a=pair['arms']['baseline'];b=pair['arms']['candidate'];game=game_for(spec)
            original={arm:json.loads(archive.read(spec['id']+'/'+arm+'/result.json')) for arm in ('baseline','candidate')}
            asteps=a['account']['bellman']['increments'];bsteps=b['account']['bellman']['increments']
            assert len(asteps)==len(bsteps)==128
            changed=next((i for i,(x,y) in enumerate(zip(a['steps'],b['steps'])) if x['action']!=y['action']),None)
            if changed is not None:
                x=original['baseline']['trace'][changed];y=original['candidate']['trace'][changed]
                assert x['before']==y['before'] and x['before_owner']==y['before_owner']
            phases=[]
            for start in (0,32,64,96):
                end=start+32
                choice=-sum(y['purpose_opportunity_loss']-x['purpose_opportunity_loss'] for x,y in zip(asteps[start:end],bsteps[start:end]))
                noise=sum(y['residual']-x['residual'] for x,y in zip(asteps[start:end],bsteps[start:end]))
                beliefs=a['steps'][start:end]
                tv=sum(.5*sum(abs(s['belief'][k]-v) for k,v in s['truth'].items()) for s in beliefs)/len(beliefs)
                phases.append(dict(start=start,end_exclusive=end,reference_choice_delta=choice,reference_residual_delta=noise,
                    mean_current_prediction_total_variation=tv))
            assert abs(sum(p['reference_choice_delta'] for p in phases)-pair['difference']['reference_choice'])<1e-12
            root_reviews=[]
            if pair['difference']['reference_choice'] < -1e-12:
                extra=[(y['purpose_opportunity_loss']-x['purpose_opportunity_loss'],i) for i,(x,y) in enumerate(zip(asteps,bsteps))]
                chosen=sorted(extra,key=lambda x:(-x[0],x[1]))[:2]
                for loss,tick in chosen:
                    assert loss>0
                    step=b['steps'][tick];world=original['candidate']['trace'][tick]['before']
                    probabilities=tuple(step['belief'][str(z)] for z in range(3))
                    value,options=reference(spec['turns'],lambda t:probabilities)
                    q=options(world)
                    equations=0
                    for action in game.keys(world):
                        expected=sum(probabilities[z]*value(game.transition(world,action,str(z))[0]) for z in range(3))
                        assert abs(q[action]-expected)<1e-12
                        equations+=1
                    best=max(q.values());keys=sorted(k for k,v in q.items() if abs(v-best)<1e-12)
                    root_reviews.append(dict(tick=tick,extra_reference_loss_on_two_actual_paths=loss,
                        world=world,action=step['action'],current_belief=step['belief'],truth=step['truth'],
                        fixed_belief_options=q,best_keys=keys,chosen_gap=best-q[step['action']],equations=equations,
                        baseline_action=a['steps'][tick]['action'],baseline_world=original['baseline']['trace'][tick]['before']))
            out.append(dict(id=spec['id'],spec=spec,first_divergence=changed,difference=pair['difference'],phases=phases,
                root_reviews=root_reviews,all_beliefs_equal=all(x['belief_equal'] for x in pair['comparisons'])))
    assert len(out)==len({r['id'] for r in out})==8
    result=dict(complete=True,scope='Read-only diagnosis of all eight audited combat pairs; not full40 acceptance',
        input_sha256=input_hashes,source=proof['source'],
        controller_changed=False,adoption=False,rows=out,
        limits='Phase accounting retains personality and acquired resources. Top two roots selected per negative reference-choice pair, known adverse selection. Current-belief reference assumes a stationary law and omits persona/future learning; not factual optimal owner behavior or sampling causality.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,sort_keys=True,separators=(',',':')),encoding='utf8')
    print(json.dumps(dict(complete=True,audited_pairs=len(out),selected_roots=sum(len(r['root_reviews']) for r in out),
        exact_bellman_equations=sum(x['equations'] for r in out for x in r['root_reviews']),adoption=False)))
