"""Check whole replay controls and record final numeric/source provenance."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
def rows(folder):
    result={}
    with (folder/'trajectories.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            key=digest([r.get('case'),r.get('profile'),r.get('domain'),r.get('layout'),r['seed'],r.get('variant'),r.get('peer_arm',r.get('arm'))])
            if key in result:raise ValueError('duplicate actual control')
            result[key]=r
    return result
def social_record(r):return {k:v for k,v in r.items() if k not in ('index','elapsed_seconds')}
def actual(r):return dict(goal=r['goal'],actions=r['actions'],trace=[{k:t[k] for k in ('before','after','root','flow','goal','decision')} for t in r['trace']])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verification-root',required=True);a=ap.parse_args();v=Path(a.verification_root).resolve();controls=[]
    x=rows(v/'choice_scale_social_20261012');y=rows(v/'choice_scale_social_cached_v2_20261012')
    shared=set(x)&set(y);assert len(shared)==12
    for k in shared:assert social_record(x[k])==social_record(y[k])
    controls.append(dict(first='uncached interrupted v1',second='cached interrupted v2',worlds=len(shared),whole_records_equal=True,scope='old estimator only; not fair public-only strength evidence'))
    z=rows(v/'choice_scale_social_public_v3_20261012');neutral={k for k in set(y)&set(z) if y[k]['peer_arm'].startswith('neutral')}
    for k in neutral:assert social_record(y[k])==social_record(z[k])
    controls.append(dict(first='cached v2',second='public-only v3',worlds=len(neutral),whole_records_equal=True,scope='neutral model controls; no predictive adoption'))
    old=rows(v/'choice_scale_transfer_20261012');new=rows(v/'choice_scale_confidence_transfer_v2_20261012')
    assert set(new)<=set(old)
    mismatches=[dict(case=new[k]['case'],profile=new[k]['profile'],seed=new[k]['seed'],arm=new[k]['arm']) for k in new if actual(old[k])!=actual(new[k])]
    controls.append(dict(first='gain subjective basis v1',second='confidence and unit coherent v2',episodes=len(new),whole_actual_records_equal=not mismatches,mismatches=mismatches))
    parity=json.loads((v/'choice_scale_default_parity_v2_20261012/result.json').read_text(encoding='utf-8'));assert parity['records']==1200 and not parity['mismatches']
    result=dict(controls=controls,source_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for d in ('reflex','tests') for p in sorted((ROOT/d).glob('*.py'))},
        default_records_equal=1200,peer_private_fields_before=json.loads((v/'peer-private-fields-before.json').read_text()),peer_private_fields_after=json.loads((v/'peer-private-fields-after.json').read_text()))
    (v/'choice-scale-controls.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(controls=controls,default_records_equal=1200),ensure_ascii=False))
if __name__=='__main__':main()
