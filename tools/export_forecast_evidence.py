"""Export completed comparisons and an auditable source bridge; no raw archives."""
import argparse,ast,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verification-root',required=True);a=ap.parse_args();v=Path(a.verification_root).resolve()
    out=ROOT/'evidence/forecast_reliability';out.mkdir(parents=True,exist_ok=True)
    analysis=v/'forecast_reliability_final_analysis';summary=json.loads((analysis/'summary.json').read_text(encoding='utf-8'))
    if summary['partial'] or summary['errors'] or any(not c['complete'] for c in summary['cohorts']):raise ValueError('completed error-free cohorts required')
    for name in ('summary.json','provenance.json'):shutil.copyfile(analysis/name,out/name)
    extra={'default_parity.json':v/'forecast_default_parity_20261012/result.json',
        'continuation_parity.json':v/'forecast-continuation-parity.json',
        'checks_diagnostic.json':v/'forecast-check-diagnostic.json',
        'peer_values.json':v/'public-peer-values.json',
        'final_controls.json':v/'forecast-final-controls.json'}
    for name,source in extra.items():shutil.copyfile(source,out/name)
    terminal_counts={};adaptive_changed=[];paired={}
    with (v/'forecast_reliability_settlement_20261012/trajectories.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line);paired.setdefault(digest([r['case'],r['profile'],r['seed']]),{})[r['arm']]=r
            if r['arm']=='short':continue
            counts=terminal_counts.setdefault(r['case']['game'],dict(terminal=0,running=0))
            for t in r['trace']:
                for sample in t['modeled']:counts['terminal' if sample['terminal'] else 'running']+=1
    for arms in paired.values():
        x,y=arms['checked'],arms['adaptive']
        if x['goal']!=y['goal']:
            adaptive_changed.append(dict(game=x['case']['game'],profile=x['profile'],seed=x['seed'],
                checked=x['goal'],adaptive=y['goal']))
    labels=[c['name'] for c in summary['cohorts']]+['public_peer_reliability_20261012']
    bridge=[]
    for label in labels:
        source=v/label/'source/reflex';changes=[]
        for p in sorted(source.glob('*.py')):
            current=ROOT/'reflex'/p.name
            if current.read_bytes()!=p.read_bytes():
                changes.append(dict(file=p.name,old_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    current_sha256=hashlib.sha256(current.read_bytes()).hexdigest(),
                    ast_equal=ast.dump(ast.parse(p.read_text(encoding='utf-8-sig')))==ast.dump(ast.parse(current.read_text(encoding='utf-8-sig')))))
        bridge.append(dict(cohort=label,changed_reflex_files=changes))
    manifest=dict(current_reflex_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'reflex').glob('*.py'))},
        source_bridge=bridge,tests_log_sha256=hashlib.sha256((v/'forecast-reliability-tests-final.log').read_bytes()).hexdigest(),
        tests=937,cloud_runtime_used=False,license_added=False,
        settlement_selected_model_counts=terminal_counts,adaptive_changed_goals=adaptive_changed,
        limitations=['frozen cohorts remain authoritative; later optional adaptive/challenger modes and strict pure-policy validation are explicit source differences',
            'initial continuation shortcut parity used pre-guard source; strict guard unit tests and repeated actual controls checked separately',
            'later peer checkpoint validation/restore changes metadata, not predictive formulas'],
        excluded_pilot=json.loads((v/'forecast_reliability_pilot_20261012/interrupted.json').read_text(encoding='utf-8-sig')))
    # Process ids and host directory identities are local audit data, not needed
    # to reproduce the experiment's scientific conditions.
    manifest['excluded_pilot']={k:val for k,val in manifest['excluded_pilot'].items() if k not in ('parent','children')}
    (out/'implementation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cohorts=labels,files=len(list(out.glob('*.json'))),episodes=sum(c['completed'] for c in summary['cohorts']),social_worlds=summary['social'][0]['worlds'])))


if __name__=='__main__':main()
