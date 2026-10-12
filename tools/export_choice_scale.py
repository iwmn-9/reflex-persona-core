"""Publish only completed comparisons and source/control receipts."""
import argparse,ast,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verification-root',required=True);ap.add_argument('--analysis',required=True);a=ap.parse_args()
    v=Path(a.verification_root).resolve();analysis=Path(a.analysis).resolve();out=ROOT/'evidence/choice_scale';out.mkdir(parents=True,exist_ok=True)
    s=json.loads((analysis/'summary.json').read_text(encoding='utf-8'))
    if s['partial'] or s['errors'] or any(not c['complete'] for c in s['cohorts']) or not s['social']['complete']:raise ValueError('complete error-free comparisons required')
    assert sum(c['completed'] for c in s['cohorts'])==640 and s['social']['worlds']==64 and s['social']['npc_episodes']==384
    for name in ('summary.json','provenance.json'):shutil.copyfile(analysis/name,out/name)
    for name,file in [('controls.json',v/'choice-scale-controls.json'),('default_parity.json',v/'choice_scale_default_parity_v2_20261012/result.json')]:shutil.copyfile(file,out/name)
    privacy={label:json.loads((v/f'peer-private-fields-{label}.json').read_text(encoding='utf-8')) for label in ('before','after')}
    (out/'peer_private_fields.json').write_text(json.dumps(privacy,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    bridges=[]
    for c in (*s['cohorts'],s['social']):
        changes=[];source=v/c['name']/'source/reflex'
        for p in sorted(source.glob('*.py')):
            current=ROOT/'reflex'/p.name
            if p.read_bytes()!=current.read_bytes():changes.append(dict(file=p.name,frozen_sha256=sha(p),current_sha256=sha(current),
                ast_equal=ast.dump(ast.parse(p.read_text(encoding='utf-8-sig')))==ast.dump(ast.parse(current.read_text(encoding='utf-8-sig')))))
        bridges.append(dict(cohort=c['name'],changed_reflex_files=changes))
    interrupted=[]
    for folder in ('choice_scale_social_20261012','choice_scale_social_cached_v2_20261012'):
        record=json.loads((v/folder/'interrupted.json').read_text(encoding='utf-8-sig'))
        interrupted.append(dict(cohort=folder,**{k:x for k,x in record.items() if k not in ('parent','wrapper','children')}))
    implementation=dict(current_reflex_sha256={p.name:sha(p) for p in sorted((ROOT/'reflex').glob('*.py'))},source_bridge=bridges,
        tests=953,tests_log_sha256=sha(v/'choice-scale-tests-final-v2.log'),ordinary_default_unchanged=True,
        old_peer_checkpoint_rejected=True,cloud_runtime_used=False,license_added=False,interrupted_social_controls=interrupted,
        exclusions=['initial development output-path failure ran zero games','interrupted old-peer-model worlds are replay controls, not public-only strength trials'],
        limitations=['early matrices use subjective gain v1; final confidence and unit correction is explicit in source bridge',
            'all old-model interrupted world rows retained locally; final strength table uses public-only v3 cohort',
            'scenario memoization is only a paired-experiment work reuse, not general NPC performance evidence'])
    (out/'implementation.json').write_text(json.dumps(implementation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(game_executions=640,social_worlds=64,npc_episodes=384,public_files=len(list(out.glob('*.json'))))))
if __name__=='__main__':main()
