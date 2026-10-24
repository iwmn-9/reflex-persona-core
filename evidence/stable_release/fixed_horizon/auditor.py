from pathlib import Path
import json,hashlib,gzip,subprocess
from datetime import datetime,timezone
repo=Path.cwd();root=repo/'results/stable_release/fixed_horizon_v2'; plan=json.loads((root/'plan.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
known=[json.loads(line) for line in gzip.decompress(Path(plan['known_worlds']['path']).read_bytes()).splitlines()]
histories={x['spec']['id']:x for x in known if x.get('retry') is True}
records=[]
for arm in ('frozen','package'):
 assert json.loads((root/arm/'progress.json').read_bytes()).get('complete') is True, arm+' not complete'
 lib=plan['libraries'][arm];assert all(sha((Path(lib['path'])/n).read_bytes())==v for n,v in lib['files'].items())
 assert {p.stem for p in (root/arm).glob('*.json') if p.name!='progress.json'}=={c['id'] for c in plan['cases']}
for case in plan['cases']:
 pair=[json.loads((root/arm/(case['id']+'.json')).read_bytes()) for arm in ('frozen','package')]
 for row in pair:
  assert row['complete'] and row['case']==case and len(row['trace'])==case['steps']
  assert row['contract']['plan_sha256']==sha((root/'plan.json').read_bytes())
  assert row['contract']['worker_sha256']==plan['worker_sha256']
 assert {x['contract']['implementation'] for x in pair}=={'frozen','package'}
 a,b=[{k:v for k,v in x.items() if k!='contract'} for x in pair];assert a==b,case['id']
 if case['kind']=='known-root':
  old=histories[case['history']]['result']['trace'][case['tick']];actual=a['trace'][0]
  fields=dict(action='action',response='response',before_owner='before_owner',after_owner='owner',purpose='selected_purpose',simulated_steps='simulated_steps',cutoff='horizon_consensus')
  assert all(actual[k]==old[v] for k,v in fields.items()),case['id']+' archived long-run mismatch'
  assert actual['world_sha256']==sha(encode(old['after']))
 records.extend(pair)
assert sha((root/'worker.py').read_bytes())==plan['worker_sha256']
assert sha(Path(plan['known_worlds']['path']).read_bytes())==plan['known_worlds']['sha256']
assert plan['cases']==json.loads((root.parent/'fixed_horizon/plan.json').read_bytes())['cases']
published=repo/'evidence/stable_release/fixed_horizon';published.mkdir(exist_ok=True)
portable=json.loads(json.dumps(plan));portable['execution_plan_sha256']=sha((root/'plan.json').read_bytes())
for lib in portable['libraries'].values():lib['path']=Path(lib['path']).relative_to(repo).as_posix()
portable['known_worlds']['path']=Path(portable['known_worlds']['path']).relative_to(repo).as_posix()
portable['path_note']='Paths are repository-relative relocation placeholders. Recreate installed wheel and extract the already published frozen source.zip, then set paths for a fresh comparison; execution plan hashes include their original local paths.'
(published/'auditor.py').write_bytes(Path(__file__).read_bytes())
(published/'plan.json').write_bytes(encode(portable))
(published/'results.jsonl.gz').write_bytes(gzip.compress(b'\n'.join(encode(x) for x in records)+b'\n',mtime=0))
proof=dict(version='fixed-horizon-package-parity-v1',at=datetime.now(timezone.utc).isoformat(),complete=True,adoption=False,source_commit='445e3f4fb4748f54dd3b2030b4bfe67baffef2ec',cases=len(plan['cases']),matched_decisions=sum(c['steps'] for c in plan['cases']),matched_historical_replay=sum(c.get('tick',0) for c in plan['cases']),package_restores=sum(c['steps'] for c in plan['cases']),matched_archived_long_run_decisions=4,exact_fields=['full decision','full forecast','captured rollout effect paths and audits','actual choices and responses','actual owner before and after every decision','fixed personality','world','RNG'],execution_plan_sha256=portable['execution_plan_sha256'],worker_sha256=plan['worker_sha256'],script='tools/verify_fixed_horizon_package.py',frozen_source='evidence/long_horizon/continuation_horizon_ablation/source.zip',frozen_algorithm_source='1f01060ee8d647e829a479083efc7faf20db2b6c1184e1864c9f6036de5a892b',wheel_sha256=sha(next((root/'wheel').glob('*.whl')).read_bytes()),files={p.name:sha(p.read_bytes()) for p in sorted(published.iterdir()) if p.name!='verification.json'},prior_failure=plan['previous_attempt'],limits=['Integration equivalence, not additional independent quality evidence.','Six-turn full trajectories across all four families/personas; one complete H32 decision per family after exactly 64 factual decisions.','Original independent40 must still finish, be replayed and meet unchanged acceptance; no adoption from these tests.'])
(published/'verification.json').write_bytes(encode(proof));print(json.dumps({k:proof[k] for k in ('complete','cases','matched_decisions','matched_historical_replay','package_restores')}))
