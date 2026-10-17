import json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from tools.compare_owned_studies import compare
root=R/'results/owner_continuity_cloud';recovery=R/'results/owner_recovery_cloud'
studies=[p.parent for p in root.glob('*/frequent/audit.json')]+[p.parent for p in root.glob('*/periodic/audit.json')]
recovery_studies=[p.parent for p in recovery.glob('*/audit.json')]
completion=R/'results/owner_recovery_completion/auction-growth'
if (completion/'audit.json').exists():recovery_studies.append(completion)
studies += recovery_studies
result=compare(studies,[('endpoint@7',candidate) for candidate in ('endpoint@1','endpoint_posterior@1','receding_corrected@1','receding_recovery@1')]+
 [('endpoint_posterior@1','receding_corrected@1'),('receding_corrected@1','receding_recovery@1')]+
 [('endpoint_posterior@7',candidate) for candidate in ('receding_corrected@1','receding_recovery@1')])
result['complete_initial_matrix']=len(list(root.glob('*/periodic/audit.json')))==8 and len(list(root.glob('*/frequent/audit.json')))==8
from reflex.laboratory import PROFILES
expected={(game,PROFILES[pid]['id'],43,regime,arm,128,1,15,0) for game in ('auction','resources','combat','gambling')
    for pid in (0,1) for regime in ('outside','shift-return') for arm in ('receding_corrected','receding_recovery')}
actual=[]
for study in recovery_studies:
    for row in map(json.loads,(study/'trajectories.jsonl').read_text().splitlines()):
        actual.append(tuple(row[k] for k in ('game','profile','seed','regime','arm','turns','interval','tail_steps','offset')))
assert len(actual)==len(set(actual)) and set(actual)<=expected
result['complete_recovery_matrix']=set(actual)==expected
ablation=[]
for study in sorted(recovery_studies):
    pairs={}
    for row in map(json.loads,(study/'trajectories.jsonl').read_text().splitlines()):
        key=tuple(row[k] for k in ('game','profile','seed','regime'))
        pairs.setdefault(key,{})[row['arm']]=row
    for key,arms in pairs.items():
        left=arms['receding_corrected'];right=arms['receding_recovery'];steps=min(left['executed'],right['executed'])
        assert all(l['response']==r['response'] and l['law']==r['law'] for l,r in zip(left['trace'],right['trace']))
        ablation.append(dict(zip(('game','profile','seed','regime'),key),matched_public_predictions=steps,
            same_actions=sum(l['action']==r['action'] for l,r in zip(left['trace'],right['trace'])),
            same_resolved_owners=sum(l['owner']==r['owner'] for l,r in zip(left['trace'],right['trace'])),
            fixed_predictor_score_delta=right['score']-left['score']))
result['control_ablation_same_predictions']=ablation
(R/'results/owner_continuity_development.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(runs=result['runs'],complete_initial_matrix=result['complete_initial_matrix'],complete_recovery_matrix=result['complete_recovery_matrix'],
 summaries=[s for s in result['summaries'] if s['group_by']=='all']),indent=2))
