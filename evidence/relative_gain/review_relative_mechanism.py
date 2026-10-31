from collections import Counter
from pathlib import Path
import hashlib,json

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
local=Path(__file__).resolve().parent;base=local.parents[2]
root=base/'reflex-persona-relative-gain'
execution=root/'results/relative_gain_screen_20261101';archive=root/'evidence/relative_gain/long_complete'
proof=read(archive/'verification.json');plan=read(execution/'plan.json')
assert proof['complete'] and proof['gate_passed'] and not proof['losses']
rows=[]
for spec in plan['specs']:
    arms={r:read(execution/spec['id']/r/'result.json') for r in plan['references']}
    old,new=arms['origin'],arms['ordinary']
    row=dict(id=spec['id'],profile=spec['profile'],score_delta=new['score']-old['score'],
        changed_actions=sum(a['action']!=b['action'] for a,b in zip(old['trace'],new['trace'])),
        arms={name:dict(score=r['score'],final=r['final'],
            actions=dict(Counter(t['action'] for t in r['trace'])),
            policies=dict(Counter(t['selected_policy'] for t in r['trace']))) for name,r in arms.items()},
        phases=[dict(start=i,score_state={name:r['trace'][min(i+32,128)-1]['after'] for name,r in arms.items()}) for i in range(0,128,32)])
    rows.append(row)
target=archive/'behavior_review.json';assert not target.exists()
target.write_text(json.dumps(dict(complete=True,rows=rows,adoption=False),sort_keys=True),encoding='utf8')
print(json.dumps([{k:r[k] for k in ('id','profile','score_delta','changed_actions','arms')} for r in rows]))
