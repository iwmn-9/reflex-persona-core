"""Check registered implementation, deterministic replay and actual transitions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.social_experiment import run


def audit(root):
    root=Path(root);registration=json.loads((root/'preregister.json').read_text(encoding='utf-8'))
    implementation=Path(__file__).resolve().parents[1]/'reflex'
    for name,expected in registration['source_hashes'].items():
        assert digest((implementation/name).read_text(encoding='utf-8'))==expected,name
    data=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    path=root/'trajectories.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==data['trajectory_sha256']
    runs=[];transitions=0;finished=0
    with path.open(encoding='utf-8') as f:
        for line in f:
            saved=json.loads(line);r=saved['run']
            generated,trace=run(r['seed'],r['domain'],r['schedule'],r['appraisal'])
            assert generated==r and trace==saved['trace']
            runs.append(r)
            for t in trace:
                before=t['before'];after=t['after'];key=t['action'];domain=r['domain']
                if key=='finished':
                    assert before['health']<=0 and before==after
                    finished+=1;continue
                assert before['health']>0
                e=before['energy'];s=before['reserve'];h=before['health']
                if key=='recover':e=min(1,e+.35);gain=.08
                else:
                    assert e>=.18;e=max(0,e-.18)
                    if key=='aid':
                        gain=(.42 if domain=='allocation' else .55) if t['cooperate'] and not t['luck_failure'] else (-.12 if domain=='allocation' else -.4)
                        if domain=='allocation':s+=gain
                        else:h=min(8,h+(.25 if gain>0 else -.75))
                    elif key=='independent':gain=.45 if domain=='allocation' else .4;s+=.3
                    elif key=='retaliate':gain=-.8;s-=.4;h-=.5
                    else:raise AssertionError('unknown actual action')
                e=max(0,e-.025);s=max(0,s-.1);h=max(0,h-(.15 if s<=0 else 0))
                expected=dict(energy=e,reserve=s,health=h,score=before['score']+gain)
                assert expected==after and gain==t['actual_gain']
                assert t['values_unchanged'];transitions+=1
    assert runs==data['runs'] and transitions==data['checks']['decisions']
    result=dict(source_identity_verified=True,run_records=len(runs),npc_episodes=4*len(runs),
        replay_identical=True,independent_actual_transitions=transitions,terminal_observation_slots=finished,
        trajectory_sha256=data['trajectory_sha256'],human_validation=False)
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    print(json.dumps(audit(args.root),indent=2))
