"""Fixed comparison of continuous versus three-stance relationship projection.

Uses the existing two engineered rule sets. More seeds are new random streams,
not unseen games or human validation. No coefficients are tuned after results.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import digest
from .social import SocialPopulation
from .social_projection import CoarseSocialPopulation
from .social_experiment import PLAN as ORIGINAL, run


PLAN=dict(version='social-omission-v1',domains=ORIGINAL['domains'],schedules=ORIGINAL['schedules'],
    seeds=list(range(16)),previously_used_seeds=list(range(8)),new_stream_seeds=list(range(8,16)),
    settings=['prediction_only','continuous','coarse'],turns=36,
    projection=dict(neutral_below=.15,nonneutral_magnitude=.5,levels=3),
    adoption_gate=dict(no_new_deaths=True,no_retaliation=True,fixed_personality=True,
        care_exploit_aid_below_prediction=True,care_repair_aid_above_exploitation=True,
        max_group_mean_purpose_drop=.4),
    caveat='engineered game-behavior comparison, not measured enjoyment or psychological calibration')


def replay_check(trace,domain):
    count=0;terminal=0
    for t in trace:
        b=t['before'];a=t['after'];key=t['action']
        if key=='finished':
            assert b['health']<=0 and a==b;terminal+=1;continue
        assert b['health']>0
        e=b['energy'];r=b['reserve'];h=b['health']
        if key=='recover':e=min(1,e+.35);gain=.08
        else:
            assert e>=.18;e=max(0,e-.18)
            if key=='aid':
                gain=(.42 if domain=='allocation' else .55) if t['cooperate'] and not t['luck_failure'] else (-.12 if domain=='allocation' else -.4)
                if domain=='allocation':r+=gain
                else:h=min(8,h+(.25 if gain>0 else -.75))
            elif key=='independent':gain=.45 if domain=='allocation' else .4;r+=.3
            elif key=='retaliate':gain=-.8;r-=.4;h-=.5
            else:raise AssertionError(key)
        e=max(0,e-.025);r=max(0,r-.1);h=max(0,h-(.15 if r<=0 else 0))
        assert a==dict(energy=e,reserve=r,health=h,score=b['score']+gain)
        assert t['actual_gain']==gain and t['values_unchanged'];count+=1
    return count,terminal


def execute(seed,domain,schedule,setting):
    return run(seed,domain,schedule,setting!='prediction_only',
        population_factory=CoarseSocialPopulation if setting=='coarse' else SocialPopulation)


def summarize(runs):
    groups=[]
    for domain in PLAN['domains']:
        for schedule in PLAN['schedules']:
            for npc in [r['npc'] for r in runs[0]['actors']]:
                settings={}
                for setting in PLAN['settings']:
                    rows=[a for r in runs if r['domain']==domain and r['schedule']==schedule and r['setting']==setting
                          for a in r['actors'] if a['npc']==npc]
                    settings[setting]=dict(mean_score=float(np.mean([a['score'] for a in rows])),
                        alive=sum(a['alive'] for a in rows),aid_to_exploiter=sum(a['aid_to_exploiter'] for a in rows),
                        aid_by_phase=[sum(a['aid_by_phase'][phase] for a in rows) for phase in range(3)])
                groups.append(dict(domain=domain,schedule=schedule,npc=npc,settings=settings))
    deaths=[];large_drops=[];care_failures=[]
    for g in groups:
        s=g['settings'];c=s['continuous'];q=s['coarse'];p=s['prediction_only']
        if q['alive']<c['alive']:deaths.append({k:g[k] for k in ('domain','schedule','npc')})
        if q['mean_score']<c['mean_score']-.4:
            large_drops.append(dict(domain=g['domain'],schedule=g['schedule'],npc=g['npc'],delta=q['mean_score']-c['mean_score']))
        if g['npc']=='care':
            if g['schedule']=='exploitative' and q['aid_to_exploiter']>=p['aid_to_exploiter']:
                care_failures.append(dict(domain=g['domain'],reason='no exploitation response'))
            if g['schedule']=='change_and_repair' and q['aid_by_phase'][2]<=q['aid_by_phase'][1]:
                care_failures.append(dict(domain=g['domain'],reason='no relationship repair'))
    return dict(groups=groups,gate=dict(passed=not(deaths or large_drops or care_failures),
        death_regressions=deaths,large_purpose_drops=large_drops,care_failures=care_failures))


def experiment(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    registration=dict(PLAN,source_hashes={name:digest(Path(__file__).with_name(name).read_text(encoding='utf-8'))
        for name in ('social_omission_experiment.py','social_projection.py','social_experiment.py','social.py','core.py','runtime.py')})
    frozen=root/'preregister.json'
    if frozen.exists():raise FileExistsError('fresh output required; do not overwrite registered comparisons')
    frozen.write_text(json.dumps(registration,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    runs=[];counts=dict(actual_transitions=0,terminal_slots=0,personality_changes=0,retaliations=0,changed_choices=0)
    signals={s:set() for s in ('continuous','coarse')}
    path=root/'trajectories.jsonl'
    with path.open('w',encoding='utf-8') as f:
        for domain in PLAN['domains']:
            for schedule in PLAN['schedules']:
                for seed in PLAN['seeds']:
                    reference=None
                    for setting in PLAN['settings']:
                        r,trace=execute(seed,domain,schedule,setting);r['setting']=setting;runs.append(r)
                        actual,terminal=replay_check(trace,domain);counts['actual_transitions']+=actual;counts['terminal_slots']+=terminal
                        counts['personality_changes']+=sum(not t['values_unchanged'] for t in trace)
                        counts['retaliations']+=sum(t['action']=='retaliate' for t in trace)
                        if setting=='continuous':reference=trace
                        if setting=='coarse':counts['changed_choices']+=sum(a['action']!=b['action'] for a,b in zip(trace,reference))
                        if setting!='prediction_only':
                            for t in trace:
                                for row in t['appraisal'].values():
                                    signal=row['attitude']*row['confidence'] if setting=='continuous' else row['stance']
                                    signals[setting].add(round(signal,12))
                        f.write(json.dumps(dict(run=r,trace=trace),ensure_ascii=False)+'\n')
    result=dict(plan=registration,runs=runs,summary=summarize(runs),checks=counts,
        relation_levels={s:len(v) for s,v in signals.items()},episodes=len(runs)*4,
        trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),human_validation=False,enjoyment_measured=False)
    result['summary']['gate']['passed'] &= counts['personality_changes']==0 and counts['retaliations']==0 and len(signals['coarse'])<=3
    (root/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def audit(output):
    root=Path(output);result=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    registration=json.loads((root/'preregister.json').read_text(encoding='utf-8'))
    assert result['plan']==registration
    for name,expected in registration['source_hashes'].items():
        assert digest(Path(__file__).with_name(name).read_text(encoding='utf-8'))==expected,name
    path=root/'trajectories.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==result['trajectory_sha256']
    runs=[];actual=terminal=0
    with path.open(encoding='utf-8') as f:
        for line in f:
            saved=json.loads(line);r=saved['run']
            generated,trace=execute(r['seed'],r['domain'],r['schedule'],r['setting']);generated['setting']=r['setting']
            assert generated==r and trace==saved['trace'];runs.append(r)
            a,t=replay_check(trace,r['domain']);actual+=a;terminal+=t
    assert runs==result['runs'] and summarize(runs)==result['summary']
    assert actual==result['checks']['actual_transitions'] and terminal==result['checks']['terminal_slots']
    proof=dict(source_identity=True,replay_identical=True,run_records=len(runs),npc_episodes=len(runs)*4,
        actual_transitions=actual,terminal_slots=terminal,trajectory_sha256=result['trajectory_sha256'],
        human_validation=False,enjoyment_measured=False)
    (root/'audit.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');return proof


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--audit',action='store_true');args=p.parse_args()
    r=audit(args.output) if args.audit else experiment(args.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ('plan','runs')},ensure_ascii=False,indent=2))
