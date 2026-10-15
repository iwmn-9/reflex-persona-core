"""Same-source replay and known-probability immediate appraisal of adverse pairs.

The diagnostic is deliberately one-step: it does NOT call an investment or a
persona sacrifice a mistake merely because its immediate return is smaller.
"""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import compile_batch,digest
from reflex.laboratory import PROFILES
from tools.run_lifecycle_soak import Adapter,distribution,run_arm


def review(root,game,profile,schedule,choice_baseline=None):
    rows=[json.loads(line) for line in (root/'trajectories.jsonl').read_text().splitlines()]
    rows={r['arm']:r for r in rows if r['game']==game and r['profile']==profile and r['schedule']==schedule and r['visibility']=='censored'}
    variants={'coupled':(Adapter,'coupled'),'channels':(Adapter,'channels')}
    left_name,right_name='coupled','channels'
    if choice_baseline is not None:
        from tools.run_lifecycle_choice import PersonaJointAdapter,PersonaChoiceAdapter
        from tools.run_lifecycle_support import AvailabilityAdapter,LegacyAvailabilityAdapter
        variants={'persona_joint':(PersonaJointAdapter,'channels'),'persona_choice':(PersonaChoiceAdapter,'channels'),
                  'purpose_joint':(AvailabilityAdapter,'channels'),'legacy_support':(LegacyAvailabilityAdapter,'channels')}
        left_name,right_name=choice_baseline,'persona_choice'
    else:assert len(rows)==2
    pid=next(i for i,p in enumerate(PROFILES) if p['id']==profile)
    left,right=rows[left_name],rows[right_name]
    changed=[t for t,(a,b) in enumerate(zip(left['trace'],right['trace'])) if a['action']!=b['action']]
    reg=json.loads((root/'registration.json').read_text())
    job=next(j for j in reg['jobs'] if j[:4]==[game,pid,left['seed'],schedule] and j[5]=='censored')
    a=Adapter(game,PROFILES[pid],left['seed']%2);details=[]
    for arm in (left_name,right_name):
        def capture(s,c,actor,result,trace):
            req=a.request(c);cc=actor._prepare(req)['effective'];p=distribution(a.symbols(),s['tick'],job[4],schedule)
            probe=copy.deepcopy(actor);probe.decide(req,False);trial=probe.pending['decision_trial']
            for candidate in cc['actions']:
                candidate['outcomes']=[dict(a.transition(s,candidate['id'],z)[1],p=chance) for z,chance in p.items()]
            b=compile_batch([cc]);d=actor.policy.decide(b,False);index=b.ids[0].index(trace['action'])
            legal={k:float(v) for k,v,yes in zip(b.ids[0],d.scores[0],d.eligible[0]) if yes}
            details.append(dict(arm=arm,tick=s['tick'],planned=trace['plan'] is not None,
                public=s,chosen=trace['action'],response=trace['response'],actual_goal=trace['goal'],
                known_probability_immediate_choice=d.records(b)[0]['action_id'],
                chosen_score=float(d.scores[0,index]),eligible_scores=legal,
                selected_in_persona_tier=bool(d.eligible[0,index]),need_debt=trace['need_debt'],reading=result['reading'],
                comparison_choices=None if trial is None else {k:trial[k] for k in ('base_choice','candidate_choice')},
                response_trial=probe.pending['response_trial']))
        factory,delivery=variants[arm]
        replay=run_arm(job,delivery,capture=capture,capture_ticks=changed,adapter_factory=factory)
        assert replay['trace']==rows[arm]['trace'],'final source did not reproduce the frozen confirmation case'
    result=dict(game=game,profile=profile,schedule=schedule,left=left_name,right=right_name,score_delta=right['score']-left['score'],
        changed_ticks=changed,source_replay=True,details=details,
        interpretation='Known-probability immediate appraisal only; no full-horizon optimality or causal persona-loss claim.')
    prefix='loss' if choice_baseline is None else 'choice_loss_'+choice_baseline
    dest=root/f'{prefix}_{game}_{profile}_{schedule}.json';dest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(dest),changes=len(changed),score_delta=result['score_delta'],source_replay=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--game',required=True)
    p.add_argument('--profile',required=True);p.add_argument('--schedule',required=True)
    p.add_argument('--choice-baseline',choices=('persona_joint','purpose_joint','legacy_support'));args=p.parse_args()
    review(Path(args.root),args.game,args.profile,args.schedule,args.choice_baseline)
