"""Replace only prediction on the SAME pre-decision owner in prior loss cases.

The shadow reader learns exclusively from the baseline's revealed public past.
Diagnostic decisions never alter its actual path. This isolates recovery from
the later differences in board, needs and progress caused by changed actions.
"""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest,compile_batch
from reflex.laboratory import PROFILES
from tools.run_lifecycle_soak import run_arm,distribution
from tools.run_lifecycle_choice import PersonaChoiceAdapter
from tools.run_recovery_bridge import BridgeAdapter


def diagnose(root,game,profile,schedule,ticks):
    pid=next(i for i,p in enumerate(PROFILES) if p['id']==profile)
    rows=[json.loads(x) for x in (root/'trajectories.jsonl').read_text().splitlines()]
    baseline=next(r for r in rows if (r['game'],r['profile'],r['schedule'],r['arm'])==(game,profile,schedule,'choice'))
    job=(game,pid,29,schedule,128,'censored',12);a=BridgeAdapter(game,PROFILES[pid],1)
    reader=a.actor(a.raw(a.start())).predictor;details=[]
    def capture(s,c,actor,result,trace):
        nonlocal reader
        assert reader.observations==c['tick']
        if c['tick'] in ticks:
            before=digest(actor.record());original=copy.deepcopy(actor);replacement=copy.deepcopy(actor)
            replacement.predictor=copy.deepcopy(reader)
            left=original.record();right=replacement.record()
            left.pop('predictor');right.pop('predictor');assert left==right
            req=a.request(c);old=original.decide(req,False);new=replacement.decide(req,False)
            assert old['decision']['action_id']==trace['action'],'targeted tick must be an unplanned original decision'
            known=actor._prepare(req)['effective'];truth=distribution(a.symbols(),c['tick'],128,schedule)
            for row in known['actions']:
                row['outcomes']=[dict(a.outcome(c,row['id'],z),p=p) for z,p in truth.items()]
            batch=compile_batch([known]);oracle=actor.policy.decide(batch,False)
            scores={k:float(v) for k,v in zip(batch.ids[0],oracle.scores[0])}
            trial=original.pending['decision_trial'];prior,candidate=actor.predictor.forecasts(c)
            details.append(dict(tick=c['tick'],public=s,original=old['decision']['action_id'],
                replacement=new['decision']['action_id'],comparison_choices=None if trial is None else
                {k:trial[k] for k in ('base_choice','candidate_choice')},
                prior=prior,candidate=candidate,recovery=reader.predict(c),
                same_person_immediate_scores=scores,
                selected_score_gain=scores[new['decision']['action_id']]-scores[old['decision']['action_id']],
                only_predictor_replaced=True,prior_observations=reader.observations))
            assert before==digest(actor.record())
        reader,_=reader.updated_event(c,{'revealed_action':trace['response']},'real-'+str(c['tick']))
    replay=run_arm(job,'channels',adapter_factory=PersonaChoiceAdapter,capture=capture,capture_ticks=tuple(range(128)))
    assert replay['trace']==baseline['trace'] and replay['score']==baseline['score']
    result=dict(game=game,profile=profile,schedule=schedule,baseline_reproduced=True,details=details,
        limitation='Known-law immediate scoring diagnostic, not a terminal-value oracle or proof that a loss was caused by personality. Progress masks and future investment are not part of this scoring reference.')
    path=root/f'same_owner_{game}_{profile}_{schedule}.json';path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(game=game,profile=profile,schedule=schedule,changes=[(r['tick'],r['original'],r['replacement'],r['selected_score_gain']) for r in details])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--game',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--schedule',required=True);p.add_argument('--ticks',type=int,nargs='+',required=True);a=p.parse_args()
    diagnose(Path(a.root),a.game,a.profile,a.schedule,a.ticks)
