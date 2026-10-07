"""Independent exact-rule replay and observed progress/checkpoint audit.

Usage: python tools/audit_purpose_recovery.py LOCAL_PURPOSE_RECOVERY_FOLDER
Full real trajectories are local; only frozen conditions/aggregates are public.
"""
from dataclasses import replace,asdict
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.progress import Activity,PurposeRequest,PurposeFeedback
from reflex.purpose_plan import PersonaProgressWatch
from reflex.pressure import NeedPressure
from reflex.purpose_recovery import GoalProbe
from reflex.laboratory import profiles
from reflex.purpose_experiment import source_hashes


def goal(w,genre,target,spec,route):
    if genre=='resources':
        from reflex.resource_world import winners,terminal
        from reflex.route_experiment import route_potential
        ws=winners(w)
        status='success' if 0 in ws else 'failure' if ws else 'draw' if terminal(w) else 'running'
        value={'success':1.,'failure':-1.,'draw':0.}.get(status,route_potential(w,0,route))
    elif genre=='combat':
        from reflex.combat import victory,terminal
        status='success' if 0 in victory(w) else 'failure' if 1 in victory(w) else 'draw' if terminal(w) else 'running'
        value={'success':1.,'failure':-1.,'draw':0.}.get(status,.6*(1-w.units[3].hp/9))
    else:
        status='scored' if w.round>=len(spec['prizes']) else 'running';value=w.scores[0]/sum(spec['prizes'])
    return dict(target=target,status=status,value=value)


def audit(folder):
    folder=Path(folder);result=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    pre=json.loads((folder/'preregister.json').read_text(encoding='utf-8'))
    assert pre['source_hashes']==result['source_hashes']==source_hashes()
    assert result['preregister_digest']==digest(pre)
    rows=[json.loads(s) for s in (folder/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==result['episodes']==len(result['runs'])
    transitions=feedbacks=deadlines=release_requests=0
    for r,short in zip(rows,result['runs']):
        assert {k:v for k,v in r.items() if k!='trace'}==short
        spec=r['spec'];genre=spec['genre'];trace=r['trace'];watch=pressure=None
        profile=next(p for p in profiles() if p['id']==r['profile']);probe=GoalProbe(spec,profile,r['variant']!='old')
        assert len({t['persona_hash'] for t in trace})==1
        for i,t in enumerate(trace):
            if i:assert trace[i-1]['after']==t['before']
            if genre=='resources':
                from reflex.resource_world import world_from_record,world_record,step,terminal
                w=world_from_record(t['before']);assert not terminal(w)
                after=step(w,t['root'])
                while not terminal(after) and after.turn!=0:after=step(after,'wait')
                if spec.get('famine') and after.round==2:after=replace(after,food_yield=0)
                rec=world_record(after)
                old_value=lambda s:(sum(s.empires[0].stock)+s.empires[0].science+s.empires[0].culture)/100
            elif genre=='auction':
                from reflex.contests import public_from_record,settle
                w=public_from_record(t['before']);assert w.round<len(spec['prizes'])
                rival=5 if spec.get('rival_shift') and w.round>=1 else max((3,)+tuple(w.last[1:]))
                bids=(int(t['root'].split(':')[1]),)+tuple(min(rival,b,8) for b in w.budgets[1:])
                after,_=settle(w,bids)
                if w.remaining:after=replace(after,prize=w.remaining[0],remaining=w.remaining[1:])
                if spec.get('revised_prize') and after.round==1:after=replace(after,prize=3)
                rec=asdict(after);old_value=lambda s:s.scores[0]/30
            else:
                from reflex.combat import battle_from_record,battle_record,resolve,legal,terminal,potential
                w=battle_from_record(t['before']);assert not terminal(w) and w.units[0].hp>0
                keys=legal(w,3);shots=[k for k in keys if k.startswith('shoot:')]
                enemy='guard' if w.tick==0 else shots[0] if shots else 'reload' if 'reload' in keys else 'guard'
                after,_=resolve(w,{0:t['root'],3:enemy},r['seed']);rec=battle_record(after)
                assert r['seed'] not in pre['model_combat_seeds'];old_value=lambda s:potential(s,0,'eliminate')/2
            assert digest(rec)==digest(t['after']);transitions+=1
            expected=goal(after,genre,r['target'],spec,r['route'])
            assert expected==t['goal']
            previous=goal(w,genre,r['target'],spec,r['route'])
            g=old_value(after)-old_value(w) if r['variant']=='old' else (expected['value']-previous['value'])/2
            assert abs(g-t['flow']['objective'])<1e-12
            if r['variant']=='release':
                c=probe.observe(w,trace[i-1]['state'] if i else None);owner=c['scope']
                req=PurposeRequest(**{**t['purpose'],'activities':{k:Activity(**v) for k,v in t['purpose']['activities'].items()}})
                fb=PurposeFeedback(**t['feedback'])
                assert asdict(probe.purpose(w,c,None if i==0 else before_previous))==t['purpose']
                if watch is None:
                    cfg=t['watch']['config'];from reflex.progress import ProgressConfig
                    watch=PersonaProgressWatch(owner,ProgressConfig(**cfg))
                    from reflex.pressure import PressureConfig
                    pressure=NeedPressure(owner,PressureConfig(**t['pressure']['config']))
                effective=pressure.apply(c);allowed,mask=watch.mask(effective,req)
                assert mask==t['progress'] and t['root'] in allowed
                watch.observe(c['tick'],t['root'],req,fb);assert watch.record()==t['watch']
                gain=max(-1.,min(1.,expected['value']-previous['value']))
                pressure.observe(c['tick'],{'growth':gain},maintained=('growth',) if fb.maintained else (),
                    completed=('growth',) if expected['status']=='success' else ())
                assert pressure.record()==t['pressure'];feedbacks+=1
                release_requests+=sum(a.kind=='wait' and bool(a.release) for a in req.activities.values())
                restored=PersonaProgressWatch.from_record(owner,watch.record())
                assert restored.record()==watch.record()
            elif t['purpose'] is not None or t['watch'] is not None:raise AssertionError('unrequested progress changed baseline')
            before_previous=w
        assert expected['status']==r['status'] and abs(expected['value']-r['goal_value'])<1e-12
        assert expected['status']!='running'
        deadlines+=expected['status']=='draw'
    for s in result['summary']:
        rs=[r for r in rows if r['split']==s['split'] and r['spec']['genre']==s['genre'] and r['variant']==s['variant']]
        assert len(rs)==s['episodes']
        assert sum(r['status']=='success' for r in rs)==s['successes']
        assert sum(r['status']=='failure' for r in rs)==s['failures']
        assert abs(sum(r['goal_value'] for r in rs)-s['goal_sum'])<1e-12
        for key in ('move_reversals','progress_applied','progress_unresolved'):
            field='unresolved' if key=='progress_unresolved' else key
            assert sum(r[key] for r in rs)==s[field]
    return dict(episodes=len(rows),rule_transitions_replayed=transitions,real_progress_updates_replayed=feedbacks,
        actual_draw_outcomes=deadlines,finite_wait_release_declarations=release_requests,
        source_freeze_verified=True,settled_goals_verified=True,checkpoint_progress_verified=True,
        trajectories_sha256=hashlib.sha256((folder/'trajectories.jsonl').read_bytes()).hexdigest())


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('folder');args=parser.parse_args()
    print(json.dumps(audit(args.folder),ensure_ascii=False,indent=2))
