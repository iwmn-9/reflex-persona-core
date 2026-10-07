"""Replay real rules and branch-local observer updates from local evidence.

This checks execution integrity, not game strength or policy optimality.
Usage: python tools/audit_observed_transfer.py LOCAL_OBSERVED_TRANSFER_FOLDER
"""
from dataclasses import asdict,replace
from pathlib import Path
import copy
import hashlib
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest,compile_batch,Policy
from reflex.progress import ProgressWatch,ProgressConfig
from reflex.pressure import NeedPressure,PressureConfig
from reflex.purpose_plan import PersonaProgressWatch
from reflex.model_observer import ModelObserver
from reflex.observed_transfer import make_probe,feedback,aggregate,can_finish
from reflex.purpose_experiment import source_hashes


def world(genre,record):
    if genre=='resources':
        from reflex.resource_world import world_from_record
        return world_from_record(record)
    if genre=='auction':
        from reflex.contests import public_from_record
        return public_from_record(record)
    if genre=='combat':
        from reflex.combat import battle_from_record
        return battle_from_record(record)
    from reflex.delivery_world import Depot
    return Depot(**record)


def real_step(w,key,genre,spec,seed):
    if genre=='resources':
        from reflex.resource_world import step,terminal
        after=step(w,key)
        while not terminal(after) and after.turn!=0:after=step(after,'wait')
        return after
    if genre=='auction':
        from reflex.contests import settle
        rival=5 if spec.get('rival_shift') and w.round>=1 else max((3,)+tuple(w.last[1:]))
        bids=(int(key.split(':')[1]),)+tuple(min(rival,b,8) for b in w.budgets[1:])
        after,_=settle(w,bids)
        return replace(after,prize=w.remaining[0],remaining=w.remaining[1:]) if w.remaining else after
    if genre=='combat':
        from reflex.combat import resolve,legal
        keys=legal(w,3);shots=[k for k in keys if k.startswith('shoot:')]
        enemy='guard' if w.tick==0 else shots[0] if shots else 'reload' if 'reload' in keys else 'guard'
        return resolve(w,{0:key,3:enemy},seed)[0]
    from reflex.delivery_world import step
    return step(w,key)


def audit(folder):
    folder=Path(folder);evaluation=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    pre=json.loads((folder/'preregister.json').read_text(encoding='utf-8'))
    assert source_hashes()==pre['source_hashes']==evaluation['source_hashes']
    assert digest(pre)==evaluation['preregister_digest']
    rows=[json.loads(s) for s in (folder/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==len(pre['jobs'])==evaluation['episodes']
    actual=models=updates=prior_matches=0;policy=Policy()
    previous_public=json.loads((Path(__file__).resolve().parents[1]/'evidence/purpose_recovery/evaluation.json').read_text(encoding='utf-8'))
    for r,job,short in zip(rows,pre['jobs'],evaluation['runs']):
        split,spec,profile,variant,seed,horizon=job
        assert r['spec']==spec and r['profile']==profile['id'] and r['variant']==variant and r['seed']==seed and r['split']==split
        assert {k:v for k,v in r.items() if k!='trace'}==short
        p=make_probe(spec,profile);genre=spec['genre'];watched=variant in ('watch','release','observed','verified')
        pressured=variant in ('pressure','release','observed','verified');initial=p.observe(p.start());scope=initial['scope']
        watch=PersonaProgressWatch(scope,ProgressConfig(grace=2,repeat_limit=2,
            proof_margin=0. if variant=='verified' else None)) if watched else None
        pressure=NeedPressure(scope,PressureConfig(grace=2)) if pressured else None
        previous=None;memory=None
        for i,t in enumerate(r['trace']):
            w=world(genre,t['before']);after=world(genre,t['after'])
            if i:assert digest(r['trace'][i-1]['after'])==digest(t['before'])
            else:assert digest(asdict(p.start()))==digest(t['before'])
            assert not p.terminal(w) and t['root'] in p.keys(w)
            exact=real_step(w,t['root'],genre,spec,seed)
            assert digest(asdict(exact))==digest(t['after']);actual+=1
            assert p.goal(exact).record()==t['goal']
            _,flow=p.actual(w,t['root'],seed);assert flow==t['flow']
            c=p.observe(w,memory)
            assert digest([c['personality'],c['values']])==t['persona_hash']
            effective=pressure.apply(c) if pressure is not None else copy.deepcopy(c)
            req=p.purpose(w,c,previous) if watched else None
            assert (None if req is None else asdict(req))==t['purpose']
            if watched:
                proof=None
                if t['progress'].get('recovery') is not None:
                    proof={k:v for k,v in t['progress']['recovery'].items() if k!='approved'}
                allowed,mask=watch.mask(effective,req,proof)
                assert mask==t['progress'] and t['root'] in allowed
            else:assert t['progress'] is None
            if variant=='observed':
                base=ModelObserver(scope,purpose=p.purpose,
                    feedback=lambda a,b,k:feedback(p,a,b,k,watched,pressured),
                    progress=watch,pressure=pressure,previous=previous)
            for endpoint in t['endpoints']:
                mw=copy.deepcopy(w);mm=copy.deepcopy(effective['state'])
                observer=base.fork() if variant=='observed' else None
                records=endpoint.get('modeled_observations',[])
                assert bool(records)==(variant=='observed')
                for j,key in enumerate(endpoint['actions']):
                    cc=copy.deepcopy(effective) if j==0 else p.observe(mw,mm)
                    if observer is not None:
                        cc,mreq,permitted,maudit=observer.prepare(mw,cc,prepared=j==0)
                    b=compile_batch([cc]);legal=b.legal
                    if observer is not None and j:legal=legal&np.array([[k in permitted for k in b.ids[0]]])
                    d=policy.decide(replace(b,legal=legal),False)
                    assert key==t['root'] if j==0 else key==d.records(b)[0]['action_id']
                    mm=replace(d,action=np.array([b.ids[0].index(key)])).records(b)[0]['next_state']
                    mbefore=mw;mw,_=p.advance(mw,key,endpoint['seed']);models+=1
                    if observer is not None:
                        observer.advance(mbefore,mw,key,cc,mreq)
                        assert records[j]==dict(tick=cc['tick'],allowed=sorted(permitted),progress=maudit,
                            checkpoint=observer.record())
                assert p.goal(mw).record()==endpoint['assessment']
                assert p.terminal(mw)==endpoint['terminal']
                assert len(records)==len(endpoint['actions']) if observer is not None else not records
            fb=feedback(p,w,exact,t['root'],watched,pressured)
            assert digest(asdict(fb))==digest(t['feedback'])
            ib=compile_batch([effective]);idc=policy.decide(ib,False)
            expected_state=replace(idc,action=np.array([ib.ids[0].index(t['root'])])).records(ib)[0]['next_state']
            assert expected_state==t['state']
            if watched:watch.observe(c['tick'],t['root'],req,fb.purpose)
            if pressured:pressure.observe(c['tick'],fb.needs,fb.maintained,fb.completed)
            assert (None if watch is None else watch.record())==t['watch']
            assert (None if pressure is None else pressure.record())==t['pressure']
            if watch is not None:assert ProgressWatch.from_record(scope,watch.record()).record()==watch.record()
            if pressure is not None:assert NeedPressure.from_record(scope,pressure.record()).record()==pressure.record()
            updates+=int(watched or pressured);previous=w;memory=t['state']
        assert p.terminal(exact) and p.goal(exact).status==r['status'] and p.goal(exact).value==r['goal_value']
        assert [t['root'] for t in r['trace']]==r['actions']
        rev=0
        if genre=='combat':
            assert r['surviving']==(exact.units[0].hp>0)
            for i in range(len(r['trace'])-2):
                a=r['trace'][i]['after']['units'][0];b=r['trace'][i+2]['after']['units'][0]
                rev+=int((a['x'],a['y'])==(b['x'],b['y']) and r['actions'][i+1].startswith('move:') and r['actions'][i+2].startswith('move:'))
        assert rev==r['move_reversals']
        if genre=='resources':assert r['shortages']==exact.empires[0].shortages
        if genre=='delivery':
            assert (r['delivered'],r['community'])==(exact.delivered,exact.community)
            assert sum(can_finish(world(genre,t['before'])) and not can_finish(world(genre,t['after'])) for t in r['trace'])==r['solvable_route_lost']
        if split=='diagnostic' and variant in ('goal','release'):
            old=next(o for o in previous_public['runs'] if o['spec']==spec and o['profile']==profile['id'] and o['variant']==variant and o['seed']==seed and o['split']=='confirmation')
            for k in ('actions','status','goal_value','surviving','move_reversals'):assert r[k]==old[k]
            prior_matches+=1
    assert aggregate(evaluation['runs'])==evaluation['summary']
    return dict(episodes=len(rows),actual_rule_transitions=actual,selected_model_transitions=models,
        actual_observer_updates=updates,prior_diagnostic_episodes_identical=prior_matches,
        real_and_modeled_checkpoint_replay=True,source_freeze_verified=True,
        trajectories_sha256=hashlib.sha256((folder/'trajectories.jsonl').read_bytes()).hexdigest())


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('folder');args=parser.parse_args()
    print(json.dumps(audit(args.folder),indent=2))
