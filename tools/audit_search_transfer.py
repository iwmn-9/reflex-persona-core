"""Replay retained shared schedules, forecast selection and actual owner updates.

This audits finite execution and information contracts, not beam optimality.
Usage: python tools/audit_search_transfer.py LOCAL_SEARCH_TRANSFER_DIRECTORY
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
from pathlib import Path
import copy
import hashlib
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import Policy,compile_batch,digest
from reflex.examples import effect
from reflex.intertemporal import Branch
from reflex.flow_rollout import rollout
from reflex.purpose_plan import goal_forecast,PersonaProgressWatch
from reflex.progress import ProgressConfig
from reflex.pressure import NeedPressure,PressureConfig
from reflex.decision_loop import DecisionLoop,Request
from reflex.judgment import Binding
from reflex.observed_transfer import make_probe,feedback,aggregate,can_finish
from reflex.purpose_experiment import source_hashes
from reflex.plan_continuity import PlanIntention,ArrivalPreference
from audit_observed_transfer import real_step


def replay_episode(args):
    r,job,format=args;split,spec,profile,variant,seed,h=job
    assert (r['split'],r['spec'],r['profile'],r['variant'],r['seed'])==(split,spec,profile['id'],variant,seed)
    p=make_probe(spec,profile);w=p.start();c=p.observe(w)
    if format=='settlement-transfer-v1':
        from reflex.settlement_transfer import configuration
        config=configuration(variant)
    elif format=='competence-transfer-v1':
        from reflex.competence_transfer import configuration
        config=configuration(variant,p.genre)
    elif variant in ('legacy','finite-banded','finite'):
        from reflex.finite_transfer import configuration
        config=configuration(variant)
    elif variant in ('beam','shooting','hybrid'):
        from reflex.hybrid_transfer import configuration
        config=configuration(variant)
    else:config=dict(principle_priority='lexicographic',max_regret=.15)
    policy=Policy(principle_priority=config['principle_priority']);max_regret=config['max_regret']
    loop=DecisionLoop(c,progress=PersonaProgressWatch(c['scope'],ProgressConfig(grace=2,repeat_limit=2,proof_margin=0.)),
        pressure=NeedPressure(c['scope'],PressureConfig(grace=2)),policy=policy)
    intention=PlanIntention(c) if variant=='continuity' else None
    previous=None;models=0;proposals_count=0
    for t in r['trace']:
        assert digest(asdict(w))==digest(t['before']);c=p.observe(w,loop.state);pr=p.purpose(w,c,previous)
        captured={};saved_intention=None if intention is None else intention.record()
        def planner(cs,ds):
            nonlocal models,proposals_count
            cc=cs[0]
            if 'continuation_search' not in t['deliberation']['metadata']:
                paths,audit=rollout(cc,w,observe=p.observe,advance=p.advance,terminal=p.terminal,
                    assess=lambda s:p.goal(s).record(),horizon=h,seeds=p.seeds,policy=policy)
                models+=sum(len(x['actions']) for rows in audit.values() for x in rows)
                return goal_forecast(cc,paths,audit,horizon=h,unit='public-turns',target=p.target,policy=policy,max_regret=max_regret,settlement_weight=config.get('settlement_weight','discounted'))
            metadata=t['deliberation']['metadata'];search=metadata['continuation_search']
            offered=() if intention is None else intention.offer(cc,target=p.target,unit='public-turns',horizon=h)
            retained=search['proposals'].get('proposal-retained')
            viable={a['id'] for a in cc['actions'] if a['legal'] and not a['known_failure']}
            assert (retained is not None)==bool(offered and offered[0] in viable)
            if retained:assert (retained['root'],retained['schedule'])==(offered[0],list(offered[1:]))
            paths={};audits={};roots={};proposals_count+=len(search['proposals'])
            assert search['width']==2 and search['depth']==config.get('depth',2)
            if variant in ('beam','shooting','hybrid','guarded','validated','discounted','absolute'):
                assert search.get('samples',0)==config['samples']
                if config['samples']:
                    assert search['pilot_evaluations']==len(viable)*config['samples']
                    assert {x['root'] for x in search['sampling']}==viable
                    assert all(x['trials']==config['samples'] and 1<=x['unique_schedules']<=config['samples'] for x in search['sampling'])
            validation=search.get('validation')
            bank=config.get('validation_seeds')
            assert bool(validation)==bool(bank)
            if validation:
                assert validation['seeds']==list(bank) and validation['discovery_seeds']==list(p.seeds)
                assert not set(bank)&set(p.seeds) and seed not in bank
                assert validation['evaluated']==len(search['proposals'])
                assert set(validation['discovery'])==set(search['proposals'])
                # Retained discovery paths are independently replayed below.
                for name,n in search['proposals'].items():
                    _,pilot=rollout(cc,w,observe=p.observe,advance=p.advance,terminal=p.terminal,
                        assess=lambda s:p.goal(s).record(),horizon=h,seeds=p.seeds,policy=policy,
                        roots=(n['root'],),schedule=tuple(n['schedule']),record_choices=True)
                    assert pilot[n['root']]==validation['discovery'][name]
                    models+=sum(len(x['actions']) for x in pilot[n['root']])
            for name,n in search['proposals'].items():
                rows=[];roots[name]=n['root'];audits[name]=n['branches']
                assert [b['seed'] for b in n['branches']]==list(bank or p.seeds)
                for branch in n['branches']:
                    state=copy.deepcopy(w);memory=copy.deepcopy(cc['state']);flows=[];actions=[];choices=[];fallbacks=[]
                    for tick in range(h):
                        if p.terminal(state):flows.append(effect());continue
                        obs=copy.deepcopy(cc) if tick==0 else p.observe(state,memory)
                        b=compile_batch([obs]);d=policy.decide(b,False)
                        viable=sorted(a['id'] for a in obs['actions'] if a['legal'] and not a['known_failure']);choices.append(viable)
                        key=n['root'] if tick==0 else d.records(b)[0]['action_id']
                        if 0<tick<=len(n['schedule']):
                            wanted=n['schedule'][tick-1]
                            if wanted in viable:key=wanted
                            else:fallbacks.append(tick)
                        assert key in viable
                        memory=replace(d,action=np.array([b.ids[0].index(key)])).records(b)[0]['next_state']
                        state,flow=p.advance(state,key,branch['seed']);flows.append(flow);actions.append(key);models+=1
                    assert actions==branch['actions'] and choices==branch['choices'] and fallbacks==branch['schedule_fallbacks']
                    assert p.terminal(state)==branch['terminal'] and p.goal(state).record()==branch['assessment']
                    rows.append(Branch(1/len(bank or p.seeds),tuple(flows),(1.,)*h))
                paths[name]=tuple(rows)
            f=goal_forecast(cc,paths,audits,horizon=h,unit='public-turns',target=p.target,plan_roots=roots,policy=policy,max_regret=max_regret,settlement_weight=config.get('settlement_weight','discounted'))
            assert f.audit=={k:v for k,v in metadata.items() if k!='continuation_search'}
            f.audit['continuation_search']=search
            incumbent=next((k for k in f.roots if f.audit['plans'][k]['proposal']=='proposal-retained'),None)
            if incumbent is not None:
                f=replace(f,continuity=ArrivalPreference(incumbent,{k:tuple(g['settlement_step'] if g['status']=='success' else None
                    for g in f.audit['endpoint'][k]) for k in f.roots}))
            captured['forecast']=f
            return f
        req=Request(c,{k:Binding(k,'public',()) for k in p.keys(w)},{k:() for k in p.keys(w)},purpose=pr)
        result=DecisionLoop.decide_batch([(loop,req)],False,planner)[0]
        assert intention is None or intention.record()==saved_intention
        key=result['decision']['action_id'];assert key==t['root']
        assert result['deliberation']==t['deliberation'] and result.get('progress')==t['progress']
        before=w;w,row=p.actual(w,key,seed)
        assert w==real_step(before,key,p.genre,spec,seed),(variant,p.genre,profile['id'],seed,c['tick'],key)
        assert digest(asdict(w))==digest(t['after']) and row==t['flow'] and p.goal(w).record()==t['goal']
        fb=feedback(p,before,w,key,True,True)
        assert json.loads(json.dumps(asdict(fb)))==t['feedback'] and digest(asdict(pr))==digest(t['purpose'])
        loop.abandon(result['ticket'],need_progress=fb.needs,maintained=fb.maintained,completed=fb.completed,purpose_feedback=fb.purpose)
        if intention is not None:
            intention.remember(c,captured['forecast'],result['deliberation'].get('selected_plan'),key)
            assert intention.record()==t['intention']
            assert PlanIntention.from_record(c,t['intention']).record()==t['intention']
        assert loop.state==t['state'] and loop.progress.record()==t['watch'] and loop.pressure.record()==t['pressure']
        assert not loop.memory.entries and digest([c['personality'],c['values']])==t['persona_hash']
        previous=before
    assert p.terminal(w) and p.goal(w).status==r['status'] and p.goal(w).value==r['goal_value']
    if p.genre=='delivery':
        from reflex.delivery_world import Depot
        assert r['solvable_route_lost']==sum(can_finish(Depot(**t['before'])) and not can_finish(Depot(**t['after'])) for t in r['trace'])
    if 'search_cost' in r:
        searches=[t['deliberation']['metadata']['continuation_search'] for t in r['trace']]
        expected_cost=dict(shared_sequence_evaluations=sum(x['evaluated'] for x in searches),
            single_branch_pilots=sum(x.get('pilot_evaluations',0) for x in searches),
            model_transition_upper_bound=sum(h*len(p.seeds)*x['evaluated']+h*x.get('pilot_evaluations',0)+
                h*len(x.get('validation',{}).get('seeds',()))*x.get('validation',{}).get('evaluated',0) for x in searches))
        if format=='competence-transfer-v1':expected_cost['validation_evaluations']=sum(x.get('validation',{}).get('evaluated',0) for x in searches)
        assert r['search_cost']==expected_cost
    return dict(actual=len(r['trace']),model=models,proposals=proposals_count)


def audit(folder):
    folder=Path(folder);pre=json.loads((folder/'preregister.json').read_text(encoding='utf-8'))
    ev=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    assert source_hashes()==pre['source_hashes']==ev['source_hashes'] and digest(pre)==ev['preregister_digest']
    rows=[json.loads(s) for s in (folder/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==len(pre['jobs'])==ev['episodes']
    assert [{k:v for k,v in r.items() if k!='trace'} for r in rows]==ev['runs']
    assert aggregate(ev['runs'])==ev['summary']
    pairs=[]
    for a in ev['runs']:
        if a['variant']!=pre.get('candidate','search'):continue
        b=next(b for b in ev['runs'] if b['variant']==pre.get('reference','verified') and
            (b['split'],b['spec'],b['profile'],b['seed'])==(a['split'],a['spec'],a['profile'],a['seed']))
        pairs.append(dict(split=a['split'],genre=a['spec']['genre'],spec=a['spec'],profile=a['profile'],
            baseline_status=b['status'],candidate_status=a['status'],goal_delta=a['goal_value']-b['goal_value'],
            survival_regression=b['surviving'] is True and a['surviving'] is False,
            shortages_delta=(a['shortages'] or 0)-(b['shortages'] or 0),
            lost_route_delta=(a['solvable_route_lost'] or 0)-(b['solvable_route_lost'] or 0)))
    assert pairs==ev['pairs']
    hold=[p for p in pairs if p['split']=='holdout']
    regressions=[p for p in hold if p['goal_delta']<0 or p['survival_regression'] or p['shortages_delta']>0 or p['lost_route_delta']>0 or
        (p['baseline_status']=='success' and p['candidate_status']!='success')]
    assert regressions==ev['holdout_regressions']
    gained=any((p['genre']=='delivery' or pre.get('candidate')!='search') and p['baseline_status']!='success' and p['candidate_status']=='success' for p in hold)
    recovered=pre.get('candidate')!='continuity' or any(r['split']=='diagnostic' and r['variant']=='continuity' and
        r['spec']==dict(genre='resources',route='mixed',limit=14) and r['profile']=='ego' and r['status']=='success' for r in ev['runs'])
    additional={}
    if pre['format']=='finite-principle-transfer-v1':
        additional=dict(known_search_failure_recovered=any(r['split']=='diagnostic-search' and r['variant']=='finite' and
            r['profile']=='ego' and r['status']=='success' for r in ev['runs']))
        assert ev['additional_requirements']==additional
    assert ev['broad_adoption_passed']==(not regressions and gained and recovered and all(additional.values()))
    with ProcessPoolExecutor(max_workers=4) as pool:counts=list(pool.map(replay_episode,((r,j,pre['format']) for r,j in zip(rows,pre['jobs']))))
    result=dict(episodes=len(rows),actual_rule_transitions=sum(c['actual'] for c in counts),
        replayed_model_transitions=sum(c['model'] for c in counts),retained_proposals=sum(c['proposals'] for c in counts),
        source_freeze_verified=True,shared_schedule_and_selection_replay=True,actual_checkpoints_replayed=True,
        adoption_gate_recomputed=True,intention_checkpoints_replayed=any(r['variant']=='continuity' for r in rows),
        independent_validation_bank_replayed=any('validation' in t['deliberation']['metadata'].get('continuation_search',{}) for r in rows for t in r['trace']),
        trajectories_sha256=hashlib.sha256((folder/'trajectories.jsonl').read_bytes()).hexdigest())
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':audit(sys.argv[1])
