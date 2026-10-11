"""Replay a published negative witness; no production game-specific remedy.

The proposed recovery was expert-authored after observing a failure. This tool
never discovers it or measures general performance. Optional re-search checks
whether an injected authored program survives actual subsequent decisions.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import Policy,digest
from reflex.contingent_games import Frame,make_bridge
from reflex.resource_world import world_from_record
from reflex.observation_plan import ObservationPlan,PlanMemory,View
from reflex.contingent_search import search_forecast
from reflex.flow_rollout import rollout
from reflex.purpose_plan import goal_forecast
from reflex.deliberation import select

def probe(record):
    if record['case']['game']!='empire_race' or not record['not_production_policy']:
        raise ValueError('recorded negative resource-world witness required')
    witness=record['constructive_witness'];b=make_bridge(witness['profile'],record['case'])
    initial=Frame(world_from_record(record['from_position']['position']),record['from_position']['turns'])
    actual=initial
    for key in witness['path']:actual,_=b.actual(actual,key,record['seed'])
    assert b.goal(actual).record()==witness['goal'] and b.goal(actual).status=='success'
    c=b.observe(initial,witness['owner_state']);p=witness['program']
    # Historical source predates version-tag serialization. Validate the same
    # owner/purpose/unit before constructing a current immutable program.
    program=ObservationPlan.make(c,p['target'],p['unit'],dict(p['entries']))
    assert program.owner==p['owner'] and program.target==b.target and program.unit==b.unit
    rows=[]
    for name in ('lexicographic','finite'):
        policy=Policy(principle_priority=name)
        options=dict(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,
            assess=lambda s:b.goal(s).record(),horizon=len(witness['path']),seeds=(401,402),policy=policy)
        paths,audit=rollout(c,initial,roots=(witness['path'][0],),public_controller=program.choose,**options)
        assert all(t['assessment']['status']=='success' and all(n['supplied'] for n in t['public_nodes'][1:]) for t in audit[witness['path'][0]])
        base,base_audit=rollout(c,initial,public_controller=lambda view:None,**options)
        paths={**base,'diagnostic-recovery':paths[witness['path'][0]]}
        audit={**base_audit,'diagnostic-recovery':audit[witness['path'][0]]}
        f=goal_forecast(c,paths,audit,horizon=len(witness['path']),target=b.target,unit=b.unit,
            plan_roots={**{k:k for k in base},'diagnostic-recovery':witness['path'][0]},settlement_weight='absolute',policy=policy)
        decisions,vote=select([c],f,policy=policy)
        rows.append(dict(priority=name,selected_root=decisions[0]['action_id'],selected_purpose=vote['selected_purpose'],
            pre_tier_recovery_goal=1.,post_tier_best_purpose=vote['best_purpose'],candidate_count=vote['plans']))
    return dict(format='negative-witness-replay-v1',actual_legal_path_replayed=True,path_length=len(witness['path']),
        programme_nodes=len(program.entries),same_static_persona=True,adoption=rows,
        warning='post-study authored path; root adoption only; not learned/discovered recovery, actual re-search success or general strength')


def replan(record):
    if record['case']['game']!='empire_race' or not record['not_production_policy']:
        raise ValueError('recorded negative resource-world witness required')
    witness=record['constructive_witness'];b=make_bridge(witness['profile'],record['case'])
    initial=Frame(world_from_record(record['from_position']['position']),record['from_position']['turns'])
    results=[]
    for name,supplied in [('lexicographic',True),('finite',True),('finite',False)]:
        world=initial;state=witness['owner_state'];c=b.observe(world,state);memory=PlanMemory(c,b.target,b.unit)
        if supplied:memory.plan=ObservationPlan.make(c,b.target,b.unit,dict(witness['program']['entries']))
        policy=Policy(principle_priority=name);steps=[]
        while not b.terminal(world):
            c=b.observe(world,state);retained,prefix=memory.offer(c)
            match=None if retained is None else retained.choose(View.make(prefix,c))
            f=search_forecast(c,world,observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,
                assess=lambda s:b.goal(s).record(),horizon=12,seeds=(401,402),width=1,revisions=0,
                patch_budget=1,target=b.target,unit=b.unit,retained=retained,prefix=prefix,
                policy=policy,settlement_weight='absolute')
            decisions,vote=select([c],f,policy=policy);assert decisions is not None
            key=decisions[0]['action_id'];memory.remember(c,f,vote.get('selected_plan'),key)
            before=b.record(world);world,_=b.actual(world,key,record['seed']);state=decisions[0]['next_state']
            assert PlanMemory.restore(c,memory.record()).record()==memory.record()
            steps.append(dict(before=before,after=b.record(world),action=key,retained_match=match,
                goal=b.goal(world).record(),selected_purpose=vote['selected_purpose']))
            assert len(steps)<=20
        results.append(dict(priority=name,initial_expert_plan=supplied,horizon=12,revisions=0,
            steps=steps,goal=b.goal(world).record()))
    return dict(format='negative-witness-replanning-v1',original_context=digest(b.observe(initial,witness['owner_state'])),
        warning='post-study selected failure; supplied expert plan; no independent samples or discovery/general-strength claim',results=results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',required=True);p.add_argument('--output',required=True)
    p.add_argument('--replan',action='store_true');args=p.parse_args()
    result=(replan if args.replan else probe)(json.loads(Path(args.record).read_text(encoding='utf-8')))
    target=Path(args.output)
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    summary={k:v for k,v in result.items() if k!='results'}
    if 'results' in result:
        summary['results']=[dict(priority=r['priority'],initial_expert_plan=r['initial_expert_plan'],
            steps=len(r['steps']),goal=r['goal']) for r in result['results']]
    print(json.dumps(summary,ensure_ascii=False))
