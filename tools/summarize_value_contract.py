"""Compare registered complete worlds; do not turn partial data into acceptance."""
import argparse,hashlib,json,sys
from itertools import combinations
from collections import Counter,defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
from tools.summarize_horizon_study import realized_account
from tools.value_contract_config import configurations,jobs


def read_run(path,job,enabled,plan,plan_hash):
    side='candidate' if enabled else 'baseline'
    configuration=configurations(plan)[side];paired_jobs=jobs({'job':job},plan)
    source=path/'trajectories.jsonl';audit=json.loads((path/'audit.json').read_text())
    assert audit['complete'] and audit['source_freeze_verified'] and audit['completed_runs']==1
    assert audit['trajectory_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    proof=json.loads((path/'contract_pair.json').read_text())
    assert proof['complete'] and proof['return_codes']==[0,0] and proof['job']==job
    assert proof.get('jobs',dict(baseline=job,candidate=job))==paired_jobs
    assert proof['sources']==plan['sources'] and proof['plan_sha256']==plan_hash
    assert proof['configurations']==configurations(plan)
    manifest=json.loads((path/'source_snapshot.json').read_text())
    assert {k:v for k,v in manifest.items() if k.endswith('.py')}==plan['sources']
    for k,v in manifest.items():assert hashlib.sha256((path/'_source'/k).read_bytes()).hexdigest()==v
    job=paired_jobs[side]
    row=json.loads(source.read_text());g,p,s,r,t,i,h,o=job
    from tools.audit_recorded_forecasts import verify_receipt
    recipe=verify_receipt(path,audit,row)
    assert [row[k] for k in ('game','profile','seed','regime','turns','interval','tail_steps','offset')]==[g,PROFILES[p]['id'],s,r,t,i,h,o]
    for key in ('cumulative_achievement','tail_goal_refinement','sampling_check','tail_replanning'):
        assert row.get(key,False)==configuration.get(key,False)
    assert row.get('tail_replanning_horizon',2)==configuration.get('tail_replanning_horizon',2)
    assert 'sampling_standard_errors' not in row and 'model_seeds' not in row and 'tail_recovery_limit' not in row
    reg=json.loads((path/'registration.json').read_text())
    assert reg['jobs']==[job] and reg['stage']==plan['stage']
    assert reg.get('configuration',{})==configuration
    assert 0<len(row['trace'])==row['executed']<=t
    for k in ('world_rng_replay','restored_action_feedback','physical_replay','owner_parity','fixed_axes'):
        assert audit['checks'][k]==row['executed']
    game=ContinuingGame(g,PROFILES[p],t,s%2)
    assert game.terminal(row['final']) and row['trace'][-1]['after']==row['final']
    for step in row['trace']:
        if configuration.get('cumulative_achievement',False):
            basis=step['cumulative_preference']
            assert basis['channels']==[['values.achievement',float(t)]]
            assert basis['origin']==game.goal(step['before']).record()
        else:assert 'cumulative_preference' not in step
    if 'host_segments' in proof:
        from tools.resume_horizon_pair import verify_segmented_result
        verify_segmented_result(path,row,proof,int(enabled))
    return row,proof,dict(path=source.relative_to(R).as_posix(),sha256=audit['trajectory_sha256'],checks=audit['checks'],recorded_forecast_replay=recipe)



def adaptation(row):
    """Observed response-learning windows; never supplied to the playing owner."""
    trace=row['trace'];starts=[0]+[i for i in range(1,len(trace)) if trace[i]['truth']!=trace[i-1]['truth']]
    result=[]
    for start,end in zip(starts,starts[1:]+[len(trace)]):
        part=trace[start:end];width=min(4,len(part));errors=[abs(t['forecast_error']) for t in part if t.get('forecast_error') is not None]
        result.append(dict(start=start,end=end,first_brier=sum(t['brier'] for t in part[:width])/width,
            last_brier=sum(t['brier'] for t in part[-width:])/width,
            forecast_count=len(errors),forecast_absolute_error=sum(errors)/len(errors) if errors else None,
            actions=dict(Counter(t['action'] for t in part)),
            goal_at_end=part[-1]['goal']))
    assert sum(x['end']-x['start'] for x in result)==len(trace)
    return result


def main(root,plan_path,partial,additional_roots=()):
    root=R/root;plan_path=R/plan_path;plan=json.loads(plan_path.read_text())
    plan_hash=hashlib.sha256(plan_path.read_bytes()).hexdigest();pairs=[];pending=[];rows={}
    for spec in plan['cases']:
        candidates=[[base/f'value-contract-{spec["key"]}-{arm}' for arm in ('baseline','candidate')]
                    for base in (root,*(R/p for p in additional_roots))]
        finished=[paths for paths in candidates if all((p/'audit.json').exists() and (p/'trajectories.jsonl').exists() and
                   json.loads((p/'contract_pair.json').read_text())['complete'] for p in paths)]
        assert len(finished)<=1,('duplicate completed pair',spec['key'])
        if not finished:pending.append(spec['key']);continue
        paths=finished[0]
        a,ap,ar=read_run(paths[0],spec['job'],False,plan,plan_hash)
        b,bp,br=read_run(paths[1],spec['job'],True,plan,plan_hash)
        assert ap==bp and a['trace'][0]['before']==b['trace'][0]['before']
        lt,rt=a['trace'],b['trace'];t=a['turns']
        assert all(x['response']==y['response'] and x['truth']==y['truth'] for x,y in zip(lt,rt))
        la,ra=realized_account(a),realized_account(b)
        row=dict(condition=spec['key'],game=a['game'],profile=a['profile'],seed=a['seed'],regime=a['regime'],turns=t,
            baseline_score=a['score'],candidate_score=b['score'],delta=b['score']-a['score'],
            conditional_increment_delta=ra['conditional_increments']-la['conditional_increments'],
            realized_residual_delta=ra['residual']-la['residual'],
            phases=[dict(tick=k,delta=rt[min(k,len(rt))-1]['goal']-lt[min(k,len(lt))-1]['goal']) for k in sorted({t//4,t//2,t*3//4,t-16,t}) if k>0],
            changed_actions=sum(x['action']!=y['action'] for x,y in zip(lt,rt)),
            baseline_actions=dict(Counter(x['action'] for x in lt)),candidate_actions=dict(Counter(x['action'] for x in rt)),
            identical_world_predictions=sum(x['law']==y['law'] for x,y in zip(lt,rt)),
            paired_external_events=min(len(lt),len(rt)),baseline_executed=len(lt),candidate_executed=len(rt),
            phase_semantics='Actual executed decisions; a settled score stays fixed after early terminal death. No fabricated later actions or feedback.',baseline=ar,candidate=br,pair_id=ap['pair_id'],
            response_learning=dict(baseline=adaptation(a),candidate=adaptation(b)))
        if 'bellman' in la:
            row.update(bellman_residual_removed_delta=ra['bellman']['residual_removed']-la['bellman']['residual_removed'],
                       bellman_residual_delta=ra['bellman']['residual']-la['bellman']['residual'])
        pairs.append(row);rows[spec['key']]=(a,b)
    groups=[]
    for level in ('game','world_seed','persona','condition'):
        bins=defaultdict(list)
        for r in pairs:
            key=(r['game'],)
            if level=='world_seed':key+=(r['seed'],)
            if level in ('persona','condition'):key+=(r['profile'],)
            if level=='condition':key+=(r['regime'],)
            bins[key].append(r)
        for key,items in sorted(bins.items()):
            ds=[r['delta'] for r in items]
            groups.append(dict(level=level,key=key,conditions=len(items),world_seeds=sorted({r['seed'] for r in items}),
                mean_delta=sum(ds)/len(ds),mean_conditional_increment_delta=sum(r['conditional_increment_delta'] for r in items)/len(items),
                better=sum(v>1e-12 for v in ds),same=sum(abs(v)<=1e-12 for v in ds),worse=sum(v< -1e-12 for v in ds),worst=min(ds)))
    persona=[];same_world=defaultdict(list)
    for key,(a,b) in rows.items():same_world[a['game'],a['seed'],a['regime']].append((a,b))
    for key,items in same_world.items():
        if len(items)<2:continue
        for left,right in combinations(items,2):
            assert left[0]['trace'][0]['before']==right[0]['trace'][0]['before']
            persona.append(dict(world=key,profiles=[left[0]['profile'],right[0]['profile']],
                baseline_action_differences=sum(x['action']!=y['action'] for x,y in zip(left[0]['trace'],right[0]['trace'])),
                candidate_action_differences=sum(x['action']!=y['action'] for x,y in zip(left[1]['trace'],right[1]['trace'])),
                baseline_shared_ticks=min(len(left[0]['trace']),len(right[0]['trace'])),
                candidate_shared_ticks=min(len(left[1]['trace']),len(right[1]['trace'])),
                limitation='All profile pairs within the same world, not independent world samples. Different behavior is not required when interests agree; not a claim of human believability.'))
    if not partial:assert not pending,pending
    result=dict(complete=not pending,stage=plan['stage'],plan_sha256=plan_hash,registered_conditions=len(plan['cases']),
        completed_conditions=len(pairs),pending=pending,comparisons=pairs,groups=groups,persona_differences=persona,
        limitation='No cross-game average. Shared-seed personas are not independent world samples. Partial summaries cannot pass acceptance. Residual removal does not eliminate state effects of earlier luck or isolate all intentional preference costs.')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','stage','registered_conditions','completed_conditions','pending')}))
    print(json.dumps([r for r in groups if r['level']=='game'],indent=2))


def cli():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--additional-roots',nargs='*',default=[]);p.add_argument('--plan',required=True);p.add_argument('--partial',action='store_true');a=p.parse_args()
    main(a.root,a.plan,a.partial,a.additional_roots)


if __name__=='__main__':cli()
