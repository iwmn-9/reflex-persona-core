"""Read-only trajectory instrumentation for a saved owner's finite continuation."""
import argparse,copy,hashlib,json,sys,time
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.closed_loop import owned_rollout
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
from tools.run_owned_value import cumulative_basis


def diagnose(row,tick,tail,recovery_limit=None,*,seeds=None):
    recovery_limit=row.get('tail_recovery_limit',16) if recovery_limit is None else recovery_limit
    seeds=tuple(row.get('model_seeds',(701,702,703))) if seeds is None else seeds
    t=row['trace'][tick];assert t['tick']==tick
    g=ContinuingGame(row['game'],next(p for p in PROFILES if p['id']==row['profile']),row['turns'],row['seed']%2)
    w=copy.deepcopy(t['before']);c=g.context(w)
    own=DecisionLoop.from_record(c,t['checkpoint'],predictor=g.actor(c,learning='segmented').predictor)
    before=digest(own.record())
    ctrl=owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,
        assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
        feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=tail,
        world_model='posterior',purpose_estimator='conditional',tail_recovery=2,tail_recovery_limit=recovery_limit,seeds=seeds,
        sampling_standard_errors=row.get('sampling_standard_errors'),
        cumulative_preference=cumulative_basis(g,row.get('cumulative_achievement',False)))
    frontier=ctrl.frontier;distribute=frontier.distribution;plan=frontier.future_planner;advance=frontier.owner_advance
    adjust=frontier.purpose_adjustment
    traces=[];active=None;pending=None;latest=None
    def watched_plan(person,request,state):
        nonlocal pending
        assert pending is None
        result=plan(person,request,state)
        pending='planned' if result is not None else 'declined'
        return result
    def watched_advance(state,key,seed,person,context):
        nonlocal pending,latest
        after,flow=advance(state,key,seed,person,context)
        latest=dict(tick=state['tick'],action=key,controller=pending or 'budget-skipped',
            before_goal=g.goal(state).value,after_goal=g.goal(after).value,after_terminal=g.terminal(after),
            after_state=copy.deepcopy(after))
        active['paths'].setdefault(str(seed),[]).append(latest)
        pending=None
        return after,flow
    def watched_adjustment(*args):
        value=adjust(*args);latest['purpose_adjustment']=value;return value
    def watched_distribution(world,person):
        nonlocal active
        assert active is None and pending is None
        active=dict(start=copy.deepcopy(world),paths={});traces.append(active)
        result=distribute(world,person)
        active['endpoint_estimates']=result.purpose_estimates
        active['trajectories']=result.trajectories
        active=None;assert pending is None
        return result
    frontier.future_planner=watched_plan;frontier.owner_advance=watched_advance;frontier.distribution=watched_distribution
    frontier.purpose_adjustment=watched_adjustment
    req=g.request(c);start=time.monotonic();capture={};detached=copy.deepcopy(own)
    selected=DecisionLoop.decide_batch([(detached,req)],False,ctrl.planner(detached,req,w,captured=capture))[0]['decision']['action_id']
    assert capture['audit']['complete']
    forecast=capture['node']['forecast']
    assert digest(own.record())==before and own.pending is None,'diagnostic mutated factual owner'
    counts=Counter(s['controller'] for d in traces for steps in d['paths'].values() for s in steps)
    paths=[steps for d in traces for steps in d['paths'].values()]
    return dict(tick=tick,owner_digest=before,saved_action=t['action'],selected_action=selected,tail_steps=tail,recovery_limit=recovery_limit,seeds=list(seeds),
        seconds=time.monotonic()-start,controller_counts=dict(counts),paths=len(paths),
        capped_paths=sum(any(s['controller']=='budget-skipped' for s in steps) for steps in paths),
        first_skipped_tick=min((s['tick'] for steps in paths for s in steps if s['controller']=='budget-skipped'),default=None),
        distributions=traces,forecast=forecast.audit,frontiers=capture['node']['frontiers'],children=capture['node']['children'],
        **({'sampling_standard_errors':row['sampling_standard_errors']} if 'sampling_standard_errors' in row else {}),
        **({'cumulative_achievement':True} if row.get('cumulative_achievement') else {}))


def pool_diagnostics(diagnostics):
    """Pool independent paths BEFORE personality gates and risk evaluation.

    This is an offline sampling diagnostic, not a new playing controller. Votes,
    already filtered purpose scores, or different future models cannot be merged
    this way. Disjoint seeds retain the same person's complete correlated flows.
    """
    if not 2<=len(diagnostics)<=8:raise ValueError('two to eight independent banks required')
    first=diagnostics[0];banks=[]
    if not first.get('owner_digest') and not first.get('source_sha256'):
        raise ValueError('owner or saved trajectory identity required')
    for d in diagnostics:
        for key in ('tick','tail_steps','recovery_limit','children','core_sources','owner_digest','source_sha256','game','profile','regime','seed','sampling_standard_errors','cumulative_achievement'):
            if d.get(key)!=first.get(key):raise ValueError('identical owner, model and horizon required')
        seeds=d.get('seeds')
        if seeds is None and d['distributions']:seeds=list(map(int,d['distributions'][0]['paths']))
        if not seeds or len(set(seeds))!=len(seeds):raise ValueError('explicit unique bank seeds required')
        banks.append(seeds)
    union=[s for bank in banks for s in bank]
    if len(set(union))!=len(union):raise ValueError('overlapping banks are not independent samples')
    keys=[(f['root'],digest(f['public'])) for f in first['frontiers']]
    result={k:copy.deepcopy(first[k]) for k in ('tick','tail_steps','recovery_limit','children')}
    if 'sampling_standard_errors' in first:result['sampling_standard_errors']=first['sampling_standard_errors']
    if first.get('cumulative_achievement'):result['cumulative_achievement']=True
    result.update(seeds=union,frontiers=[dict(root=f['root'],public=copy.deepcopy(f['public'])) for f in first['frontiers']],distributions=[])
    for d in diagnostics:
        if [(f['root'],digest(f['public'])) for f in d['frontiers']]!=keys:
            raise ValueError('identical conditional roots required')
    for index in range(len(keys)):
        start=first['distributions'][index]['start'];paths={};trajectories=[]
        for d,bank in zip(diagnostics,banks):
            source=d['distributions'][index]
            if source['start']!=start or set(source['paths'])!=set(map(str,bank)):
                raise ValueError('complete same-root bank paths required')
            paths.update(copy.deepcopy(source['paths']))
            for trajectory in source['trajectories']:
                entry=copy.deepcopy(list(trajectory));entry[0]*=len(bank)/len(union);trajectories.append(entry)
        result['distributions'].append(dict(start=copy.deepcopy(start),paths=paths,trajectories=trajectories))
    return result


def prefix_choice(row,tick,diagnostic,horizon):
    """Revalue a recorded prefix; no extra simulation or different model sample.

    Valid only for this fixed two-step recovery continuation. Extending its tail
    changes neither earlier predictions nor earlier owner choices. This is not
    a general truncation rule for arbitrary horizon-dependent future planners.
    """
    from reflex.examples import effect
    from reflex.flow_rollout import Branch
    from reflex.purpose_plan import goal_forecast
    if row.get('sampling_standard_errors')!=diagnostic.get('sampling_standard_errors'):
        raise ValueError('matching sampling-margin configuration required')
    if row.get('cumulative_achievement',False)!=diagnostic.get('cumulative_achievement',False):
        raise ValueError('matching cumulative preference configuration required')
    assert 1<=horizon<=diagnostic['tail_steps']+1 and tick==diagnostic['tick']
    g=ContinuingGame(row['game'],next(p for p in PROFILES if p['id']==row['profile']),row['turns'],row['seed']%2)
    w=copy.deepcopy(row['trace'][tick]['before']);c=g.context(w)
    own=DecisionLoop.from_record(c,row['trace'][tick]['checkpoint'],predictor=g.actor(c,learning='segmented').predictor)
    req=g.request(c);effective=own.preview(req)[0];paths={};audit={};purpose_samples={}
    sampling_margin=row.get('sampling_standard_errors')
    seeds=diagnostic.get('seeds')
    if sampling_margin is not None and seeds is None:raise ValueError('explicit aligned model seeds required')
    mapping={(f['root'],digest(f['public'])):d for f,d in zip(diagnostic['frontiers'],diagnostic['distributions'])}
    children_by_root=diagnostic.get('children')
    if children_by_root is None:
        # Earlier instrumentation retained branch masses but not tree children.
        # Recover only when all children are present as nonterminal frontiers.
        plans={p['root']:p['paths'] for p in diagnostic['forecast']['plans'].values()};offsets=Counter();children_by_root={}
        for f,d in zip(diagnostic['frontiers'],diagnostic['distributions']):
            key=f['root'];count=len(d['trajectories']);start=offsets[key]
            probability=sum(p['probability'] for p in plans[key][start:start+count])
            children_by_root.setdefault(key,[]).append(dict(public=f['public'],probability=probability))
            offsets[key]+=count
        assert all(offsets[k]==len(v) for k,v in plans.items()),'legacy recording omits terminal children'
    for key,children in children_by_root.items():
        paths[key]=[];audit[key]=[]
        if sampling_margin is not None:purpose_samples[key]=[0. for _ in seeds]
        for child in children:
            public=child['public'];event=next(e for e in g.events(w,key) if e.public==public)
            if g.terminal(public):
                tails=[(1.,(),(),g.goal(public).record(),True)];details=[[]]
            else:
                recorded=mapping[key,digest(public)]
                assert recorded['start']==public
                if sampling_margin is not None:assert list(recorded['paths'])==list(map(str,seeds))
                tails=recorded['trajectories'];details=list(recorded['paths'].values())
            truncated=[]
            for (p,flows,actions,goal,ended),steps in zip(tails,details):
                count=min(horizon-1,len(flows))
                goal=g.goal(public).record() if not count else dict(target=g.target,
                    status='scored' if steps[count-1]['after_terminal'] else 'running',value=steps[count-1]['after_goal'])
                ended=goal['status']!='running'
                mean=goal['value']+sum(s['purpose_adjustment'] for s in steps[:count])
                if sampling_margin is not None:
                    if g.terminal(public):
                        for i in range(len(seeds)):purpose_samples[key][i]+=child['probability']*mean
                    else:
                        index=len(truncated)
                        purpose_samples[key][index]+=child['probability']*mean
                effects=(event.flow,)+tuple(flows[:count])+(effect(),)*(horizon-count-1)
                branch=Branch(child['probability']*p,effects,(1.,)*horizon)
                record=dict(actions=[key]+list(actions[:count]),terminal=ended,assessment=goal,purpose_mean_estimate=mean)
                truncated.append((branch,record))
            coverage=all(r['terminal'] for _,r in truncated)
            for branch,record in truncated:
                record['terminal_coverage']=coverage;paths[key].append(branch);audit[key].append(record)
    effective['actions']=[a for a in effective['actions'] if a['id'] in paths]
    forecast=goal_forecast(effective,paths,audit,horizon=horizon,unit='owner-decisions',target=g.target,
        policy=own.policy,max_regret=.15,settlement_weight='absolute',outcome_limit=256,
        objective_origin=g.goal(w),opportunity_regret=True,
        cumulative_preference=cumulative_basis(g,row.get('cumulative_achievement',False)))
    if sampling_margin is not None:
        from dataclasses import replace
        from reflex.sampling_band import PurposeSampling
        forecast=replace(forecast,purpose_sampling=PurposeSampling(tuple(seeds),
            tuple((plan,tuple(purpose_samples[roots[0]])) for plan,roots in forecast.roots.items()),
            'recorded-prefix',sampling_margin))
    result=DecisionLoop.decide_batch([(own,req)],False,lambda *args:forecast)[0]
    return dict(horizon=horizon,action=result['decision']['action_id'],
        purpose={forecast.roots[k][0]:v for k,v in forecast.purpose.items()},audit=forecast.audit)


def main():
    p=argparse.ArgumentParser();p.add_argument('study');p.add_argument('--profile',required=True)
    p.add_argument('--regime',required=True);p.add_argument('--tick',type=int,required=True)
    p.add_argument('--tail',type=int,default=63);p.add_argument('--recovery-limit',type=int)
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.study)/'trajectories.jsonl'
    row=next(x for x in map(json.loads,source.read_text().splitlines())
        if x['arm']=='receding_recovery' and x['profile']==a.profile and x['regime']==a.regime)
    result=diagnose(row,a.tick,a.tail,a.recovery_limit,seeds=tuple(a.seeds) if a.seeds is not None else None)
    result.update(game=row['game'],profile=row['profile'],regime=row['regime'],seed=row['seed'],
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        core_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve().parents[1]/'reflex').glob('*.py')},
        limitation='Read-only model continuation under the saved owner, not a realized outcome.')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('forecast','distributions','core_sources','frontiers','children')}),flush=True)


if __name__=='__main__':main()
