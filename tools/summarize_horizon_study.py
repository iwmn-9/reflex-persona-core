"""Match fixed-horizon worlds without treating personas as independent seeds."""
import argparse,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame


def realized_account(row):
    """Separate realized one-step noise from observed actions, for diagnosis.

    The true response law belongs only to this offline evaluator. Conditional
    increments include personality and accumulated resources, which can still
    reflect earlier luck. They are not a pure intelligence score, a removal of
    all chance effects, or an off-policy value estimate.
    """
    profile=next(p for p in PROFILES if p['id']==row['profile'])
    game=ContinuingGame(row['game'],profile,row['turns'],row['seed']%2)
    initial=game.goal(game.start()).value;expected=0.;actual=0.;parts=[]
    for t in row['trace']:
        before=game.goal(t['before']).value
        gain=game.goal(t['after']).value-before
        conditional=sum(prob*(game.goal(game.transition(t['before'],t['action'],z)[0]).value-before)
                        for z,prob in t['truth'].items())
        expected+=conditional;actual+=gain
        parts.append(dict(tick=t['tick'],conditional=conditional,residual=gain-conditional))
    assert abs(initial+actual-row['score'])<1e-10,'actual score must telescope'
    residual=actual-expected
    result=dict(initial=initial,conditional_increments=expected,residual=residual,
                reconstruction_error=initial+expected+residual-row['score'],increments=parts)
    if row['game']=='auction':
        from tools.diagnose_auction_continuity import reference
        from tools.run_owned_value import law
        q=reference(row['turns'],row['seed']%2,lambda t:tuple(law(t,row['turns'],row['regime']).values()))
        value=lambda t,w:((w['score']-w['rival'])/(8*row['turns'])+
            (.25*w['cash']/(8*row['turns']) if t==row['turns'] else max(q(t,int(w['cash'])).values())))
        start=value(0,game.start());loss=0.;noise=0.
        for t in row['trace']:
            choices=q(t['tick'],int(t['before']['cash']))
            loss+=max(choices.values())-choices[t['action']]
            chosen=(t['before']['score']-t['before']['rival'])/(8*row['turns'])+choices[t['action']]
            noise+=value(t['tick']+1,t['after'])-chosen
        assert abs(start-loss+noise-row['score'])<1e-10,'Bellman account must reconstruct final score'
        result['bellman']=dict(initial_perfect_information=start,purpose_opportunity_loss=loss,
                              residual=noise,residual_removed=start-loss,
                              reconstruction_error=start-loss+noise-row['score'])
    elif row['game']=='combat':
        from tools.combat_reference import account
        result['bellman']=account(row)
    return result


def summarize(repo,root,plan,partial=False,additional_roots=()):
    rows={};origins={};missing=[];incomplete=[];accounts={}
    def add(row,origin):
        key=tuple(row[k] for k in ('game','profile','seed','regime'))+(row['tail_steps']+1,)
        assert key not in rows,('duplicate condition',key)
        rows[key]=row;origins[key]=str(origin)
    for old in plan['retained_baselines']:
        source=repo/old['path']/'trajectories.jsonl'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==old['sha256']
        selected=[r for r in map(json.loads,source.read_text().splitlines()) if r['arm']=='receding_recovery']
        assert sorted([r['game'],r['profile'],r['regime']] for r in selected)==sorted(old['conditions'])
        for row in selected:add(row,source)
    for job in plan['new_jobs']:
        candidates=[base/('horizon-'+job['key']) for base in (root,*additional_roots)]
        finished=[p for p in candidates if (p/'trajectories.jsonl').exists() and (p/'audit.json').exists()]
        assert len(finished)<=1,('duplicate completed execution',job['key'])
        if not finished:
            (incomplete if any(p.exists() for p in candidates) else missing).append(job['key']);continue
        path=finished[0];source=path/'trajectories.jsonl'
        audit=json.loads((path/'audit.json').read_text())
        assert audit['complete'] and audit['source_freeze_verified'] and audit['completed_runs']==1
        assert audit['trajectory_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        row=json.loads(source.read_text());g,p,s,r,turns,interval,tail,offset=job['job']
        assert [row['game'],row['profile'],row['seed'],row['regime'],row['turns'],row['interval'],row['tail_steps'],row['offset']]==[g,PROFILES[p]['id'],s,r,turns,interval,tail,offset]
        add(row,source)
    if not partial:assert not missing and not incomplete,(missing,incomplete)
    comparisons=[]
    for key,right in sorted(rows.items()):
        if key[-1]==16 or key[:-1]+(16,) not in rows:continue
        baseline_key=key[:-1]+(16,);left=rows[baseline_key];lt=left['trace'];rt=right['trace']
        paired_host=None
        if plan.get('execution_pairing'):
            pair_records=[json.loads((Path(origins[k]).parent/'pair_execution.json').read_text()) for k in (baseline_key,key)]
            assert pair_records[0]==pair_records[1] and pair_records[0]['complete'] and pair_records[0]['same_source'],'incomplete or unmatched host pairing'
            snapshots=[json.loads((Path(origins[k]).parent/'source_snapshot.json').read_text()) for k in (baseline_key,key)]
            assert snapshots[0]==snapshots[1],'paired source drift'
            paired_host=pair_records[0]['pair_id']
        assert left['turns']==right['turns'] and left['interval']==right['interval']==1
        for a,b in zip(lt,rt):assert a['response']==b['response'] and a['truth']==b['truth'],'unmatched world events'
        for k in (baseline_key,key):
            if k not in accounts:accounts[k]=realized_account(rows[k])
        la=accounts[baseline_key];ra=accounts[key]
        def score_at(trace,tick):return trace[min(tick,len(trace))-1]['goal']
        checkpoint=max(1,right['turns']-16)
        gains=dict(conditional_increment_delta=ra['conditional_increments']-la['conditional_increments'],
                   realized_residual_delta=ra['residual']-la['residual'])
        if 'bellman' in la:
            gains.update(bellman_residual_removed_delta=ra['bellman']['residual_removed']-la['bellman']['residual_removed'],
                         bellman_residual_delta=ra['bellman']['residual']-la['bellman']['residual'])
        comparisons.append(dict(game=key[0],profile=key[1],seed=key[2],regime=key[3],horizon=key[4],
            delta=right['score']-left['score'],baseline_score=left['score'],candidate_score=right['score'],
            tick112_delta=score_at(rt,112)-score_at(lt,112),
            preterminal_tick=checkpoint,preterminal_delta=score_at(rt,checkpoint)-score_at(lt,checkpoint),
            **gains,
            phase_gains=[dict(tick=t,delta=score_at(rt,t)-score_at(lt,t)) for t in sorted({max(1,right['turns']*i//4) for i in range(1,5)})],
            baseline_executed=len(lt),candidate_executed=len(rt),
            same_host_pair_id=paired_host,
            identical_world_prediction_steps=sum(a['law']==b['law'] for a,b in zip(lt,rt)),
            max_world_prediction_difference=max(abs(a['law'][z]-b['law'][z]) for a,b in zip(lt,rt) for z in a['law']),
            action_changes=sum(a['action']!=b['action'] for a,b in zip(lt,rt)),
            baseline_actions=dict(Counter(t['action'] for t in lt)),candidate_actions=dict(Counter(t['action'] for t in rt)),
            baseline_source=origins[key[:-1]+(16,)],candidate_source=origins[key]))
    groups=[]
    for level in ('game','world_seed','persona','condition'):
        bins=defaultdict(list)
        for row in comparisons:
            key=(row['game'],row['horizon'])
            if level=='world_seed':key+=(row['seed'],)
            if level in ('persona','condition'):key+=(row['profile'],)
            if level=='condition':key+=(row['regime'],)
            bins[key].append(row)
        for key,items in sorted(bins.items()):
            ds=[x['delta'] for x in items]
            groups.append(dict(level=level,key=key,conditions=len(items),world_seeds=sorted({x['seed'] for x in items}),
                mean_delta=sum(ds)/len(ds),mean_tick112_delta=sum(x['tick112_delta'] for x in items)/len(items),
                mean_preterminal_delta=sum(x['preterminal_delta'] for x in items)/len(items),
                mean_conditional_increment_delta=sum(x['conditional_increment_delta'] for x in items)/len(items),
                mean_realized_residual_delta=sum(x['realized_residual_delta'] for x in items)/len(items),
                better=sum(d>1e-12 for d in ds),same=sum(abs(d)<=1e-12 for d in ds),worse=sum(d< -1e-12 for d in ds),worst=min(ds)))
    return dict(complete=not missing and not incomplete,registered_runs=plan.get('total_development_runs',len(plan['new_jobs'])),
        collected_runs=len(rows),missing=missing,incomplete=incomplete,comparisons=comparisons,groups=groups,
        stage=plan['stage'],
        limitation='No cross-game strength mean. Shared-seed personas are not independent world samples. Partial summaries are not adoption evidence. Conditional increments retain earlier luck through the realized states; diagnostic accounts do not replace final purpose scores.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/long_horizon_development')
    p.add_argument('--additional-roots',nargs='*',default=[])
    p.add_argument('--plan',default='evidence/long_horizon/development_plan.json');p.add_argument('--partial',action='store_true')
    p.add_argument('--output',default='results/long_horizon_development/summary.json');a=p.parse_args()
    result=summarize(Path('.'),Path(a.root),json.loads(Path(a.plan).read_text()),a.partial,tuple(map(Path,a.additional_roots)))
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('missing','incomplete','comparisons','groups')}))
    print(json.dumps([x for x in result['groups'] if x['level']=='game'],indent=2))
