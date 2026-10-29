"""Long-play mechanism screen; never substitutes for the registered 40/8/8."""
import argparse
import copy
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reflex.core import character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from reflex.model_observer import resolve_feedback
from reflex.receding_owner import RecedingOwner
from tools.continuing_games import ContinuingGame
from tools.continuation_portfolio import portfolio_planner
from tools.owned_checkpoint import OwnedCheckpoint, encoded, sha
from tools.preparation_world import DelayedYield, PreparationChain
from tools.run_owned_value import cumulative_basis, law, sample
from tools.summarize_horizon_study import realized_account


def game_for(spec):
    if spec['mechanism']=='world':
        return ContinuingGame(spec['game'],PROFILES[spec['profile']],spec['turns'],spec['seed']%2)
    cls = DelayedYield if spec['mechanism'] == 'delayed' else PreparationChain
    return cls(spec['length'], payout=spec['payout'], turns=spec['turns'], profile=PROFILES[spec['profile']])


def execution(root, spec, retry, *, max_decisions=None, reader_selection='advice'):
    if reader_selection not in ('advice', 'current_model') or reader_selection == 'current_model' and not retry:
        raise ValueError('explicit reader selection on the candidate portfolio required')
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    hashes = {n: sha(Path(__file__).with_name(n).read_bytes()) for n in (
        'run_continuation_portfolio.py', 'continuation_portfolio.py', 'preparation_world.py','horizon_consensus.py')}
    model_seeds=(701,702,703) if spec['mechanism']=='world' else (701,)
    adopted=spec['mechanism']=='world' and not retry
    consensus=bool(spec.get('controller',{}).get('horizon_consensus',False)) and retry
    config = dict(spec=spec, retry=retry, sources=hashes, horizon=32 if consensus else 16, seeds=list(model_seeds),
        reference='adopted-owner-continuity-H16' if adopted else 'reflex-tail')
    if reader_selection != 'advice':
        config['reader_selection'] = reader_selection
    g = game_for(spec); w = g.start(); c = g.context(w)
    actor = g.actor(c, learning='segmented'); axes = character_key(c)
    if reader_selection == 'current_model':
        from tools.current_model_reader import CurrentModelReader
        actor.predictor = CurrentModelReader.from_reader(actor.predictor, c['scope'])
    store = OwnedCheckpoint(root/'checkpoint.json', ('continuation-portfolio', spec['id']),
        'portfolio' if retry else 'reflex-tail', configuration=config, every=1, max_decisions=max_decisions)
    trace=[]; checks=dict(owner_parity=0, fixed_axes=0, restored=0); rng=np.random.default_rng(spec.get('seed',0))
    start=time.monotonic()
    if store.saved is not None:
        saved=store.saved; w=saved['world']; trace=saved['trace']; checks=saved['checks']
        actor=DecisionLoop.from_record(g.context(trace[-1]['before']), saved['owner'], predictor=actor.predictor)
        rng.bit_generator.state=saved['rng']; start-=saved['elapsed']
    while not g.terminal(w):
        c=g.context(w, actor.state); req=g.request(c); cap={}; prior=actor.record()
        detached=copy.deepcopy(actor)
        if adopted:
            controller=RecedingOwner(actor,observe=g.context,events=g.events,terminal=g.terminal,
                assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
                feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=15,
                seeds=model_seeds,world_model='posterior',purpose_estimator='conditional',tail_recovery=2)
            result=controller.decide(w)
        else:
            options=dict(spec.get('controller',{}));options.pop('horizon_consensus',None)
            factory=portfolio_planner
            if consensus:
                from tools.horizon_consensus import consensus_planner
                factory=consensus_planner
            planner=factory(actor, req, w, observe=g.context, events=g.events, terminal=g.terminal,
                assess=lambda w:g.goal(w).record(), request_for=g.request, feedback=g.feedback,
                model_id=g.model_id, target=g.target, horizon=32 if consensus else 16, seeds=model_seeds,
                cumulative_preference=cumulative_basis(g, True), include_retry=retry, captured=cap,
                **options)
            result=DecisionLoop.decide_batch([(actor, req)], False, planner)[0]
        key=result['decision']['action_id']
        truth=law(w['tick'],spec['turns'],spec['regime']) if spec['mechanism']=='world' else {'0':1.}
        response=sample(truth,rng.random())
        after,flow=g.transition(w,key,response)
        conditional=sum(p*(g.goal(g.transition(w,key,z)[0]).value-g.goal(w).value) for z,p in truth.items())
        # The schedule is owned by the real world only. The controller receives
        # the changed payout after this event, never its time or future value.
        for tick,payout in spec.get('changes',[]):
            if after['tick']==tick:after['payout']=payout
        feedback=g.feedback(w,after,key,c,flow)
        resolve_feedback(actor,result['ticket'],feedback)
        replay=DecisionLoop.decide_batch([(detached,req)],False,_counterfactual_roots=(key,))[0]
        resolve_feedback(detached,replay['ticket'],feedback)
        assert actor.record()==detached.record()
        assert character_key(result['context'])==axes
        checks['owner_parity']+=1;checks['fixed_axes']+=1
        if w['tick']%23==0:
            record=json.loads(encoded(actor.record()))
            actor=DecisionLoop.from_record(c,record,predictor=actor.predictor)
            assert actor.record()==record;checks['restored']+=1
        diagnostic=result['deliberation']
        if adopted:
            proposal='adopted-H16';predicted=list(controller.last['node']['returns'][0][2])
            simulated_steps=None
        else:
            f=cap['forecast'];selected=diagnostic['selected_plan'];proposal=f.audit['plans'][selected]['proposal']
            predicted=cap['modeled'][proposal][0]['actions'];simulated_steps=cap['simulated_steps']
        trace.append(dict(tick=w['tick'],before=w,after=after,action=key,
            before_owner=digest(prior),owner=digest(actor.record()),
            selected_policy=proposal,selected_purpose=diagnostic['selected_purpose'],
            predicted_actions=predicted,simulated_steps=simulated_steps,
            response=response,truth=truth,conditional_increment=conditional))
        if consensus:trace[-1]['horizon_consensus']=cap['horizon_consensus']
        w=after
        store.save(w,actor,rng,trace,checks,time.monotonic()-start,terminal=g.terminal(w))
        if len(trace)%32==0:print(json.dumps(dict(id=spec['id'],retry=retry,turn=len(trace),cash=w['cash'])),flush=True)
        if store.stopping(len(trace)) and not g.terminal(w):return None
    replay=g.start();replay_rng=np.random.default_rng(spec.get('seed',0))
    for entry in trace:
        assert entry['before']==replay
        assert sample(entry['truth'],replay_rng.random())==entry['response']
        replay,_=g.transition(replay,entry['action'],entry['response'])
        for tick,payout in spec.get('changes',[]):
            if replay['tick']==tick:replay['payout']=payout
        assert entry['after']==replay
    assert replay==w and g.terminal(w) and len(trace)<=spec['turns']
    assert replay_rng.bit_generator.state==rng.bit_generator.state
    row=dict(configuration=config,complete=True,final=w,score=g.goal(w).value,
        trace=trace,checks=checks,seconds=time.monotonic()-start,physical_replay=len(trace))
    row['runtime']=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform())
    (root/'result.json').write_bytes(encoded(row))
    return row


def pair(root,spec):
    rows=[]
    for retry in (False,True):
        if not retry and 'reference' in spec:
            data=(Path(root)/spec['reference']['archive']).read_bytes()
            assert sha(data)==spec['reference']['sha256']
            rows.append(json.loads(data))
            continue
        directory=Path(root)/spec['id']/('candidate' if retry else 'baseline')
        dest=directory/'result.json'
        row=json.loads(dest.read_bytes()) if dest.exists() else execution(directory,spec,retry)
        envelope=json.loads((directory/'checkpoint.json').read_bytes());saved=envelope['payload']
        assert sha(encoded(saved))==envelope['sha256']
        assert saved['trace']==row['trace'] and saved['world']==row['final']
        assert saved['contract']['configuration']==row['configuration']
        assert row['configuration']['spec']==spec and row['complete']
        assert row['physical_replay']==len(row['trace'])<=spec['turns'] and game_for(spec).terminal(row['final'])
        rows.append(row)
    if 'reference' in spec:
        assert all(a['response']==b['response'] and a['truth']==b['truth']
            for a,b in zip(rows[0]['trace'],rows[1]['trace']))
        assert rows[0]['trace'][0]['before']==rows[1]['trace'][0]['before']
    else:
        assert rows[0]['runtime']==rows[1]['runtime'],'fresh controls require the same execution environment'
    conditional=lambda row: (sum(t['conditional_increment'] for t in row['trace'])
        if 'conditional_increment' in row['trace'][0] else realized_account(row)['conditional_increments'])
    result=dict(id=spec['id'],spec=spec,cash=[r['final']['cash'] for r in rows],
        delta=rows[1]['score']-rows[0]['score'],cash_delta=rows[1]['final']['cash']-rows[0]['final']['cash'],
        scores=[r['score'] for r in rows],executed=[len(r['trace']) for r in rows],
        conditional_delta=conditional(rows[1])-conditional(rows[0]),
        seconds=[r['seconds'] for r in rows],
        actions=[{k:sum(t['action']==k for t in r['trace']) for k in sorted({t['action'] for t in r['trace']})} for r in rows])
    print(json.dumps(dict(completed_pair=result)),flush=True)
    return result


def prepare_references(root,summary_path):
    """Reuse audited controls exactly; these become known-case diagnostics."""
    from tools.summarize_value_contract import read_run
    base=Path(__file__).resolve().parents[1]
    summary=json.loads(Path(summary_path).read_bytes())
    plan_path=base/'evidence/long_horizon/sampling_check_confirmation_plan.json'
    plan=json.loads(plan_path.read_bytes());plan_hash=sha(plan_path.read_bytes())
    assert summary['complete'] and summary['completed_conditions']==40 and summary['plan_sha256']==plan_hash
    chosen=[r for r in summary['comparisons'] if r['seed']==56101 and r['profile'] in ('growth','care')]
    assert len(chosen)==8 and {r['game'] for r in chosen}=={'auction','combat','resources','gambling'}
    root.mkdir(parents=True,exist_ok=True);(root/'references').mkdir(exist_ok=True)
    specs=[]
    for row in chosen:
        registered=next(c for c in plan['cases'] if c['key']==row['condition'])
        source=base/row['baseline']['path']
        control,proof,receipt=read_run(source.parent,registered['job'],False,plan,plan_hash)
        assert receipt['sha256']==row['baseline']['sha256']
        assert control['tail_steps']==15 and control['arm']=='receding_recovery'
        assert not any(control.get(k,False) for k in ('cumulative_achievement','tail_goal_refinement','sampling_check','tail_replanning'))
        archive=f"references/{row['condition']}.json"
        data=source.read_bytes();dest=root/archive
        if dest.exists():assert dest.read_bytes()==data
        else:dest.write_bytes(data)
        specs.append(dict(id=row['condition'],mechanism='world',game=row['game'],profile=registered['job'][1],
            turns=128,seed=row['seed'],regime=row['regime'],reference=dict(archive=archive,
                sha256=sha(data),original_path=row['baseline']['path'],audit=receipt,
                original_plan_sha256=plan_hash,pair_id=proof['pair_id'])))
    specs.sort(key=lambda s:s['id'])
    path=root/'reference_specs.json'
    if path.exists():assert json.loads(path.read_bytes())==specs
    else:path.write_bytes(encoded(specs))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--frozen',action='store_true');parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--worlds',action='store_true')
    parser.add_argument('--references-summary')
    parser.add_argument('--purpose-estimator',choices=('sampled','conditional'),default='sampled')
    parser.add_argument('--tail-recovery',type=int,choices=(0,2),default=0)
    parser.add_argument('--recovery-cache',type=int,default=0)
    parser.add_argument('--coherent-cumulative-mean',action='store_true')
    parser.add_argument('--fresh-controls',action='store_true')
    parser.add_argument('--horizon-consensus',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve()
    if not args.frozen:
        from tools.freeze_experiment import dispatch
        if args.references_summary:
            if not args.worlds:raise ValueError('audited references require the world diagnostic')
            prepare_references(root,args.references_summary)
        return dispatch(root,Path(__file__).name,['--workers',str(args.workers),
            '--purpose-estimator',args.purpose_estimator,'--tail-recovery',str(args.tail_recovery)]+
            ['--recovery-cache',str(args.recovery_cache)]+
            (['--coherent-cumulative-mean'] if args.coherent_cumulative_mean else [])+
            (['--fresh-controls'] if args.fresh_controls else [])+
            (['--horizon-consensus'] if args.horizon_consensus else [])+
            (['--worlds'] if args.worlds else []))
    specs=[]
    for mechanism,length,good in (('delayed',11,6.),('chain',5,14.)):
        for profile in (0,2):
            for kind,payout,changes in (('stable',good,[]),('loss',good,[[73,2.]]),('opportunity',2.,[[29,good]])):
                specs.append(dict(id=f'{mechanism}-{profile}-{kind}',mechanism=mechanism,length=length,
                    payout=payout,changes=changes,profile=profile,turns=128))
    if args.worlds:
        references=root/'reference_specs.json'
        specs=json.loads(references.read_bytes()) if references.exists() else [
            dict(id=f'{game}-{profile}',mechanism='world',game=game,profile=profile,turns=128,
                seed=58101 if profile==0 else 58102,regime='irregular' if profile==0 else 'outside')
                for game in ('auction','resources','combat','gambling') for profile in (0,2)]
        if args.fresh_controls:specs=[{k:v for k,v in s.items() if k!='reference'} for s in specs]
    if args.purpose_estimator!='sampled' or args.tail_recovery or args.recovery_cache:
        specs=[dict(s,controller=dict(purpose_estimator=args.purpose_estimator,
            tail_recovery=args.tail_recovery,recovery_cache=args.recovery_cache)) for s in specs]
    if args.coherent_cumulative_mean:
        if args.purpose_estimator!='conditional':raise ValueError('coherent cumulative mean requires conditional mean')
        specs=[dict(s,controller=dict(s['controller'],coherent_cumulative_mean=True)) for s in specs]
    if args.horizon_consensus:
        if not args.coherent_cumulative_mean:raise ValueError('coherent cumulative mean required')
        specs=[dict(s,controller=dict(s['controller'],horizon_consensus=True)) for s in specs]
    plan=dict(version=2,specs=specs,source=sha(encoded(json.loads((root/'source_snapshot.json').read_bytes()))),
        purpose='mechanism screen only; no adoption and no replacement of registered 40/8/8',
        criteria='all cases complete; fixed personality and actual feedback parity; report every loss; preparation gain and abandonment of unprofitable work; new opportunity inside a former fixed window' if not args.worlds else
            'All 8 pairs complete or actual death; no excluded losses. Same personality, factual learning and world RNG. Compare adopted H16 against candidate; each family mean score and conditional increment must not decrease, but this development screen cannot establish adoption.')
    plan_path=root/'plan.json'
    if plan_path.exists():assert json.loads(plan_path.read_bytes())==plan
    else:plan_path.write_bytes(encoded(plan))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(pair,root,s) for s in specs]
        for future in as_completed(futures):
            rows.append(future.result())
            (root/'summary.json').write_bytes(encoded(dict(complete=len(rows)==len(specs),
                completed_pairs=len(rows),registered_pairs=len(specs),rows=sorted(rows,key=lambda r:r['id']))))


if __name__=='__main__':main()
