"""Fixed paired long-play diagnosis of origin versus ordinary-relative gain."""
import argparse, copy, ctypes, json, os, platform, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from reflex.core import character_key,digest
from reflex.decision_loop import DecisionLoop
from reflex._continuation_portfolio import portfolio_planner
from reflex.laboratory import PROFILES
from reflex.model_observer import resolve_feedback
from tools.owned_checkpoint import OwnedCheckpoint,encoded,sha
from tools.run_gain_retry_screen import specs as previous_specs
from tools.run_continuation_portfolio import game_for as previous_game
from tools.background_preparation import BackgroundChain
from tools.run_owned_value import cumulative_basis


def specifications():
    return previous_specs()+[dict(id=f'background-{profile}-{kind}',mechanism='background',
        profile=profile,length=5,payout=payout,background=1.125,turns=128,changes=[])
        for profile in (0,2) for kind,payout in (('profitable',14.),('bad',5.))]


def game_for(spec):
    if spec['mechanism']!='background':return previous_game(spec)
    return BackgroundChain(spec['background'],spec['length'],spec['payout'],spec['turns'],PROFILES[spec['profile']])


def save(path,value):
    tmp=path.with_suffix('.new');tmp.write_bytes(encoded(value));tmp.replace(path)


def audit(folder,spec,reference):
    result=json.loads((folder/'result.json').read_bytes())
    envelope=json.loads((folder/'checkpoint.json').read_bytes());saved=envelope['payload']
    assert sha(encoded(saved))==envelope['sha256']
    assert result['complete'] and result['configuration']==saved['contract']['configuration']
    assert result['configuration']['spec']==spec and result['configuration']['retry_gain_reference']==reference
    assert result['trace']==saved['trace'] and len(result['trace'])==spec['turns']
    g=game_for(spec);world=g.start();actor=g.actor(g.context(world),learning='segmented')
    axes=character_key(g.context(world));rng=np.random.default_rng(0)
    for row in result['trace']:
        assert row['before']==world and row['before_owner']==digest(actor.record())
        context=g.context(world,actor.state)
        chosen=DecisionLoop.decide_batch([(actor,g.request(context))],False,_counterfactual_roots=(row['action'],))[0]
        rng.random();after,flow=g.transition(world,row['action'],'0')
        for tick,payout in spec['changes']:
            if after['tick']==tick:after['payout']=payout
        resolve_feedback(actor,chosen['ticket'],g.feedback(world,after,row['action'],context,flow))
        assert after==row['after'] and digest(actor.record())==row['owner']
        assert character_key(chosen['context'])==axes
        world=after
    assert g.terminal(world) and world==saved['world']==result['final']
    assert actor.record()==saved['owner'] and rng.bit_generator.state==saved['rng']
    assert all(result['checks'][k]==spec['turns'] for k in ('owner_parity','fixed_axes','restored'))
    assert result['score']==g.goal(world).value
    return dict(complete=True,score=result['score'],actual_steps=spec['turns'],restored=spec['turns'],
        owner_world_rng_persona_replayed=True,source_runtime=saved['contract']['runtime'],
        files={n:sha((folder/n).read_bytes()) for n in ('checkpoint.json','result.json')})


def execute(root,spec,reference,plan_sha):
    root=Path(root);plan=json.loads((root/'plan.json').read_bytes())
    assert sha((root/'plan.json').read_bytes())==plan_sha
    assert all(sha((ROOT/n).read_bytes())==h for n,h in plan['sources'].items())
    if os.name=='nt':
        k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
        k.SetPriorityClass.argtypes=(ctypes.c_void_p,ctypes.c_uint)
        assert k.SetPriorityClass(k.GetCurrentProcess(),0x4000)
    folder=root/spec['id']/reference;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return spec['id'],reference,audit(folder,spec,reference)
    horizon=spec.get('horizon',8)
    config=dict(spec=spec,retry=True,horizon=horizon,seeds=[701],retry_until_gain=True,
        retry_gain_reference=reference,plan_sha256=plan_sha)
    store=OwnedCheckpoint(folder/'checkpoint.json',('relative-gain',spec['id']),reference,configuration=config,every=1)
    g=game_for(spec);world=g.start();actor=g.actor(g.context(world),learning='segmented')
    axes=character_key(g.context(world));rng=np.random.default_rng(0)
    trace=[];checks=dict(fixed_axes=0,owner_parity=0,restored=0);elapsed=0.
    if store.saved is not None:
        saved=store.saved;world=saved['world'];trace=saved['trace'];checks=saved['checks'];elapsed=saved['elapsed']
        actor=DecisionLoop.from_record(g.context(trace[-1]['before']),saved['owner'],predictor=actor.predictor)
        rng.bit_generator.state=saved['rng'];assert actor.record()==saved['owner']
    started=time.monotonic()
    while not g.terminal(world):
        context=g.context(world,actor.state);request=g.request(context)
        before=actor.record();detached=copy.deepcopy(actor);cap={}
        planner=portfolio_planner(actor,request,world,observe=g.context,events=g.events,terminal=g.terminal,
            assess=lambda w:g.goal(w).record(),request_for=g.request,feedback=g.feedback,model_id=g.model_id,
            target=g.target,horizon=horizon,seeds=(701,),world_model='posterior',
            cumulative_preference=cumulative_basis(g,True),include_retry=True,captured=cap,
            purpose_estimator='conditional',tail_recovery=2,coherent_cumulative_mean=True,
            tail_purpose_reflex=True,retry_until_gain=True,retry_gain_reference=reference)
        result=DecisionLoop.decide_batch([(actor,request)],False,planner)[0]
        key=result['decision']['action_id'];rng.random();after,flow=g.transition(world,key,'0')
        for tick,payout in spec['changes']:
            if after['tick']==tick:after['payout']=payout
        observed=g.feedback(world,after,key,context,flow)
        resolve_feedback(actor,result['ticket'],observed)
        forced=DecisionLoop.decide_batch([(detached,request)],False,_counterfactual_roots=(key,))[0]
        resolve_feedback(detached,forced['ticket'],observed)
        assert actor.record()==detached.record() and character_key(result['context'])==axes
        record=json.loads(encoded(actor.record()));actor=DecisionLoop.from_record(context,record,predictor=actor.predictor)
        assert actor.record()==record
        for name in checks:checks[name]+=1
        selection=result['deliberation'];forecast=cap['forecast']
        proposal=forecast.audit['plans'][selection['selected_plan']]['proposal']
        trace.append(dict(tick=world['tick'],before=world,after=after,action=key,
            before_owner=digest(before),owner=digest(record),truth={'0':1.},response='0',
            selected_policy=proposal,selected_purpose=selection['selected_purpose'],
            predicted_actions=cap['modeled'][proposal][0]['actions'],simulated_steps=cap['simulated_steps']))
        world=after
        store.save(world,actor,rng,trace,checks,elapsed+time.monotonic()-started,terminal=g.terminal(world))
        save(folder/'progress.json',dict(pid=os.getpid(),turn=len(trace),case=spec['id'],reference=reference,complete=False))
        if len(trace)%32==0:print(json.dumps(dict(case=spec['id'],reference=reference,turn=len(trace))),flush=True)
    save(folder/'result.json',dict(configuration=config,complete=True,final=world,trace=trace,checks=checks,
        score=g.goal(world).value,seconds=elapsed+time.monotonic()-started))
    proof=audit(folder,spec,reference);save(folder/'audit.json',proof)
    save(folder/'progress.json',dict(pid=os.getpid(),turn=len(trace),case=spec['id'],reference=reference,complete=True))
    assert all(sha((ROOT/n).read_bytes())==h for n,h in plan['sources'].items())
    return spec['id'],reference,proof


def main(root,workers):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    handle=(root/'coordinator.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for directory in ('reflex','tools','tests')
        for p in (ROOT/directory).rglob('*.py')}
    plan=dict(version=1,specs=specifications(),references=['origin','ordinary'],sources=sources,
        runtime=dict(python=sys.version,numpy=np.__version__,system=platform.system(),machine=platform.machine()),
        arithmetic={k:os.environ.get(k) for k in ('NPY_DISABLE_CPU_FEATURES','OPENBLAS_CORETYPE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        adoption=False,criteria='All14 paired conditions,28 executions,3584 actual steps; every owner/world/RNG/persona and restoration audited. No negative paired score, strict improvement on both profitable background conditions. All losses retained. No world-cohort or independent adoption from this mechanism screen.')
    if (root/'plan.json').exists():assert json.loads((root/'plan.json').read_bytes())==plan
    else:save(root/'plan.json',plan)
    plan_sha=sha((root/'plan.json').read_bytes());results={};failures=[]
    save(root/'status.json',dict(complete=False,state='running',pid=os.getpid(),executions_complete=0,expected=28,failures=[],adoption=False))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(execute,str(root),spec,reference,plan_sha):(spec['id'],reference)
            for spec in plan['specs'] for reference in plan['references']}
        for future in as_completed(futures):
            try:
                case,reference,proof=future.result();results[(case,reference)]=proof
                print(json.dumps(dict(case=case,reference=reference,complete=True,score=proof['score'])),flush=True)
            except Exception as error:
                failures.append(dict(job=futures[future],error=repr(error)))
            save(root/'status.json',dict(complete=False,state='running',pid=os.getpid(),executions_complete=len(results),expected=28,failures=failures,adoption=False))
    if failures:
        save(root/'status.json',dict(complete=False,state='failed',pid=os.getpid(),executions_complete=len(results),expected=28,failures=failures,adoption=False))
        raise RuntimeError('registered executions failed; preserve results and inspect causes')
    pairs=[dict(id=s['id'],delta=results[(s['id'],'ordinary')]['score']-results[(s['id'],'origin')]['score'],
        arms={r:results[(s['id'],r)] for r in plan['references']}) for s in plan['specs']]
    assert sum(a['actual_steps'] for row in pairs for a in row['arms'].values())==3584
    passed=all(p['delta']>=-1e-12 for p in pairs) and all(p['delta']>1e-12 for p in pairs if p['id'].startswith('background-') and p['id'].endswith('-profitable'))
    save(root/'summary.json',dict(complete=True,gate_passed=passed,pairs=pairs,actual_steps=3584,plan_sha256=plan_sha,adoption=False))
    save(root/'status.json',dict(complete=True,state='complete',pid=os.getpid(),executions_complete=28,expected=28,failures=[],adoption=False,gate_passed=passed))
    handle.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if not 1<=a.workers<=4:p.error('one to four local workers required')
    main(a.root,a.workers)
