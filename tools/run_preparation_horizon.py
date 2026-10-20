"""Controlled long-play diagnostic of future competence, separate from games.

Every pair lasts 128 actual turns. Only the future owner's horizon differs.
The optional public payout shock is revealed after turn 64, never to planning
callbacks beforehand. This isolates delayed preparation; it is not evidence
of robust opponent prediction, human believability or game-wide adoption.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest,character_key
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from reflex.receding_owner import RecedingOwner
from tools.preparation_world import DelayedYield
from tools.run_owned_value import cumulative_basis
from tools.owned_checkpoint import OwnedCheckpoint,encoded,sha


def execution(root,delay,shock,future_horizon,*,turns=128,outer_horizon=16,max_decisions=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    cfg=dict(delay=delay,shock=shock,future_horizon=future_horizon,outer_horizon=outer_horizon,
             adapter_sha256=hashlib.sha256(Path(__file__).with_name('preparation_world.py').read_bytes()).hexdigest(),
             driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    checkpoint=OwnedCheckpoint(root/'checkpoint.json',('delayed-yield',0,0,shock,turns,1,outer_horizon-1,0),
        'receding-recovery-diagnostic',configuration=cfg,every=1,max_decisions=max_decisions)
    g=DelayedYield(delay,turns=turns);w=g.start();own=g.actor(g.context(w),learning='segmented')
    trace=[];rng=np.random.default_rng(0);checks=Counter();started=time.monotonic()
    if checkpoint.saved is not None:
        saved=checkpoint.saved;w=saved['world'];trace=saved['trace'];checks=Counter(saved['checks'])
        own=DecisionLoop.from_record(g.context(trace[-1]['before']),saved['owner'],predictor=own.predictor)
        rng.bit_generator.state=saved['rng'];started-=saved['elapsed']
    fixed=character_key(g.context(g.start()))
    while not g.terminal(w):
        c=g.context(w);req=g.request(c);saved=copy.deepcopy(own);before=digest(own.record())
        controller=RecedingOwner(own,observe=g.context,events=g.events,terminal=g.terminal,
            assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
            feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=outer_horizon-1,
            seeds=(701,),world_model='posterior',purpose_estimator='conditional',
            tail_recovery=2,tail_recovery_mode='always',tail_recovery_limit=outer_horizon-1,
            tail_recovery_horizon=future_horizon,cumulative_preference=cumulative_basis(g,True))
        result=controller.decide(w);key=result['decision']['action_id']
        assert character_key(result['context'])==fixed
        after,flow=g.transition(w,key,'0')
        # A factual external change to a public variable, absent from forecasts.
        if shock and after['tick']==turns//2:after['payout']=2.
        fb=g.feedback(w,after,key,c,flow)
        controller.resolve(result['ticket'],fb)
        forced=DecisionLoop.decide_batch([(saved,req)],False,_counterfactual_roots=(key,))[0]
        resolve_feedback(saved,forced['ticket'],fb)
        assert digest(saved.record())==digest(own.record())
        trace.append(dict(tick=w['tick'],before=w,action=key,after=after,
            before_owner=before,owner=digest(own.record()),goal=g.goal(after).value,
            selected_purpose=result['deliberation']['selected_purpose']))
        checks['fixed_axes']+=1;checks['owner_parity']+=1
        w=after
        checkpoint.save(w,own,rng,trace,checks,time.monotonic()-started,terminal=g.terminal(w))
        if len(trace)%16==0:
            print(json.dumps(dict(delay=delay,shock=shock,future_horizon=future_horizon,turn=len(trace))),flush=True)
        if checkpoint.stopping(len(trace)) and not g.terminal(w):return None
    # Replay the entire public world; no diagnostic simulation changes facts.
    state=g.start()
    for t in trace:
        assert t['before']==state
        state,_=g.transition(state,t['action'],'0')
        if shock and state['tick']==turns//2:state['payout']=2.
        assert state==t['after']
    assert state==w and g.terminal(state) and len(trace)==turns
    out=dict(configuration=cfg,turns=turns,executed=len(trace),score=g.goal(w).value,
        final=w,seconds=time.monotonic()-started,checks=dict(checks),physical_replay=len(trace),
        preparation_ticks=[t['tick'] for t in trace if t['action']=='y'],trace=trace)
    (root/'result.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    return out


def pair(root,spec):
    rows=[]
    for h in (2,16):
        path=Path(root)/f"delay-{spec['delay']}-shock-{int(spec['shock'])}"/str(h)
        result=path/'result.json'
        row=json.loads(result.read_bytes()) if result.exists() else execution(path,**spec,future_horizon=h)
        assert row['executed']==128 and row['configuration']['future_horizon']==h
        envelope=json.loads((path/'checkpoint.json').read_bytes());saved=envelope['payload']
        assert sha(encoded(saved))==envelope['sha256']
        assert saved['trace']==row['trace'] and saved['world']==row['final']
        assert saved['contract']['configuration']==row['configuration']
        assert all(row['configuration'][k]==v for k,v in spec.items())
        assert row['checks']==dict(fixed_axes=128,owner_parity=128) and row['physical_replay']==128
        rows.append(row)
    result=dict(**spec,baseline=rows[0]['score'],candidate=rows[1]['score'],
        delta=rows[1]['score']-rows[0]['score'],
        preparation_ticks=[r['preparation_ticks'] for r in rows],seconds=[r['seconds'] for r in rows])
    print(json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true')
    a=p.parse_args();root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        dispatch(root,Path(__file__).name,[]);return
    plan=dict(cases=[dict(delay=d,shock=s) for d in (3,7,11) for s in (False,True)],
        turns=128,outer_horizon=16,future_horizons=[2,16],outer_seeds=[701],inner_seeds=[1701,1702,1703],
        interpretation=__doc__,acceptance='All six pairs complete; no adverse case hidden; this mechanism-only diagnostic cannot establish adoption.')
    reg=root/'registration.json'
    if reg.exists():assert json.loads(reg.read_bytes())==plan
    else:reg.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8')
    manifest=json.loads((root/'source_snapshot.json').read_bytes())
    for file,sha in manifest.items():assert hashlib.sha256((root/'_source'/file).read_bytes()).hexdigest()==sha
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(pair,root,spec) for spec in plan['cases']]):rows.append(future.result())
    blind_prefixes=0
    for delay in (3,7,11):
        for horizon in (2,16):
            runs=[json.loads((root/f'delay-{delay}-shock-{s}'/str(horizon)/'result.json').read_bytes()) for s in (0,1)]
            for a,b in zip(runs[0]['trace'][:64],runs[1]['trace'][:64]):
                assert all(a[k]==b[k] for k in ('before','action','before_owner')),'unrevealed future shock affected a decision'
            blind_prefixes+=1
    (root/'summary.json').write_text(json.dumps(dict(complete=True,comparisons=rows,
        identical_pre_shock_prefixes=blind_prefixes),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
