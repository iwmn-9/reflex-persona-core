"""Common bridge coverage for the new endpoint hook; not strength evidence."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.contingent_games import BRIDGES,make_bridge
from reflex.frontier_value import PolicyTail
from reflex.laboratory import profiles
from reflex.integrated_trials import request_for
from tools.closed_loop_models import controller
from tools.run_contingent_coverage import cases
from tools.run_frontier_value import owned


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,[])
    selected=[c for c in cases('development') if c['game'] in BRIDGES]
    pre=dict(version='frontier-connection-probe-v1',cases=selected,excluded=[c for c in cases('development') if c['game'] not in BRIDGES],
        source_snapshot=digest(json.loads((root/'source_snapshot.json').read_text())),tree_horizon=1,tail_steps=1,tail_seeds=[701],real_decisions=2,
        scope='same hook across existing common bridges; no whole-game strength claim; unconnected separate observed adapters and closed mahjong remain excluded')
    (root/'preregister.json').write_text(json.dumps(pre,indent=2)+'\n');rows=[];errors=[]
    with (root/'trajectories.jsonl').open('x',encoding='utf-8') as stream:
        for i,case in enumerate(selected):
            start=time.perf_counter()
            try:
                b=make_bridge(profiles()[0],case);w=b.start();actor=owned(b,w);ctrl=controller(b,horizon=1,nodes=4096,outcomes=4096)
                ctrl.frontier=PolicyTail(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,assess=lambda w:b.goal(w).record(),
                    request_for=ctrl.request_for,feedback=ctrl.feedback,model_id=ctrl.model_id,target=b.target,seeds=(701,),steps=1)
                trace=[]
                for _ in range(2):
                    if b.terminal(w):break
                    c=b.observe(b.project(w),actor.state);req=request_for(c,case['game']);cap={}
                    before=b.record(w);d=DecisionLoop.decide_batch([(actor,req)],False,ctrl.planner(actor,req,b.project(w),captured=cap))[0]
                    key=d['decision']['action_id'];assert cap['audit']['complete']
                    assert key in {a['id'] for a in c['actions'] if a['legal'] and not a['known_failure']}
                    actor.abandon(d['ticket']);assert DecisionLoop.from_record(c,actor.record(),policy=actor.policy).record()==actor.record()
                    w,flow=b.actual(w,key,48000+i);trace.append(dict(before=before,root=key,after=b.record(w),flow=flow,owner=digest(actor.record()),tree=cap['audit']))
                check=b.start()
                for t in trace:
                    assert b.record(check)==t['before'];check,flow=b.actual(check,t['root'],48000+i);assert b.record(check)==t['after'] and flow==t['flow']
                r=dict(index=i,case=case,trace=trace,replay=True,elapsed=time.perf_counter()-start);rows.append(r)
                stream.write(json.dumps(r)+'\n');stream.flush();print(i+1,case['game'],round(r['elapsed'],2),flush=True)
            except Exception as e:errors.append(dict(case=case,error=repr(e)));print('ERROR',repr(e),flush=True)
    result=dict(planned=len(selected),complete=len(rows),errors=errors,decisions=sum(len(r['trace']) for r in rows))
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    if errors:raise RuntimeError('failed connections retained')


if __name__=='__main__':main()
