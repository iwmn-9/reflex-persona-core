"""Two real decisions per existing common bridge; connection, not strength evidence."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.contingent_games import BRIDGES
from reflex.laboratory import profiles
from tools.run_closed_loop import setup,decision
from tools.run_contingent_coverage import cases


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--frozen',action='store_true')
    args=ap.parse_args();root=Path(args.root).resolve()
    if not args.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,[])
    selected=cases('development');supported=[c for c in selected if c['game'] in BRIDGES]
    excluded=[c for c in selected if c['game'] not in BRIDGES]
    assert {c['game'] for c in supported}==set(BRIDGES)
    profile=next(p for p in profiles() if p['id']=='growth')
    pre=dict(format='closed-loop-connection-probe-v1',cases=supported,excluded=excluded,profile=profile,
        actual_seeds=[55100+i for i in range(len(supported))],model_seeds=[401,402],horizon=1,real_decisions=2,
        limitations=['one registered opening and persona per bridge','no strength or whole-game claim',
            'separate observed adapters and closed mahjong not connected by the common bridge',
            'learning, social and progress modules off; separate mechanism tests cover their composition'],
        source_snapshot_sha256=hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest())
    (root/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[];errors=[]
    with (root/'trajectories.jsonl').open('x',encoding='utf-8') as stream:
        for i,case in enumerate(supported):
            start=time.perf_counter()
            try:
                b,p,ctrl=setup(case,profile,'sampled_tree',1);w=b.start();actor=DecisionLoop(b.observe(b.project(w)),policy=p);trace=[]
                for used in range(2):
                    if b.terminal(w):break
                    before=b.record(w);c=b.observe(b.project(w),actor.state)
                    result,cap,remaining=decision(b,p,ctrl,w,actor,'sampled_tree',1,used)
                    key=result['decision']['action_id']
                    assert key in {a['id'] for a in c['actions'] if a['legal'] and not a['known_failure']}
                    assert actor.pending is None and cap.get('audit') is not None
                    assert DecisionLoop.from_record(c,actor.record(),policy=p).record()==actor.record()
                    w,flow=b.actual(w,key,55100+i)
                    trace.append(dict(before=before,after=b.record(w),root=key,flow=flow,
                        goal=b.goal(w).record(),tree=cap['audit'],remaining=remaining,owner=digest(actor.record())))
                assert trace
                check=b.start()
                for row in trace:
                    assert b.record(check)==row['before'];check,flow=b.actual(check,row['root'],55100+i)
                    assert b.record(check)==row['after'] and flow==row['flow'] and b.goal(check).record()==row['goal']
                r=dict(index=i,case=case,trace=trace,steps=len(trace),replay=True,elapsed=time.perf_counter()-start)
                stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush();rows.append({k:v for k,v in r.items() if k!='trace'})
                print(i+1,'/',len(supported),case['game'],len(trace),'decisions',round(r['elapsed'],2),flush=True)
            except Exception as error:
                errors.append(dict(index=i,case=case,error=repr(error)));print('ERROR',case['game'],repr(error),flush=True)
    result=dict(planned=len(supported),complete=len(rows),errors=errors,runs=rows,
        preregister_sha256=digest(pre),trajectories_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest())
    (root/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if errors:raise RuntimeError('connection failures retained')


if __name__=='__main__':main()
