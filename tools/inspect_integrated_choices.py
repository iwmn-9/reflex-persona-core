"""Replay one frozen game job and expose all candidate forecasts at one tick."""
import argparse,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--job',required=True);p.add_argument('--tick',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(a.source).resolve()))
    import reflex.integrated_trials as trials
    from reflex.core import compile_batch
    from reflex.deliberation import select
    job=json.loads(Path(a.job).read_text(encoding='utf-8'));original=trials.goal_forecast;captured=[]
    def recording(c,paths,audit,**kw):
        f=original(c,paths,audit,**kw)
        if c['tick']==a.tick:
            b=compile_batch(f.contexts);frame=getattr(f,'return_frame',None)
            d=kw['policy'].decide(b,False,**({} if frame is None else {'return_frame':frame}))
            decision,selection=select([c],f,kw['policy'])
            captured.append(dict(context=c,forecast_contexts=f.contexts,purpose=f.purpose,roots=f.roots,
                return_frame=None if frame is None else frame.record(),scores=d.scores.tolist(),selection=selection,
                selected=decision[0]['action_id'],forecast_audit=f.audit,all_modeled=trials.compact(audit)))
        return f
    trials.goal_forecast=recording
    r=trials.play(job['case'],job['profile'],job['seed'],job['arm'],horizon=job['horizon'])
    assert len(captured)==1
    out=dict(job=job,selected_tick=a.tick,captured=captured[0],goal=r['goal'],actions=r['actions'],action_replay_verified=r['action_replay_verified'])
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(arm=job['arm'],goal=r['goal'],selected=captured[0]['selected'])))
    return 0

if __name__=='__main__':raise SystemExit(main())
