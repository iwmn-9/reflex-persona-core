"""One registered, checkpointed horizon comparison; rules and owner are unchanged."""
import argparse,json,platform,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--job',required=True);p.add_argument('--stage',default='development')
    p.add_argument('--recovery-limit',type=int,default=16)
    p.add_argument('--model-seeds',type=int,nargs='+',default=[701,702,703])
    p.add_argument('--sampling-standard-errors',type=float)
    p.add_argument('--cumulative-achievement',action='store_true')
    p.add_argument('--tail-goal-refinement',action='store_true')
    p.add_argument('--frozen',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve();job=json.loads(a.job)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        args=['--job',a.job,'--stage',a.stage,'--recovery-limit',str(a.recovery_limit),'--model-seeds',*map(str,a.model_seeds)]
        if a.sampling_standard_errors is not None:args+=['--sampling-standard-errors',str(a.sampling_standard_errors)]
        if a.cumulative_achievement:args+=['--cumulative-achievement']
        if a.tail_goal_refinement:args+=['--tail-goal-refinement']
        dispatch(root,Path(__file__).name,args);return
    from tools.run_owned_value import run,law
    from tools.continuing_games import GAMES
    from reflex.laboratory import PROFILES
    from tools.audit_owned_value import audit
    assert isinstance(job,list) and len(job)==8
    g,pid,seed,regime,turns,interval,tail,offset=job
    assert g in GAMES and type(pid) is int and 0<=pid<len(PROFILES)
    assert type(seed) is int and type(turns) is int and turns>0
    assert interval==1 and offset==0 and type(tail) is int and 1<=tail<=127
    law(0,turns,regime)
    registration=dict(jobs=[job],arms=['receding_recovery'],world_seeds=[seed],stage=a.stage,semantics=__doc__)
    assert 1<=a.recovery_limit<=256
    assert 1<=len(a.model_seeds)<=32 and len(set(a.model_seeds))==len(a.model_seeds)
    configuration={}
    if a.recovery_limit!=16:configuration['tail_recovery_limit']=a.recovery_limit
    if a.model_seeds!=[701,702,703]:configuration['model_seeds']=a.model_seeds
    if a.sampling_standard_errors is not None:configuration['sampling_standard_errors']=a.sampling_standard_errors
    if a.cumulative_achievement:configuration['cumulative_achievement']=True
    if a.tail_goal_refinement:configuration['tail_goal_refinement']=True
    if configuration:registration['configuration']=configuration
    reg=root/'registration.json'
    if reg.exists():assert json.loads(reg.read_text())==registration,'cannot change a resumed study'
    else:reg.write_text(json.dumps(registration,indent=2)+'\n',encoding='utf8')
    environment=dict(python=sys.version,numpy=np.__version__,system=platform.system(),release=platform.release(),machine=platform.machine())
    env=root/'environment.json'
    if not env.exists():env.write_text(json.dumps(environment,indent=2)+'\n',encoding='utf8')
    output=root/'trajectories.jsonl'
    if output.exists():raise FileExistsError('completed study must not be repeated')
    result=run(job,'receding_recovery',checkpoint=root/'checkpoint.json',checkpoint_every=4,tail_recovery_limit=a.recovery_limit,model_seeds=tuple(a.model_seeds),sampling_standard_errors=a.sampling_standard_errors,cumulative_achievement=a.cumulative_achievement,tail_goal_refinement=a.tail_goal_refinement)
    assert result is not None
    temporary=output.with_suffix('.new')
    temporary.write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf8');temporary.replace(output)
    (root/'summary.json').write_text(json.dumps([{k:v for k,v in result.items() if k!='trace'}],indent=2)+'\n',encoding='utf8')
    audit(root)
    print(json.dumps(dict(complete=True,job=job,score=result['score'])),flush=True)


if __name__=='__main__':main()
