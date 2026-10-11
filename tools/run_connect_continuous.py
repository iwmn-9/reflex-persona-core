"""Fresh actual games: an improvement at every owner turn, not only one root."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_connect_policy_rollout import worker as pair
from tools.run_connect_rival_prior import run


def worker(job):
    seed,profile,samples,rival=job
    a=pair((seed,profile,samples,'incumbent',rival,'mixture','single'))
    b=pair((seed,profile,samples,'incumbent',rival,'mixture','continuous'))
    assert a[0]==b[0]
    for row,label in zip((a[0],a[1],b[1]),('baseline','single','continuous')):
        row.update(condition=label,rival=rival)
    return [a[0],a[1],b[1]]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--start',type=int,required=True)
    p.add_argument('--seeds',type=int,default=16);p.add_argument('--samples',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--frozen',action='store_true');a=p.parse_args()
    if a.frozen:
        run(a.root,a.start,a.seeds,a.samples,a.workers,('baseline','single','continuous'),worker,dict(
            version='public-continuous-policy-improvement-v1',
            primary='continuous minus baseline own terminal credit separately for actual minimax2 and minimax4; all profile losses disclosed',
            secondary='continuous minus single intervention using same unlearned strength prior and same 8 terminal trials',
            model='future owner base SHORT re-searches; actual continuous owner reapplies root improvement; future upgraded owner is NOT predicted exactly',
            origin='registered before completed rival-prior outcomes; operational question is whether repeated improvement remains useful',
            limits=['known public opponent families, 16 board clusters','uniform rival prior, no online type inference',
                    'actual continuous owner differs from predicted base owner; no exact self-consistency guarantee','no new efficacy in other genres or human-level claim']))
    else:
        from tools.freeze_experiment import dispatch
        dispatch(a.root,Path(__file__).name,['--start',str(a.start),'--seeds',str(a.seeds),'--samples',str(a.samples),'--workers',str(a.workers)])
