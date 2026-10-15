"""Check that final source reproduces a frozen controller trajectory exactly."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.laboratory import PROFILES
from tools.run_prediction_recovery import run


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--game',required=True)
    p.add_argument('--profile',required=True);p.add_argument('--regime',required=True);p.add_argument('--arm',default='recovery')
    p.add_argument('--seed',type=int,default=43);a=p.parse_args();root=Path(a.root)
    rows=[json.loads(x) for x in (root/'trajectories.jsonl').read_text().splitlines()]
    expected=next(r for r in rows if (r['game'],r['profile'],r['regime'],r['seed'],r['arm'])==(a.game,a.profile,a.regime,a.seed,a.arm))
    pid=next(i for i,p in enumerate(PROFILES) if p['id']==a.profile)
    actual=run((a.game,pid,a.seed,a.regime,expected['turns'],expected['plan_every']),a.arm)
    assert actual==expected,'final-source trajectory drift'
    receipt=dict(game=a.game,profile=a.profile,regime=a.regime,seed=a.seed,arm=a.arm,turns=actual['turns'],exact_replay=True)
    (root/f'replay_{a.game}_{a.profile}_{a.regime}_{a.seed}_{a.arm}.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))
