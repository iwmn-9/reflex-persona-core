"""Offline same-depth reference. No true law or result is sent to an NPC."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys


def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();folder=Path(__file__).resolve().parent;source=Path(args.source).resolve()
    plan=json.loads((folder/'sampling_plan.json').read_bytes())
    review=json.loads(gzip.decompress((folder/'review.json.gz').read_bytes()))
    manifest=json.loads((folder/'sampling_source_snapshot.json').read_bytes())
    sha=lambda raw:hashlib.sha256(raw).hexdigest()
    assert sha(encoded(manifest))==plan['source']==review['source']
    for name,h in manifest.items():assert sha((source/name).read_bytes())==h
    sys.path.insert(0,str(source))
    from tools.diagnose_auction_continuity import reference
    from tools.run_continuation_portfolio import game_for
    from tools.run_owned_value import law
    rows=[];checks=0;max_error=0.
    for case in plan['cases']:
        pair=next(r for r in review['pairs'] if r['spec']['id']==case['original_case_id'])
        spec=pair['spec'];step=pair['arms']['candidate']['steps'][case['tick']]
        game=game_for(spec);turns=spec['turns'];tick=case['tick'];cash=int(step['cash'])
        distributions=dict(truth=lambda t:tuple(law(t,turns,spec['regime']).values()),
            frozen_belief=lambda t,p=tuple(step['belief'][str(z)] for z in range(3)):p)
        out=[]
        for end in sorted({min(tick+32,turns),turns}):
            for kind,distribution in distributions.items():
                # The existing reference normalizes by its supplied endpoint.
                # Rescale to the original game's fixed objective unit; do not
                # let a shorter diagnostic horizon change the objective.
                base=reference(end,spec['seed']%2,distribution)
                q=lambda t,c:{k:v*end/turns for k,v in base(t,c).items()}
                values=q(tick,cash)
                for t in range(tick,end):
                    for amount in range(13):
                        world=copy.deepcopy(game.start());world.update(tick=t,cash=float(amount))
                        for action,value in q(t,amount).items():
                            expected=0.
                            for symbol,probability in enumerate(distribution(t)):
                                after,_=game.transition(world,action,str(symbol))
                                future=(0. if t+1==end else max(q(t+1,int(after['cash'])).values())-.25*after['cash']/(8*turns))
                                expected+=probability*(game.goal(after).value-game.goal(world).value+future)
                            error=abs(value-.25*amount/(8*turns)-expected)
                            assert error<1e-12,(case['id'],kind,t,amount,action,error)
                            checks+=1;max_error=max(max_error,error)
                out.append(dict(law=kind,horizon=end-tick,endpoint=end,values=values,
                    best=max(values,key=values.get),actual_action=step['action'],
                    actual_opportunity_loss=max(values.values())-values[step['action']]))
        rows.append(dict(id=case['id'],comparisons=out))
    result=dict(complete=True,adoption=False,source=plan['source'],cases=rows,
        script_sha256=sha(Path(__file__).read_bytes()),bellman_equations_checked=checks,max_error=max_error,
        limit='Evaluator-only purpose optimum; ignores personality and future learning. Current belief is held stationary. Neither reference is the playing model or a new independent comparison.')
    Path(args.output).write_bytes(encoded(result))
    print(json.dumps(dict(checks=checks,max_error=max_error,choices=[dict(id=r['id'],choices=[(v['law'],v['horizon'],v['best']) for v in r['comparisons']]) for r in rows])),flush=True)


if __name__=='__main__':main()
