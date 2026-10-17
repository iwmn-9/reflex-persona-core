"""Post-hoc conditional-purpose increments along actual continuing paths.

This is a martingale decomposition, not an oracle policy or a counterfactual
estimate of which policy is best. The true response law is diagnostic only.
The visited states already depend on earlier random outcomes and decisions.
"""
import argparse, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from reflex.laboratory import PROFILES
from tools.compare_owned_studies import compare
from tools.continuing_games import ContinuingGame
from tools.run_owned_value import law


def review(studies, baselines, candidate):
    validated=compare(studies,[]);runs=[];groups=defaultdict(dict)
    rule_hash=hashlib.sha256((R/'tools/continuing_games.py').read_bytes()).hexdigest()
    for study in map(Path,studies):
        manifest=json.loads((study/'source_snapshot.json').read_text())
        assert manifest['tools/continuing_games.py']==rule_hash
        for row in map(json.loads,(study/'trajectories.jsonl').read_text().splitlines()):
            game=ContinuingGame(row['game'],next(p for p in PROFILES if p['id']==row['profile']),row['turns'],row['seed']%2)
            increments=[]
            for t in row['trace']:
                before=t['before'];initial=game.goal(before).value
                p=law(t['tick'],row['turns'],row['regime']);assert p==t['truth']
                expected=math.fsum(prob*(game.goal(game.transition(before,t['action'],symbol)[0]).value-initial)
                                   for symbol,prob in p.items())
                actual=t['goal']-initial
                increments.append(dict(tick=t['tick'],conditional_gain=expected,
                                       innovation=actual-expected,realized_gain=actual))
            initial=game.goal(row['trace'][0]['before']).value
            expected=math.fsum(t['conditional_gain'] for t in increments)
            noise=math.fsum(t['innovation'] for t in increments)
            error=abs(initial+expected+noise-row['score'])
            assert error<1e-12,('telescoping accounting failed',error)
            record={k:row[k] for k in ('game','profile','seed','regime','turns','tail_steps','offset')}
            record.update(arm=row['arm']+'@'+str(row['interval']),score=row['score'],initial_score=initial,
                          conditional_gain=expected,innovation=noise,accounting_error=error,increments=increments)
            runs.append(record)
            key=tuple(record[k] for k in ('game','profile','seed','regime','turns','tail_steps','offset'))
            assert record['arm'] not in groups[key];groups[key][record['arm']]=record
    pairs=[]
    for key,arms in sorted(groups.items()):
        if candidate not in arms:continue
        new=arms[candidate]
        for baseline in baselines:
            if baseline not in arms:continue
            old=arms[baseline]
            assert old['initial_score']==new['initial_score']
            expected=new['conditional_gain']-old['conditional_gain']
            noise=new['innovation']-old['innovation']
            delta=new['score']-old['score']
            assert abs(expected+noise-delta)<1e-12
            pairs.append(dict(zip(('game','profile','seed','regime','turns','tail_steps','offset'),key),
                              baseline=baseline,candidate=candidate,score_delta=delta,
                              conditional_gain_delta=expected,innovation_delta=noise))
    return dict(sources=validated['sources'],rule_sha256=rule_hash,runs=runs,pairs=pairs,
        limitations=[
            'True response probabilities are used only after the completed runs, never passed to the NPC.',
            'Conditional gain is accumulated along each policy\'s actual visited states, not a counterfactual policy value.',
            'A positive conditional-gain difference in one realized path does not prove superior expected performance.',
            'States inherit previous luck and resource decisions. Delayed effects are included only when realized by the transition.',
            'Innovation is realized minus conditional mean; it is not a personality cost or model-error estimate.',
            'The accounting tolerance applies only to scalar diagnostic sums, never to owner-state audit equality.'
        ])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--studies',nargs='+',required=True)
    p.add_argument('--baselines',nargs='+',default=['endpoint@7','endpoint_posterior@7','endpoint@8','endpoint_posterior@8'])
    p.add_argument('--candidate',default='receding_recovery@1');p.add_argument('--output',required=True);a=p.parse_args()
    result=review(a.studies,a.baselines,a.candidate)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(runs=len(result['runs']),pairs=result['pairs']),indent=2))
