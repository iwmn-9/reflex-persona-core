"""Post-hoc execution review: changes, late-game effects, and persistent losses.

These summaries explain a completed comparison; they are not new trials or a
replacement acceptance criterion. Game scores remain separate. After a real
terminal state, its settled score is carried forward without inventing actions.
"""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from tools.compare_owned_studies import compare
from tools.run_owned_value import law


def review(studies,baselines,candidate):
    validated=compare(studies,[]);groups=defaultdict(dict)
    for study in map(Path,studies):
        for row in map(json.loads,(study/'trajectories.jsonl').read_text().splitlines()):
            key=tuple(row[k] for k in ('game','profile','seed','regime','turns','tail_steps','offset'))
            groups[key][row['arm']+'@'+str(row['interval'])]=dict(
                score=row['score'],goals=[t['goal'] for t in row['trace']],
                brier=[t['brier'] for t in row['trace']],
                actions=[t['action'] for t in row['trace']])
    pairs=[]
    for key,arms in sorted(groups.items()):
        if candidate not in arms:continue
        game,profile,seed,regime,turns,tail,offset=key
        n=arms[candidate]
        changes=[0]+[t for t in range(1,turns) if law(t,turns,regime)!=law(t-1,turns,regime)]+[turns]
        for baseline in baselines:
            if baseline not in arms:continue
            b=arms[baseline]
            def at(row,t):
                # Initial state is identical in a matched pair; only its
                # difference is used, so zero is a common origin here.
                return row['goals'][min(t,len(row['goals']))-1] if t else 0.
            def difference(t):return at(n,t)-at(b,t)
            phases=[]
            for index,(start,end) in enumerate(zip(changes,changes[1:])):
                middle=(start+end)//2
                for half,lo,hi in [('early',start,middle),('late',middle,end)]:
                    if lo==hi:continue
                    bn=b['brier'][lo:hi];nn=n['brier'][lo:hi]
                    phases.append(dict(phase=index,start=lo,end=hi,half=half,
                        purpose_gain_delta=difference(hi)-difference(lo),
                        baseline_decisions=len(bn),candidate_decisions=len(nn),
                        baseline_brier=float(np.mean(bn)) if bn else None,
                        candidate_brier=float(np.mean(nn)) if nn else None))
            cutoff=max(0,turns-tail-1)
            pairs.append(dict(game=game,profile=profile,seed=seed,regime=regime,turns=turns,
                baseline=baseline,candidate=candidate,total_delta=n['score']-b['score'],
                before_last_forecast_window=difference(cutoff),
                last_forecast_window_delta=difference(turns)-difference(cutoff),
                baseline_executed=len(b['goals']),candidate_executed=len(n['goals']),
                phases=phases))
    summaries=[]
    by_group=defaultdict(list)
    for p in pairs:by_group[(p['game'],p['profile'],p['regime'],p['baseline'])].append(p)
    for key,rows in sorted(by_group.items()):
        summaries.append(dict(zip(('game','profile','regime','baseline'),key),world_seeds=len(rows),
            mean_delta=float(np.mean([r['total_delta'] for r in rows])),
            losses=sum(r['total_delta']< -1e-12 for r in rows),
            mean_before_last_window=float(np.mean([r['before_last_forecast_window'] for r in rows])),
            mean_last_window=float(np.mean([r['last_forecast_window_delta'] for r in rows])),
            phases=[dict(phase=p['phase'],half=p['half'],start=p['start'],end=p['end'],
                mean_gain_delta=float(np.mean([r['phases'][i]['purpose_gain_delta'] for r in rows])))
                for i,p in enumerate(rows[0]['phases'])]))
    return dict(sources=validated['sources'],pairs=pairs,profile_regime_groups=summaries,
        limitations=['Post-hoc explanation, not an independent confirmation or a causal estimate of learning.',
                    'Phase scores also reflect resources and injuries inherited from earlier decisions.',
                    'Fewer than two world seeds cannot establish systematic behavior.',
                    'A settled terminal score is carried forward; no post-terminal predictions are invented.',
                    'Improvement before the final forecast window is not proof for an unbounded world.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--studies',nargs='+',required=True)
    p.add_argument('--baselines',nargs='+',default=['endpoint@7','endpoint_posterior@7','endpoint@8','endpoint_posterior@8'])
    p.add_argument('--candidate',default='receding_recovery@1');p.add_argument('--output',required=True);a=p.parse_args()
    result=review(a.studies,a.baselines,a.candidate)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(pairs=len(result['pairs']),groups=len(result['profile_regime_groups']))))
