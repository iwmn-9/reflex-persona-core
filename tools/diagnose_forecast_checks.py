"""Read frozen decisions to distinguish inactive checks from model failures."""
import argparse,json,statistics
from collections import Counter
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=Path(a.cohort);groups={};rows=[]
    with (root/'trajectories.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            for decision_step,t in enumerate(r['trace']):
                q=t.get('check')
                if q is None:continue
                key=(r['case']['game'],r['arm']);g=groups.setdefault(key,dict(reasons=Counter(),checks=0,changed=0,unprotected=0,se=[],terminal_goal_scores=0,adaptive=0))
                g['checks']+=1;g['reasons'][q['reason']]+=1
                changed=q['candidate_root']!=q['reference_root'];g['changed']+=changed
                g['unprotected']+=changed and (q['candidate'] is None or q['reference'] is None)
                g['adaptive']+=t.get('adaptive') is not None
                if 'standard_error' in q:g['se'].append(q['standard_error'])
                g['terminal_goal_scores']+=sum(m['terminal'] for m in t['modeled'])
                if changed:rows.append(dict(case=r['case'],profile=r['profile'],seed=r['seed'],arm=r['arm'],step=decision_step,check=q,final=r['goal']))
    summary=[]
    for (game,arm),g in sorted(groups.items()):
        se=g.pop('se');g['reasons']=dict(g['reasons']);summary.append(dict(game=game,arm=arm,**g,
            average_se=statistics.mean(se) if se else None,zero_se=sum(abs(x)<1e-12 for x in se),scored_checks=len(se)))
    result=dict(summary=summary,changed_checks=rows,
        limitations='missing reference may be discovery-tier restriction or independent-tier change; neither is physical illegality; predictive check is not a real-result guarantee')
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
