"""Join fixed development controls and holdout results; no controller tuning.

Repeated controls are reused, not counted as new trials. Switching of a gate
is diagnostic only: the number alone does not distinguish adaptation/churn.
"""
import argparse,collections,json
from pathlib import Path


def read(root,name):return json.loads((root/name).read_text())
def rows(root):return [json.loads(line) for line in (root/'trajectories.jsonl').read_text().splitlines()]


def aggregate(pairs):
    grouped=collections.defaultdict(list)
    for p in pairs:
        for baseline,score in p['scores'].items():
            if baseline!='persona_choice':
                grouped[(p['game'],baseline)].append(p['scores']['persona_choice']-score)
    return [dict(game=g,against=b,better=sum(x>1e-12 for x in ds),equal=sum(abs(x)<=1e-12 for x in ds),
                 worse=sum(x< -1e-12 for x in ds),mean_delta=sum(ds)/len(ds))
            for (g,b),ds in sorted(grouped.items())]


def review(development,support,holdout,output):
    d=read(development,'audit.json');s=read(support,'audit.json');h=read(holdout,'audit.json')
    assert not read(development,'registration.json')['holdout'] and read(holdout,'registration.json')['holdout']
    lookup={(p['game'],p['profile'],p['schedule']):p for p in s['pairs']}
    pairs=[]
    for p in d['pairs']:
        control=lookup[(p['game'],p['profile'],p['schedule'])]
        pairs.append(dict(p,scores=dict(p['scores'],purpose_joint=control['scores']['support'],
                                       legacy_support=control['scores']['legacy-support'])))
    diagnostics=[]
    for label,root in (('development',development),('holdout',holdout)):
        for row in rows(root):
            last={};activations=revocations=0
            # Evidence states are recorded AFTER feedback, not a claim that
            # the next actual root necessarily uses that candidate (a planner
            # or a legal/progress restriction can supersede it).
            for step in row['trace']:
                now={k:v for k,v in step['evidence'].items() if k.startswith('response:') and ':decision:persona-v1' in k}
                for k,v in now.items():
                    activations+=int(v=='active' and last.get(k)!='active')
                    revocations+=int(v!='active' and last.get(k)=='active')
                last=now
            diagnostics.append(dict(study=label,game=row['game'],profile=row['profile'],schedule=row['schedule'],
                arm=row['arm'],activations=activations,revocations=revocations))
    result=dict(development=dict(pairs=pairs,comparisons=aggregate(pairs)),
        holdout=dict(pairs=h['pairs'],comparisons=aggregate(h['pairs'])),gate_transitions=diagnostics,
        limitations=['Development reuses the seed-17 support controls; these are not additional simulations.',
            'Each matrix has one shared real seed, four fixed profiles, and two schedules, not 32 independent seeds.',
            'Purpose points and personal preference are different metrics; a score reduction alone does not identify a personality cost.',
            'Post-feedback gate switches alone neither establish nor disprove excessive plan changes.'])
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k]['comparisons'] for k in ('development','holdout')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--development',type=Path,required=True);p.add_argument('--support',type=Path,required=True)
    p.add_argument('--holdout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    review(a.development,a.support,a.holdout,a.output)
