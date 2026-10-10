"""Fresh-stream test of omitting the automatic reward for harming disliked NPCs."""
from collections import Counter
import json
from pathlib import Path
from .encounters import run
from .encounter_experiment import PLAN as BASE, experiment as compare, audit as replay
from .social_withdrawal import WithdrawalSocialPopulation


PLAN=dict(BASE,version='npc-withdrawal-v1',seeds=list(range(4020,4028)),
    diagnostic_seeds=list(range(4010,4018)),
    diagnostic_source_commit='d694a55b73133e468791fe9eda0bfb3d7489b8ce',
    settings=['prediction','continuous','coarse','withdrawal'],
    intervention='omit positive grudge adjustment for negative-support actions; same memory and three stances',
    adoption_gate=dict(score_better_at_least_worse=True,no_new_deaths=True,
        mutual_claim_pairs_lower=True,cooperation_pairs_not_lower=True,
        max_care_group_mean_drop=.25,automatic_grudge_harm_bonus_zero=True))


def execute(seed,domain,layout,setting,turns=48,*,contact='free'):
    if setting!='withdrawal':return run(seed,domain,layout,setting,turns,contact=contact)
    r,t=run(seed,domain,layout,'coarse',turns,contact=contact,population_factory=WithdrawalSocialPopulation)
    r['setting']='withdrawal';return r,t


def adoption(r):
    lookup={tuple(x[k] for k in ('domain','layout','contact','seed','setting')):x for x in r['runs']}
    counts=Counter();care_regressions=[]
    for key,new in lookup.items():
        if key[-1]!='withdrawal':continue
        old=lookup[key[:-1]+('coarse',)]
        baseline=lookup[key[:-1]+('prediction',)]
        for a,b,p in zip(new['actors'],old['actors'],baseline['actors']):
            assert a['npc']==b['npc']==p['npc']
            delta=a['score']-b['score'];counts['better' if delta>1e-9 else 'worse' if delta< -1e-9 else 'same']+=1
            counts['new_deaths']+=a['health']<=0 and (b['health']>0 or p['health']>0)
        for setting,x in (('coarse',old),('withdrawal',new)):
            counts[setting+'_mutual_claim_pairs']+=x['interaction'].get('mutual_claim_pairs',0)
            counts[setting+'_cooperation_pairs']+=x['cooperation_pairs']
    for g in r['summary']['groups']:
        old=g['settings']['coarse']['profile_mean_score']['care'];new=g['settings']['withdrawal']['profile_mean_score']['care']
        if new<old-.25:care_regressions.append(dict(domain=g['domain'],layout=g['layout'],contact=g['contact'],delta=new-old))
    checks=dict(score_better_at_least_worse=counts['better']>=counts['worse'],no_new_deaths=counts['new_deaths']==0,
        mutual_claim_pairs_lower=counts['withdrawal_mutual_claim_pairs']<counts['coarse_mutual_claim_pairs'],
        cooperation_pairs_not_lower=counts['withdrawal_cooperation_pairs']>=counts['coarse_cooperation_pairs'],
        care_group_mean_drop_within_limit=not care_regressions)
    return dict(passed=all(checks.values()),checks=checks,paired_counts=dict(counts),care_group_regressions=care_regressions)


def experiment(output):
    r=compare(output,plan=PLAN,executor=execute,extra_sources=('social_withdrawal.py','encounter_withdrawal_experiment.py'))
    r['adoption']=adoption(r)
    (Path(output)/'evaluation.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return r


def audit(output):
    p=replay(output,executor=execute);root=Path(output);r=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    assert adoption(r)==r['adoption'];harm_actions=0
    with (root/'trajectories.jsonl').open(encoding='utf-8') as f:
        for line in f:
            s=json.loads(line)
            if s['run']['setting']!='withdrawal':continue
            for t in s['trace']:
                for actor in t['appraisal']:
                    for key,row in actor.items():
                        if key.startswith('claim:') and row['stance']<0:
                            assert row['adjustment']==0;harm_actions+=1
    p.update(disliked_harm_actions_without_bonus=harm_actions,adoption=r['adoption'])
    (root/'audit.json').write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8');return p


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    r=audit(args.output) if args.audit else experiment(args.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ('plan','runs','summary')},indent=2))
