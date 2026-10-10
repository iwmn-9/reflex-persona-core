"""Registered endogenous interaction comparison; no scripted opponent responses."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import digest
from .encounters import LAYOUTS, RULES, run


PLAN=dict(version='npc-encounters-v1',domains=list(RULES),layouts=list(LAYOUTS),
    contacts=['free','public'],settings=['prediction','continuous','coarse'],
    seeds=list(range(4010,4018)),development_seed=4000,turns=48,
    contact_slot_turns=4,public_history_window=8,
    rules=RULES,profile_layouts=LAYOUTS,
    comparisons='matched worlds within contact regime; contact regimes have distinct owner RNG scopes',
    acceptance='executable generic-core connection, fair observations, fixed axes, exact replay; no blanket performance promotion',
    caveat='two engineered worlds; not human playtesting, negotiation language or learned universal game evaluation')


def interaction_counts(trace,contact):
    counts=Counter();harm=set();cooperation=Counter();health={};examples=[]
    for t in trace:
        for r in t['rows']:
            counts['actual_choices' if r['kind']!='finished' else 'terminal_slots']+=1
            if r['target'] is not None and not r['matched']:counts['unmatched_attempts']+=1
            if r['matched']:
                key=(r['npc'],r['target'])
                if r['kind']==r['other_kind']=='cooperate':
                    counts['cooperative_actor_turns']+=1;cooperation[key]+=1
                    if key in harm and not r['luck_failure']:
                        counts['cooperation_after_harm']+=1
                        if len(examples)<3:examples.append(dict(tick=t['tick'],npc=r['npc'],target=r['target'],profile=r['profile']))
                if r['kind']=='cooperate' and r['other_kind']=='claim':harm.add(key);counts['exploited']+=1
                if r['kind']==r['other_kind']=='claim':counts['mutual_claim_actor_turns']+=1
                if r['luck_failure']:counts['joint_luck_failure_actor_turns']+=1
            health[r['npc']]=r['after']['health']
        assert t['fixed_personality']
    counts['cooperation_pairs']=counts['cooperative_actor_turns']//2
    counts['mutual_claim_pairs']=counts['mutual_claim_actor_turns']//2
    counts['repeated_cooperative_directed_relations']=sum(n>=3 for n in cooperation.values())
    return dict(counts),examples


def summarize(runs):
    groups=[];pairs={s:Counter() for s in ('continuous','coarse')}
    lookup={tuple(r[k] for k in ('domain','layout','contact','seed','setting')):r for r in runs}
    for domain in PLAN['domains']:
        for layout in PLAN['layouts']:
            for contact in PLAN['contacts']:
                settings={}
                for setting in PLAN['settings']:
                    rows=[r for r in runs if (r['domain'],r['layout'],r['contact'],r['setting'])==(domain,layout,contact,setting)]
                    role=defaultdict(list)
                    for r in rows:
                        for a in r['actors']:role[a['profile']].append(a['score'])
                    settings[setting]=dict(mean_world_score=float(np.mean([r['mean_score'] for r in rows])),
                        alive=sum(r['alive'] for r in rows),npc_episodes=sum(len(r['actors']) for r in rows),
                        matches=sum(r['matches'] for r in rows),cooperation_pairs=sum(r['cooperation_pairs'] for r in rows),
                        exploited=sum(r['exploited'] for r in rows),
                        mutual_claim_pairs=sum(r['interaction'].get('mutual_claim_pairs',0) for r in rows),
                        cooperation_after_harm=sum(r['interaction'].get('cooperation_after_harm',0) for r in rows),
                        repeated_cooperative_directed_relations=sum(r['interaction'].get('repeated_cooperative_directed_relations',0) for r in rows),
                        profile_mean_score={k:float(np.mean(v)) for k,v in sorted(role.items())})
                groups.append(dict(domain=domain,layout=layout,contact=contact,settings=settings))
    for key,r in lookup.items():
        if key[-1] not in pairs:continue
        baseline=lookup[key[:-1]+('prediction',)]
        for a,b in zip(r['actors'],baseline['actors']):
            assert (a['npc'],a['profile'])==(b['npc'],b['profile'])
            delta=a['score']-b['score'];pairs[key[-1]]['better' if delta>1e-9 else 'worse' if delta< -1e-9 else 'same']+=1
            pairs[key[-1]]['new_deaths']+=a['health']<=0<b['health']
    return dict(groups=groups,paired_actor_scores_against_prediction={k:dict(v) for k,v in pairs.items()})


def experiment(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    registration=dict(PLAN,source_hashes={name:digest(Path(__file__).with_name(name).read_text(encoding='utf-8'))
        for name in ('encounter_experiment.py','encounters.py','social_projection.py','social.py','core.py','runtime.py','laboratory.py')})
    path=root/'preregister.json'
    if path.exists():raise FileExistsError('fresh output required; no overwritten registration')
    path.write_text(json.dumps(registration,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    runs=[];tracepath=root/'trajectories.jsonl';checks=Counter()
    with tracepath.open('w',encoding='utf-8') as f:
        for domain in PLAN['domains']:
            for layout in PLAN['layouts']:
                for contact in PLAN['contacts']:
                    for seed in PLAN['seeds']:
                        for setting in PLAN['settings']:
                            r,trace=run(seed,domain,layout,setting,PLAN['turns'],contact=contact)
                            counts,examples=interaction_counts(trace,contact);r.update(interaction=counts,repair_examples=examples)
                            runs.append(r);checks.update(counts)
                            f.write(json.dumps(dict(run=r,trace=trace),ensure_ascii=False)+'\n')
                    print(f'completed {domain}/{layout}/{contact}',flush=True)
    result=dict(plan=registration,runs=runs,summary=summarize(runs),checks=dict(checks),world_runs=len(runs),
        npc_episodes=sum(len(r['actors']) for r in runs),trajectory_sha256=hashlib.sha256(tracepath.read_bytes()).hexdigest(),
        human_validation=False,enjoyment_measured=False,scripted_responses=False,training_performed=False)
    (root/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


def audit(output):
    root=Path(output);r=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    registration=json.loads((root/'preregister.json').read_text(encoding='utf-8'));assert registration==r['plan']
    for name,expected in registration['source_hashes'].items():
        assert digest(Path(__file__).with_name(name).read_text(encoding='utf-8'))==expected,name
    path=root/'trajectories.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==r['trajectory_sha256']
    runs=[];counts=Counter()
    with path.open(encoding='utf-8') as f:
        for line in f:
            saved=json.loads(line);s=saved['run']
            generated,trace=run(s['seed'],s['domain'],s['layout'],s['setting'],s['turns'],contact=s['contact'])
            c,examples=interaction_counts(trace,s['contact']);generated.update(interaction=c,repair_examples=examples)
            assert generated==s and trace==saved['trace'];runs.append(s);counts.update(c)
    assert runs==r['runs'] and summarize(runs)==r['summary'] and dict(counts)==r['checks']
    proof=dict(source_identity=True,all_worlds_replayed=len(runs),all_actual_choices=counts['actual_choices'],
        terminal_slots=counts['terminal_slots'],personality_axes_fixed=True,scripted_responses=False,
        trajectory_sha256=r['trajectory_sha256'],human_validation=False,enjoyment_measured=False)
    (root/'audit.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');return proof


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--audit',action='store_true');args=p.parse_args()
    r=audit(args.output) if args.audit else experiment(args.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ('plan','runs','summary')},indent=2))
