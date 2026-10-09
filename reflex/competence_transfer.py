"""Separate persona preferences, purpose competence and proposal validation."""
from .laboratory import profiles
from .observed_transfer import run
from .search_transfer import experiment as compare


def configuration(variant,genre):
    if variant not in ('hybrid','guarded','validated'):raise ValueError('known competence ablation required')
    return dict(search=True,width=2,depth=2,samples=12,principle_priority='finite',
        max_regret=2. if variant=='hybrid' else .15,
        validation_seeds=tuple(range(1000,1008)) if variant=='validated' and genre=='combat' else None)


def jobs():
    diagnostic=[dict(genre='delivery',limit=9,capacity=2,supply=3)]
    holdout=[dict(genre='delivery',limit=9,capacity=2,supply=4),
        dict(genre='delivery',limit=8,capacity=3,supply=4),
        dict(genre='resources',route='mixed',limit=18),
        dict(genre='auction',prizes=[12,6,18],budget=14,rival_shift=True,revised_prize=True),
        dict(genre='combat',hp=7,limit=12,distance=4)]
    for split,specs in (('diagnostic',diagnostic),('holdout',holdout)):
        for s in specs:
            for p in profiles():
                for seed in (380,381) if s['genre']=='combat' else (380,):
                    for v in ('hybrid','guarded','validated'):yield split,s,p,v,seed,6


def worker(job):
    split,s,p,v,seed,h=job;kw=configuration(v,s['genre'])
    r=run(s,p,'verified',seed,h,**kw);r.update(split=split,variant=v)
    searches=[t['deliberation']['metadata']['continuation_search'] for t in r['trace']]
    r['search_cost']=dict(shared_sequence_evaluations=sum(x['evaluated'] for x in searches),
        single_branch_pilots=sum(x.get('pilot_evaluations',0) for x in searches),
        validation_evaluations=sum(x.get('validation',{}).get('evaluated',0) for x in searches),
        model_transition_upper_bound=sum(h*(4 if s['genre']=='combat' else 1)*x['evaluated']+
            h*x.get('pilot_evaluations',0)+h*len(x.get('validation',{}).get('seeds',()))*
            x.get('validation',{}).get('evaluated',0) for x in searches))
    return r


def experiment(output,progress=None,workers=4):
    return compare(output,progress,workers,tasks=jobs(),candidate='validated',reference='hybrid',runner=worker,
        registration=dict(format='competence-transfer-v1',
            parameters=dict(horizon=6,proof_margin=0.,
                policies={g:{v:configuration(v,g) for v in ('hybrid','guarded','validated')}
                    for g in ('delivery','resources','auction','combat')}),
            status='purpose-corridor and deeper-search development inspected on delivery limit9/capacity2/supply3; prior hybrid outcomes exposed; these holdout parameter combinations and actual seeds fixed before execution; four existing rule engines, not unseen-game generalization',
            adoption='validated versus hybrid: all holdout purpose/survival/shortage/reachable-route measures must be preserved and at least one failure must become success; guarded-only ablation reported independently; never silently replace default policy',
            semantics='same traits, principles, need pressure, finite Policy coefficients and search budget; .15 existing purpose-regret constraint versus full [-1,1] corridor; optional final reevaluation of frozen candidates on disjoint model seeds; no actual future, imagined experience learning, game-specific priority or score calibration'))
