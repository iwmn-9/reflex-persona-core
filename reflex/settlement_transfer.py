"""Compare terminal time semantics without a hard purpose corridor."""
from .laboratory import profiles
from .observed_transfer import run
from .search_transfer import experiment as compare


def configuration(variant):
    if variant not in ('discounted','absolute'):raise ValueError('known settlement time semantics required')
    return dict(search=True,width=2,depth=2,samples=12,principle_priority='finite',
        max_regret=2.,settlement_weight=variant)


def jobs():
    diagnostic=[dict(genre='delivery',limit=9,capacity=2,supply=3)]
    holdout=[dict(genre='delivery',limit=8,capacity=3,supply=3),
        dict(genre='resources',route='science',opening='conversion',limit=6),
        dict(genre='auction',prizes=[14,8,19],budget=15,rival_shift=True,revised_prize=True),
        dict(genre='combat',hp=8,limit=11,distance=4)]
    for split,specs in (('diagnostic',diagnostic),('holdout',holdout)):
        for s in specs:
            for p in profiles():
                for seed in (400,401) if s['genre']=='combat' else (400,):
                    for v in ('discounted','absolute'):yield split,s,p,v,seed,6


def worker(job):
    split,s,p,v,seed,h=job
    r=run(s,p,'verified',seed,h,**configuration(v));r.update(split=split,variant=v)
    searches=[t['deliberation']['metadata']['continuation_search'] for t in r['trace']]
    r['search_cost']=dict(shared_sequence_evaluations=sum(x['evaluated'] for x in searches),
        single_branch_pilots=sum(x.get('pilot_evaluations',0) for x in searches),
        model_transition_upper_bound=sum(h*(4 if s['genre']=='combat' else 1)*x['evaluated']+
            h*x.get('pilot_evaluations',0) for x in searches))
    return r


def experiment(output,progress=None,workers=4):
    return compare(output,progress,workers,tasks=jobs(),candidate='absolute',reference='discounted',runner=worker,
        registration=dict(format='settlement-transfer-v1',
            parameters=dict(horizon=6,proof_margin=0.,policies={v:configuration(v) for v in ('discounted','absolute')}),
            status='known delivery diagnostic exposed; terminal-time interpretation developed on that diagnostic and synthetic mechanism checks; new holdout parameter combinations frozen before comparison, four existing rule engines; conversion opening tests late allocation, not replacing previous mixed-goal full-start tests; not unseen-game generalization',
            adoption='optional absolute mode only if every holdout purpose/survival/shortage/reachable-route measure is preserved and at least one failure becomes success; no default switch and no post-holdout tuning',
            semantics='same finite Policy coefficients, personality/needs/principles, horizon/search/pilots/model bank, full purpose corridor; only subjective modeled terminal success/failure/draw/scored time decay removed; nonterminal cutoff proxy and need/value/style/cost flows retain time decay; confidence retained; outcome once, never repeated income; permits meaningful principle-driven goal sacrifice; no actual future or empirical learning'))
