"""Frozen ablation of beam search, random shooting and their shared pool."""
from .laboratory import profiles
from .observed_transfer import run
from .search_transfer import experiment as compare


def configuration(variant):
    if variant not in ('beam','shooting','hybrid'):raise ValueError('known search portfolio required')
    return dict(search=True,width=2,depth=0 if variant=='shooting' else 2,
        samples=0 if variant=='beam' else 12,principle_priority='finite',max_regret=2.)


def jobs():
    diagnostic=[dict(genre='delivery',limit=9,capacity=2,supply=3)]
    holdout=[dict(genre='delivery',limit=10,capacity=2,supply=4),
        dict(genre='resources',route='mixed',limit=17),
        dict(genre='auction',prizes=[7,16,10],budget=13,rival_shift=True,revised_prize=True),
        dict(genre='combat',hp=9,limit=10,distance=4)]
    for split,specs in (('diagnostic',diagnostic),('holdout',holdout)):
        for s in specs:
            for p in profiles():
                for seed in (360,361) if s['genre']=='combat' else (360,):
                    for v in ('beam','shooting','hybrid'):yield split,s,p,v,seed,6


def worker(job):
    split,s,p,v,seed,h=job
    r=run(s,p,'verified',seed,h,**configuration(v));r.update(split=split,variant=v)
    searches=[t['deliberation']['metadata']['continuation_search'] for t in r['trace']]
    r['search_cost']=dict(shared_sequence_evaluations=sum(x['evaluated'] for x in searches),
        single_branch_pilots=sum(x.get('pilot_evaluations',0) for x in searches),
        model_transition_upper_bound=sum(h*(len((820,821,822,823)) if s['genre']=='combat' else 1)*x['evaluated']+
            h*x.get('pilot_evaluations',0) for x in searches))
    return r


def experiment(output,progress=None,workers=4):
    return compare(output,progress,workers,tasks=jobs(),candidate='hybrid',reference='beam',runner=worker,
        registration=dict(format='hybrid-search-transfer-v1',
            parameters=dict(horizon=6,proof_margin=0.,policies={v:configuration(v) for v in ('beam','shooting','hybrid')}),
            status='known failed delivery diagnostic; new parameter combinations in four existing engines fixed before comparison; no result-dependent tuning; not unseen-game generalization',
            adoption='optional hybrid only if every holdout purpose/survival/shortage/reachable-route measure is preserved and at least one holdout failure becomes success; report shooting-only separately; no global default change',
            semantics='finite personality scorer and exact public model remain fixed; uniform legal model pilots generate full-horizon sequences, frozen and scored on all model seeds; beam and sampled candidates share the same ranking; baseline proposal for every root survives; only the selected root executes then replans; pilots are not execution randomness or empirical training; budgets are not equal between methods'))
