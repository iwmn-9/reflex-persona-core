"""Finite principle weights instead of unbounded pre-selection vetoes.

The main comparison needs no beam or retained intention. A separate known
search failure checks the same policy consistently through future choices.
"""
from .laboratory import profiles
from .observed_transfer import run
from .search_transfer import experiment as compare


def configuration(variant):
    if variant not in ('legacy','finite-banded','finite'):raise ValueError('known finite-principle comparison required')
    return dict(principle_priority='lexicographic' if variant=='legacy' else 'finite',
        max_regret=2. if variant=='finite' else .15)


def jobs():
    diagnostic=[dict(genre='resources',route='mixed',limit=14),dict(genre='combat',hp=6,limit=8,distance=4)]
    holdout=[dict(genre='resources',route='mixed',limit=16),dict(genre='delivery',limit=9,capacity=2,supply=3),
        dict(genre='auction',prizes=[8,17,11],budget=12,rival_shift=True,revised_prize=True),
        dict(genre='combat',hp=8,limit=9,distance=4)]
    for split,specs in (('diagnostic',diagnostic),('holdout',holdout)):
        for s in specs:
            for p in profiles():
                for seed in (340,341) if s['genre']=='combat' and split=='holdout' else (340,):
                    for v in ('legacy','finite-banded','finite'):yield split,s,p,v,seed,6
    for v in ('legacy','finite-banded','finite'):
        yield 'diagnostic-search',diagnostic[0],profiles()[3],v,340,6


def worker(job):
    split,s,p,v,seed,h=job
    r=run(s,p,'verified',seed,h,search=split=='diagnostic-search',**configuration(v))
    r.update(split=split,variant=v);return r


def requirements(runs,pairs):
    return dict(known_search_failure_recovered=any(r['split']=='diagnostic-search' and r['variant']=='finite' and
        r['profile']=='ego' and r['status']=='success' for r in runs))


def experiment(output,progress=None,workers=4):
    return compare(output,progress,workers,tasks=jobs(),candidate='finite',reference='legacy',runner=worker,requirements=requirements,
        registration=dict(format='finite-principle-transfer-v1',
            parameters=dict(horizon=6,width=2,depth=2,proof_margin=0.,policies={v:configuration(v) for v in ('legacy','finite-banded','finite')}),
            status='known resource/combat diagnostics; new parameter combinations in four existing rule engines frozen before comparison; no holdout tuning; not unseen-game transfer',
            adoption='finite candidate only if all holdout pairs preserve goal outcome/survival without increased shortages or reachable-route loss, at least one holdout success improves, and the known resource-ego SEARCH failure is recovered; finite-banded separates removing the hard principle veto from removing the purpose corridor',
            semantics='main episodes have ordinary reflex future continuations, no beam or intention; search diagnostic alone uses the unchanged bounded beam; identical personality, weights, legal transitions and seed separation; finite uses the same weighted scores without the lexicographic tier, max_regret=2 removes the goal corridor and permits substantial principled sacrifice'))
