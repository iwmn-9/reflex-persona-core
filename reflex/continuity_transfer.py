"""Same finite search, plus owned prior intention and modeled arrival preference."""
from .laboratory import profiles
from .search_transfer import experiment as compare


def jobs():
    diagnostic=[dict(genre='resources',route='mixed',limit=14),
                dict(genre='delivery',limit=10,capacity=2,supply=2)]
    holdout=[dict(genre='resources',route='mixed',limit=15),
             dict(genre='delivery',limit=9,capacity=3,supply=3),
             dict(genre='auction',prizes=[12,7,15],budget=10,rival_shift=True,revised_prize=True),
             dict(genre='combat',hp=6,limit=8,distance=4)]
    for split,specs in (('diagnostic',diagnostic),('holdout',holdout)):
        for s in specs:
            for p in profiles():
                for seed in (301,302) if s['genre']=='combat' else (301,):
                    for v in ('verified','search','continuity'):yield split,s,p,v,seed,6


def experiment(output,progress=None,workers=4):
    return compare(output,progress,workers,tasks=jobs(),candidate='continuity',reference='search',
        registration=dict(format='continuity-transfer-v1',
            status='previous resource/delivery holdout now diagnostic; new four-genre parameter combinations and combat actual seeds frozen before this comparison; known rule engines, not unseen-game transfer; no holdout tuning',
            adoption='optional candidate only if every holdout pair versus search preserves goal outcome/survival without increased shortages or reachable-route loss, at least one holdout success improves, and the known resource-ego diagnostic failure is recovered; verified is a second reference, not automatically dominated',
            continuity='retain only the all-model-branch common remaining action prefix after actual execution; reevaluate on the current observation; prune later modeled-success arrivals only when the incumbent remains supported by persona/progress/purpose and no higher expected purpose is offered; no imposed game action, reward coefficient, or hypothetical learning'))
