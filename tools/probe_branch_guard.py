"""Exact coupling diagnostic; not calibration or a replacement risk rule."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import Policy
from reflex.examples import context,action,effect
from reflex.intertemporal import Branch
from reflex.purpose_plan import Goal,goal_forecast
from reflex.repair_stability import safeguard


def check(values):
    actor=Policy(principle_priority='finite')
    c=context('coupling',[action(k,effect()) for k in ('a0','a1')],
              needs={'physiology':0.,'safety':0.})
    def forecast(addition):
        endpoints={'old':(.4,.6),'other':(0.,0.)}
        roots={'old':'a0','other':'a1'}
        if addition: endpoints['added']=values;roots['added']='a1'
        paths={k:tuple(Branch(.5,(effect(),),(1.,)) for v in pair) for k,pair in endpoints.items()}
        rows={k:[dict(seed=s,terminal=True,actions=[roots[k]],
                      assessment=Goal('finish','scored',v).record()) for s,v in enumerate(pair)]
              for k,pair in endpoints.items()}
        f=goal_forecast(c,paths,rows,horizon=1,unit='turns',target='finish',max_regret=0.,
                        plan_roots=roots,policy=actor,settlement_weight='absolute')
        f.audit['continuation_search']={'proposals':{k:{'branches':copy.deepcopy(v)} for k,v in rows.items()}}
        return f
    f=safeguard(c,forecast(True),forecast(False),actor)
    return f.audit['repair_stability']


def diagnostic():
    aligned=check((.5,.7)); crossed=check((.7,.5))
    assert aligned['protected'] and aligned['accepted']
    assert crossed['protected'] and not crossed['accepted']
    assert abs(sum(aligned['branch_deltas'])/2-.1)<1e-12
    assert abs(sum(crossed['branch_deltas'])/2-.1)<1e-12
    return dict(baseline_distribution=[.4,.6],candidate_distribution=[.5,.7],
        baseline_mean=.5,candidate_mean=.6,aligned=aligned,crossed=crossed,
        scope='same exact marginal outcome distributions, different declared pairing; no real-world trial',
        conclusion='per-branch dominance depends on coupling, not solely return distribution or knowledge confidence')


if __name__=='__main__': print(json.dumps(diagnostic(),indent=2))
