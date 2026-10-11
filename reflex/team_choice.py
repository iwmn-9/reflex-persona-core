"""Optional cooperation from communicated bids, with bounded individual consent.

Membership and the common purpose are supplied by the game. A forecast is a
proposal, not permission to command an unsigned ally or read a rival's mind.
No new personality axes, experience samples or persistent psychological state.
"""
import copy
from dataclasses import replace
import numpy as np
from .core import Policy, compile_batch, digest, identifier
from .deliberation import select as joint_select


def concession(context):
    """Engineering projection of existing axes, in current Policy score units."""
    p=context['personality'];v=context['values']
    return float(np.clip(.05+.25*p['agreeableness']+.1*v['benevolence']-.05*v['power'],.05,.4))


def select(base,forecast,*,group,mode='consent'):
    identifier(group)
    if mode not in ('sum','consent'):raise ValueError('declared cooperation mode required')
    owners=[c['scope']['npc'] for c in base]
    if len(set(owners))!=len(owners):raise ValueError('distinct consenting owners required')
    if len({(c['scope']['game'],c['scope']['episode'],c['tick']) for c in base})!=1:
        raise ValueError('one game, episode and simultaneous tick required')
    # Reuse the existing complete legality, owner, unchanged-facts and forecast
    # contract before computing a different optional selection criterion.
    policy=Policy(principle_priority='finite')
    legacy,audit=joint_select(base,forecast,policy)
    audit.update(group=group,members=owners,cooperation_mode=mode)
    if mode=='sum' or legacy is None:return legacy,audit
    names=tuple(sorted(forecast.roots))
    contexts=[]
    for original in forecast.contexts:
        c=copy.deepcopy(original);by={a['id']:a for a in c['actions']}
        c['actions']=[by[k] for k in names];contexts.append(c)
    b=compile_batch(contexts);d=policy.decide(b,False)
    purpose=np.array([forecast.purpose[k] for k in names])
    allowed=purpose>=purpose.max()-forecast.max_regret-1e-12
    maximum=np.max(np.where(d.eligible,d.scores,-np.inf),axis=1)
    regret=maximum[:,None]-d.scores
    limits=np.array([concession(c) for c in base])
    consent=d.eligible & (regret<=limits[:,None]+1e-12)
    acceptable=allowed & consent.all(0)
    audit.update(concessions=dict(zip(owners,limits.tolist())),
        consent_options=consent.sum(1).tolist(),accepted_plans=int(acceptable.sum()))
    if not acceptable.any():
        audit.update(adopted=False,reason='no common purpose plan within each communicated concession')
        return None,audit
    j=int(np.argmax(np.where(acceptable,-regret.mean(0),-np.inf)));key=names[j]
    records=replace(d,action=np.full(len(base),j,dtype=int)).records(b)
    for i,(c,root,record) in enumerate(zip(base,forecast.roots[key],records)):
        a=next(a for a in c['actions'] if a['id']==root);old=c['state']
        record.update(context_hash=digest(c),action_id=root,target=a['target'],reading_used=True)
        record['next_state'].update(intent_action=root,
            age=min(old['age']+1,1000000) if old['intent_action']==root else 0)
    audit.update(adopted=True,selected_plan=key,selected_purpose=float(purpose[j]),
        subjective_regret=float(-regret[:,j].mean()),
        member_regrets=dict(zip(owners,regret[:,j].tolist())),
        reason='common purpose with individually bounded concession')
    return records,audit
