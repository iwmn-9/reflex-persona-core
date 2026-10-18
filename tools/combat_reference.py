"""Evaluator-only finite combat value, including death, health and ammunition.

This independent Bellman reference knows the test world's true response law.
It is never a predictor or a policy supplied to an NPC. Opportunity loss also
includes personality's sacrifices and unknown future changes, not just mistakes.
"""
from functools import lru_cache


def reference(turns, probabilities):
    if type(turns) is not int or turns<1:raise ValueError('positive finite horizon required')
    laws=[tuple(probabilities(t)) for t in range(turns)]
    if any(len(p)!=3 or any(not 0<=x<=1 for x in p) or abs(sum(p)-1)>1e-12 for p in laws):
        raise ValueError('three normalized public response probabilities required')
    # Very short worlds can reach the goal's clipping floor. Carry accumulated
    # score in that small exact tree. For >=5 turns, legal score <=3*t and the
    # minimum terminal health (-7 quarters) make clipping unreachable; score is
    # then an additive constant, avoiding a redundant long-run state dimension.
    carry_score=turns<5

    @lru_cache(None)
    def value(t,quarters,ammo,score):
        if t==turns or quarters<=0:
            return max(-3*turns,min(3*turns,score+quarters/4-12)) if carry_score else quarters/4
        return max(x[1] for x in options(t,quarters,ammo,score))

    def options(t,quarters,ammo,score):
        wear=4 if t%8==7 else 0
        choices=[('guard',value(t+1,min(48,quarters+8)-wear,ammo,score)),
                 ('reload',value(t+1,quarters-1-wear,3,score))]
        if ammo:
            hit=(0 if carry_score else 3)+value(t+1,quarters-wear,ammo-1,score+3 if carry_score else 0)
            miss=value(t+1,quarters-4-wear,ammo-1,score)
            choices.extend((f'strike:{z}',p*hit+(1-p)*miss) for z,p in enumerate(laws[t]))
        return tuple(choices)

    def state(w):
        t=w['tick'];h=w['hp']*4;a=w['ammo']
        if (type(t) is not int or not 0<=t<=turns or h!=int(h) or not -7<=h<=48
            or type(a) is not int or not 0<=a<=3 or not 0<=w['score']<=3*t or w['score']%3):
            raise ValueError('public state from the declared combat rules required')
        return t,int(h),a,int(w['score']) if carry_score else 0

    def public_value(w):
        return ((0 if carry_score else w['score']-12)+value(*state(w)))/(3*turns)

    def public_choices(w):
        t,h,a,score=state(w)
        if t==turns or h<=0:return {}
        return {k:((0 if carry_score else w['score']-12)+v)/(3*turns) for k,v in options(t,h,a,score)}

    return public_value,public_choices


def account(row):
    if row['game']!='combat':raise ValueError('combat reference is evaluator-only for that rule family')
    from tools.run_owned_value import law
    value,choices=reference(row['turns'],lambda t:tuple(law(t,row['turns'],row['regime']).values()))
    initial=value(row['trace'][0]['before']);loss=noise=0.;steps=[]
    for t in row['trace']:
        q=choices(t['before']);regret=max(q.values())-q[t['action']]
        residual=value(t['after'])-q[t['action']]
        loss+=regret;noise+=residual
        steps.append(dict(tick=t['tick'],purpose_opportunity_loss=regret,residual=residual))
    error=initial-loss+noise-row['score']
    assert abs(error)<1e-10,'combat Bellman account must reconstruct final score'
    return dict(initial_perfect_information=initial,purpose_opportunity_loss=loss,
        residual=noise,residual_removed=initial-loss,reconstruction_error=error,increments=steps)
