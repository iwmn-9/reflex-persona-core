"""Optional observed-behavior predictions with game-owned, recorded use gates."""
from dataclasses import replace
from .core import Policy, compile_batch
from .goofspiel import observe


def choice(context):
    b=compile_batch([context]); result=Policy().decide(b)
    return b,result,result.records(b)[0]


def use_gate(ahead,locked,maintained,same_action,eligible,gain,required_gain):
    if locked: return False,'mathematically_locked_lead'
    if same_action: return True,'same_action'
    if ahead and maintained: return False,'advantage_maintained_under_model'
    if not eligible: return True,'strongest_value_tier_changed'
    if gain>required_gain: return True,'material_persona_gain'
    return False,'gain_below_uncertainty_margin'


def decide(s,viewer,profile,goal,seed,tick,episode,memory=None,budget=None,beliefs=None):
    """Compare equal per-root budgets and common random variates, never peek.

    Safety gates are game design heuristics. Wilson/SE describe MC sampling,
    not whether the behavior hypotheses describe the real opponent correctly.
    """
    base,base_stats=observe(s,viewer,profile,goal,seed,tick,episode,memory,budget,common_random=True)
    _,_,baseline=choice(base)
    stats=dict(reason='no_reliable_public_evidence',model_evaluated=False,reading_applied=False,
               action_changed=False,baseline_action=baseline['action_id'],baseline_evaluation=base_stats,
               total_nodes=base_stats.get('additional_nodes',0),node_cap=budget.max_nodes if budget else 0)
    if beliefs is None or beliefs.strongest_confidence<.2:
        stats['belief_confidence']=0. if beliefs is None else beliefs.confidence
        return base,baseline,stats
    if beliefs.nonuniform_mass<.05:
        stats.update(reason='near_uniform_forecast',belief_confidence=beliefs.confidence,
                     nonuniform_mass_upper_bound=beliefs.nonuniform_mass)
        return base,baseline,stats
    rivals=[p for i,p in enumerate(s.scores) if i!=viewer]; ahead=s.scores[viewer]>max(rivals)
    locked=goal=='win_share' and s.scores[viewer]>max(rivals)+sum(s.prizes[s.round:])
    if locked:
        stats.update(reason='mathematically_locked_lead',belief_confidence=beliefs.confidence)
        return base,baseline,stats
    # Both evaluations share ONE total transition budget. Adapter/JSON work
    # remains outside this mechanical node cap, as in the underlying engine.
    read_budget=replace(budget,max_nodes=budget.max_nodes-stats['total_nodes']) if budget else None
    learned,read_stats=observe(s,viewer,profile,goal,seed,tick,episode,memory,read_budget,beliefs,common_random=True)
    stats['total_nodes']+=read_stats.get('additional_nodes',0)
    b,r,d=choice(learned); names=b.ids[0]; old=names.index(baseline['action_id']); new=names.index(d['action_id'])
    gain=float(r.scores[0,new]-r.scores[0,old]); required=.025
    old_prediction=read_stats.get('actions',{}).get(baseline['action_id'],{})
    new_prediction=read_stats.get('actions',{}).get(d['action_id'],{})
    if budget is not None:
        # Conservative, game-defined objective-only margin. It does not capture
        # uncertainty in all personality axes or model misspecification.
        required+=.65*sum((p.get('return_se') or 0.) for p in (old_prediction,new_prediction))
    interval=old_prediction.get('win_rate_interval95')
    maintained=bool(interval and interval[0]>.5 and old_prediction.get('win_share',0)>=.6)
    if budget is not None and not read_stats['used']:
        applied,reason=False,'learned_rollout_incomplete'
    else:
        applied,reason=use_gate(ahead and goal=='win_share',locked,maintained,d['action_id']==baseline['action_id'],
                               bool(r.eligible[0,old]),gain,required)
    stats.update(reason=reason,model_evaluated=True,reading_applied=applied,
                 action_changed=applied and d['action_id']!=baseline['action_id'],
                 belief_confidence=beliefs.confidence,beliefs=beliefs.record(),
                 learned_action=d['action_id'],persona_gain=gain,required_gain=required,
                 currently_ahead=ahead,maintained_under_model=maintained,learned_evaluation=read_stats)
    return (learned,d,stats) if applied else (base,baseline,stats)
