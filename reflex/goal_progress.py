"""A common fallback for indistinguishable terminal goal predictions.

The game adapter supplies both terminal success samples and bounded progress
samples. Progress never overrules a distinguishable success forecast. It is a
model-relative competence boundary, not a claim of optimal play or calibration.
"""
import copy
import numpy as np
from .goal_guard import choose_with_goal


def relative_progress(scores,viewer,*,direction,scale):
    """Bounded distance from the leading rival; units come from the adapter.

    Works with any participant count >= 2 and arbitrary numerical score units.
    Unlike rank alone, this remains informative even when every candidate loses.
    """
    scores=np.asarray(scores,dtype=float)
    if scores.ndim!=3 or scores.shape[2]<2 or type(viewer) is not int or not 0<=viewer<scores.shape[2]:
        raise ValueError('candidate/sample/participant scores and a valid viewer required')
    if not np.isfinite(scores).all() or direction not in (-1,1) or not np.isfinite(scale) or scale<=0:
        raise ValueError('finite scores, score direction and positive score units required')
    rivals=[i for i in range(scores.shape[2]) if i!=viewer]
    margin=direction*scores[:,:,viewer]-(direction*scores[:,:,rivals]).max(2)
    return .5+np.arctan(margin/scale)/np.pi


def progress_context(context,names,progress):
    """Replace the constant success signal while retaining all personality axes.

    The same bounded outcome compression as the adapter is used. Root effects
    and risk are retained; this does not add traits or change values/needs.
    """
    from .planning import compress_outcomes
    result=copy.deepcopy(context);lookup={name:i for i,name in enumerate(names)}
    for act in result['actions']:
        samples=progress[lookup[act['id']]]
        # A constant terminal goal means all samples have the same winner credit;
        # the adapter therefore packed one row. Expand that row with progress.
        rows=[]
        for value in np.unique(samples):
            row=copy.deepcopy(act['outcomes'][0]);row['p']=float((samples==value).mean())
            row['objective']=2*float(value)-1;row['values']['achievement']=row['objective'];rows.append(row)
        act['outcomes']=compress_outcomes(rows)
    result['facts']['goal_signal']='勝利予測が全候補・全標本で同じ定数のため、ゲームが供給した首位との差の進捗値で比較'
    return result


def choose_with_progress(context,names,success,progress,*,max_regret=.12):
    """Secondary evidence is used only on a wholly constant primary plateau.

    Both arrays are independently validated, even if the primary is informative.
    Exactly tied *means* alone do not establish a plateau: different stochastic
    success events must keep their risk and outcome information.
    """
    success=np.asarray(success,dtype=float);progress=np.asarray(progress,dtype=float)
    if success.shape!=progress.shape or progress.ndim!=2 or not np.isfinite(progress).all() or np.any((progress<0)|(progress>1)):
        raise ValueError('matched finite normalized progress samples required')
    # Validate the complete primary contract before changing context or selecting.
    original,primary_guard=choose_with_goal(context,names,success,max_regret=max_regret)
    flat=bool(success.size and np.all(success==0))
    if not flat:
        primary_guard['signal']='terminal_success';return context,original,primary_guard
    adjusted=progress_context(context,names,progress)
    result,guard=choose_with_goal(adjusted,names,progress,max_regret=max_regret)
    guard.update(signal='goal_progress_on_constant_success',primary_constant=float(success.flat[0]),
                 original_success_personality_action=original['action_id'])
    return adjusted,result,guard
