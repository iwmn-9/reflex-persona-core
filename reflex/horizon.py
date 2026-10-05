"""Game-independent temporal purpose evaluation, separate from personality.

Games supply the same bounded purpose meaning at each completed model step.
Earlier purposeful progress can break perpetual 'act later' ties. This is an
explicit objective tradeoff, not a learned win probability or a forced lease.
"""
import math


def purpose_return(path,progress_weight=0.):
    if not path or isinstance(progress_weight,bool) or not isinstance(progress_weight,(int,float)) or not math.isfinite(progress_weight) or not 0<=progress_weight<=1:
        raise ValueError('nonempty purpose path and bounded temporal weight required')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not -1<=x<=1 for x in path):
        raise ValueError('same bounded purpose at each model step required')
    return float((1-progress_weight)*path[-1]+progress_weight*sum(path)/len(path))


def compare_paths(observations,forecasts,terminal=False):
    """Audit chosen model paths against later real observations of SAME target.

    Indices count adapter observation units (ticks or completed own rounds).
    Only absorbing terminal games permit extrapolation beyond their final row.
    Replanning can change the continuation: errors are diagnostics, never
    causal training samples or automatic model-confidence corrections.
    """
    purpose_return(observations)
    if type(terminal) is not bool:raise ValueError('explicit absorbing terminal required')
    immediate=[];delayed=[];deferred=0;unrealized=0
    for start,path,waiting in forecasts:
        if type(start) is not int or not 0<=start<len(observations) or type(waiting) is not bool:
            raise ValueError('owned observation index and explicit waiting label required')
        purpose_return(path)
        def observed(offset):
            j=start+offset
            return observations[j] if j<len(observations) else observations[-1] if terminal else None
        actual=observed(1)
        if actual is not None:immediate.append(abs(path[0]-actual))
        late=observed(len(path))
        if late is not None:delayed.append(abs(path[-1]-late))
        if waiting and path[0]<=observations[start]+.005 and max(path)>observations[start]+.005:
            deferred+=1
            unrealized+=int(late is not None and late<=observations[start]+.005)
    return dict(immediate_samples=len(immediate),immediate_absolute_error=sum(immediate),
        delayed_samples=len(delayed),delayed_absolute_error=sum(delayed),
        modeled_later_progress_while_waiting=deferred,unrealized_later_progress=unrealized)
