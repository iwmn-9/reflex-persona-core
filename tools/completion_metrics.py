"""Offline outcome/recovery metrics. Truth never enters an agent through this module."""
import math

WINDOW = 8
BRIER_LIMIT = 0.10
RECOVERY_LIMIT = 16
EPS = 1e-12


def distribution(value):
    if not isinstance(value, dict) or not value:
        raise ValueError('nonempty discrete distribution required')
    if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in value.values()):
        raise ValueError('finite nonnegative probabilities required')
    if not math.isclose(math.fsum(value.values()), 1., abs_tol=EPS, rel_tol=0):
        raise ValueError('probabilities must sum to one')
    return value


def summarize_trace(trace, *, scheduled_turns, terminal, truth_schedule):
    """Different trajectories are not interpreted as matched-state regret."""
    if type(scheduled_turns) is not int or scheduled_turns < 1 or type(terminal) is not bool:
        raise ValueError('explicit positive schedule and actual terminal status required')
    if not trace or len(trace) > scheduled_turns:
        raise ValueError('nonempty bounded actual history required')
    if len(truth_schedule) != scheduled_turns:
        raise ValueError('full evaluator-only schedule required; do not drop early-ending cases')
    for q in truth_schedule:
        distribution(q)
    observations = []
    for tick, row in enumerate(trace):
        if row['tick'] != tick or row['before']['tick'] != tick or row['after']['tick'] != tick+1:
            raise ValueError('contiguous actual history required')
        if tick and row['before'] != trace[tick-1]['after']:
            raise ValueError('world history changed')
        p = distribution(row['response_prediction']); q = distribution(row['truth'])
        if q != truth_schedule[tick]:
            raise ValueError('registered response law changed')
        if set(p) != set(q) or row['response'] not in q:
            raise ValueError('same exhaustive response support required')
        delta = row['conditional_increment']
        if type(delta) not in (int, float) or not math.isfinite(delta):
            raise ValueError('finite conditional increment required')
        brier = math.fsum((p[k]-q[k])**2 for k in q)
        probability = p[row['response']]
        observations.append(dict(tick=tick, brier_excess=brier,
            observed_log_loss=None if probability == 0 else -math.log(probability),
            impossible_observation=probability == 0, conditional_increment=delta))
    starts = [0]+[t for t in range(1,scheduled_turns) if truth_schedule[t] != truth_schedule[t-1]]
    regimes = []
    for index, begin in enumerate(starts):
        planned_end = starts[index+1] if index+1 < len(starts) else scheduled_turns
        end = min(planned_end, len(trace))
        values = [r['brier_excess'] for r in observations[begin:end]]
        streak = 0; recovered = None
        for offset in range(WINDOW-1, len(values)):
            rolling = math.fsum(values[offset-WINDOW+1:offset+1])/WINDOW
            streak = streak+1 if rolling <= BRIER_LIMIT+EPS else 0
            if streak == 2:
                recovered = offset+1
                break
        closed = end == planned_end
        # A short episode may end before enough observations exist. It never passes.
        censored = not closed or len(values) < RECOVERY_LIMIT
        regimes.append(dict(start=begin,end=max(begin,end),planned_end=planned_end,
            observed_steps=len(values),kind='initial' if begin == 0 else 'change',
            recovered_in=recovered,within_16=not censored and recovered is not None and recovered <= RECOVERY_LIMIT,
            censored=censored,unrecovered=recovered is None,
            mean_brier_excess=None if not values else math.fsum(values)/len(values)))
    windows = []
    for begin in range(0,len(trace),WINDOW):
        rows = observations[begin:begin+WINDOW]
        windows.append(dict(start=begin,end=begin+len(rows),
            conditional_increment=math.fsum(r['conditional_increment'] for r in rows),
            mean_brier_excess=math.fsum(r['brier_excess'] for r in rows)/len(rows),
            impossible_observations=sum(r['impossible_observation'] for r in rows)))
    changes = [r for r in regimes if r['kind'] == 'change']
    eligible = [r for r in changes if r['planned_end']-r['start'] >= 32]
    return dict(actual_steps=len(trace),complete=terminal,regimes=regimes,windows=windows,
        qualifying_change_segments=len(eligible),recovered_within_16=sum(r['within_16'] for r in eligible),
        prediction_recovery_rate=None if not eligible else sum(r['within_16'] for r in eligible)/len(eligible),
        unrecovered_changes=sum(r['unrecovered'] for r in changes),
        censored_changes=sum(r['censored'] for r in changes),
        unobserved_changes=sum(r['observed_steps']==0 for r in changes),
        all_mean_brier_excess=math.fsum(r['brier_excess'] for r in observations)/len(observations),
        behavior_recovery='unmeasured: requires same-owner learned/frozen continuation controls',
        persona_long_term_regret='unmeasured: requires a same-persona settled reference')
