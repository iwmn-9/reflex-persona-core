"""Optional flat Monte Carlo terminal evaluation; no MCTS or shared-policy changes.

Adapter contract: terminal(s), chance(s), sample(s,rng), legal(s), step(s,a),
begin_trial(), choose(s,rng,policy), evaluate(s)->TerminalEvaluation.
Only the model's observed/hypothesized states may enter the adapter.
"""
from dataclasses import dataclass, asdict
import copy
import math
import numpy as np
from .core import digest
from .planning import compress_outcomes, vector


@dataclass(frozen=True)
class RolloutBudget:
    samples: int = 8
    max_nodes: int = 20000
    max_steps: int = 2048
    min_samples: int = 4
    rollout_policy: str = 'tactical'

    def __post_init__(self):
        for key,lo,hi in (('samples',1,256),('max_nodes',0,1000000),('max_steps',1,4096),('min_samples',1,256)):
            value=getattr(self,key)
            if type(value) is not int or not lo<=value<=hi: raise ValueError(f'{key} outside [{lo},{hi}]')
        if self.min_samples>self.samples: raise ValueError('min_samples exceeds requested samples')
        if self.rollout_policy not in ('random','tactical','persona'): raise ValueError('unknown rollout policy')


@dataclass(frozen=True)
class TerminalEvaluation:
    outcome: dict
    win_share: float
    won: bool
    draw: bool
    game_score: float | None = None


def wilson(successes,n):
    """95% binomial sampling interval, not model accuracy or real-match confidence."""
    if not n: return None
    z=1.959963984540054; p=successes/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d; radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,center-radius),min(1.,center+radius)]


def _summary(samples,exact):
    n=len(samples)
    if not n: return dict(samples=0,exact=exact,win_rate=None,win_share=None,draw_rate=None,
                         mean_return=None,return_se=None,win_rate_interval95=None,mean_score=None)
    rewards=np.array([s.outcome['objective'] for s in samples],dtype=float)
    wins=sum(s.won for s in samples); scores=[s.game_score for s in samples if s.game_score is not None]
    return dict(samples=n,exact=exact,win_rate=wins/n,win_share=sum(s.win_share for s in samples)/n,
                draw_rate=sum(s.draw for s in samples)/n,mean_return=float(rewards.mean()),
                return_se=0. if exact else float(rewards.std(ddof=1)/math.sqrt(n)) if n>1 else None,
                win_rate_interval95=[wins/n,wins/n] if exact else wilson(wins,n),
                mean_score=float(np.mean(scores)) if scores else None)


def evaluate_actions(roots,adapter,budget,scope):
    """Equal, completed sample rounds; discard an incomplete round for every root.

    max_nodes counts additional state transitions, including discarded samples,
    excluding root construction, adapter choice/evaluation work and wall time.
    A node/length limit produces no invented terminal reward. Below min_samples,
    used=False instructs the caller to retain its ordinary immediate evaluation.
    Separate chance and action streams use common trial seeds across root actions.
    """
    if not roots: raise ValueError('root candidates required')
    names=sorted(roots); collected={a:[] for a in names}
    stats=dict(config=asdict(budget),root_candidates=len(names),additional_nodes=0,
               completed_samples=0,discarded_terminal_samples=0,budget_exhausted=False,
               length_exhausted=False,used=False,actions={})

    class Cut(Exception): pass
    def debit():
        if stats['additional_nodes']>=budget.max_nodes:
            stats['budget_exhausted']=True; raise Cut()
        stats['additional_nodes']+=1

    for trial in range(budget.samples):
        packet={}
        try:
            for name in names:
                state=roots[name]; adapter.begin_trial()
                # Model sampling never consumes the real game's RNG/future deck.
                trial_seed=int(digest([scope,'flat-monte-carlo',trial])[:16],16)
                chance_rng=np.random.default_rng(trial_seed)
                action_rng=np.random.default_rng(trial_seed ^ 0xD1B54A32D192ED03)
                steps=0
                while not adapter.terminal(state):
                    if steps>=budget.max_steps:
                        stats['length_exhausted']=True; raise Cut()
                    debit()
                    if adapter.chance(state): state=adapter.sample(state,chance_rng)
                    else:
                        move=adapter.choose(state,action_rng,budget.rollout_policy)
                        if move not in adapter.legal(state): raise ValueError('rollout chose illegal action')
                        state=adapter.step(state,move)
                    steps+=1
                result=adapter.evaluate(state)
                if not isinstance(result,TerminalEvaluation): raise ValueError('terminal evaluation required')
                if type(result.won) is not bool or type(result.draw) is not bool or not 0<=result.win_share<=1:
                    raise ValueError('invalid win/draw result')
                if not np.all(np.isfinite(vector(result.outcome))) or not -1<=result.outcome['objective']<=1:
                    raise ValueError('invalid terminal effects')
                if result.game_score is not None and not np.isfinite(result.game_score): raise ValueError('invalid game score')
                packet[name]=result
        except Cut:
            stats['discarded_terminal_samples']+=len(packet); break
        for name in names: collected[name].append(packet[name])
        stats['completed_samples']+=1

    stats['used']=stats['completed_samples']>=budget.min_samples
    outcomes={}
    for name in names:
        samples=collected[name]; stats['actions'][name]=_summary(samples,adapter.terminal(roots[name]))
        if not stats['used']: continue
        # Identical numeric outcomes are exact frequency aggregation, then the
        # shared 8-outcome contract may approximate within-bucket downside.
        groups={}
        for sample in samples:
            row=copy.deepcopy(sample.outcome); row.pop('p',None); key=digest(row)
            if key not in groups: groups[key]=[row,0]
            groups[key][1]+=1
        rows=[dict(row,p=count/len(samples)) for row,count in groups.values()]
        outcomes[name]=compress_outcomes(rows)
    stats['compressed_actions']=sum(len({digest(s.outcome) for s in collected[a]})>8 for a in names)
    return outcomes,stats
