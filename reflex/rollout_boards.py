"""Public-information tabletop models for comparison-only terminal rollouts."""
import copy
import numpy as np
from .core import Policy
from .examples import action
from .board_models import ConnectAdapter, ThanksAdapter, neutral, card_points
from .monte_carlo import TerminalEvaluation, evaluate_actions


def connect_tactical(state,rng):
    """Win now, avoid an immediate legal loss if possible, otherwise random.

    No positional evaluation/real opponent personality is used in this baseline.
    """
    names=state.legal(); side=state.turn
    def would_win(bits,col,height):
        if height>=6: return False
        b=bits | (1<<(7*col+height))
        return any((pair:=(b & (b>>d))) & (pair>>(2*d)) for d in (1,7,6,8))
    columns=[int(a.split(':')[1]) for a in names]
    wins_now=[col for col in columns if would_win(state.discs[side],col,state.heights[col])]
    threats={col for col in columns if would_win(state.discs[1-side],col,state.heights[col])}
    wins=[f'DROP:{col}' for col in wins_now]
    # Our drop only changes the next available height in its own column.
    safe=[f'DROP:{col}' for col in columns if not (threats-{col}) and
          not would_win(state.discs[1-side],col,state.heights[col]+1)]
    options=wins or safe or list(names)
    return options[int(rng.integers(len(options)))]


def thanks_tactical(state,rng):
    """Myopic own-score improvement, forced take, otherwise refuse."""
    if 'PASS' not in state.legal(): return 'TAKE'
    added=card_points(state.cards[state.turn]+(state.card,))-card_points(state.cards[state.turn])
    return 'TAKE' if added<=state.pot else 'PASS'


class BoardRolloutModel:
    """Receives only public base state and one known persona, never a real roster/deck."""
    def __init__(self,adapter,base,profile,seed,tick,episode,policy_state=None):
        self.adapter=adapter; self.base=base; self.viewer=base.turn
        self.profile=copy.deepcopy(profile); self.memory=copy.deepcopy(policy_state)
        self.seed=seed; self.tick=tick; self.episode=episode; self.policy=Policy()

    def __getattr__(self,name): return getattr(self.adapter,name)

    def begin_trial(self):
        self.memories={self.viewer:copy.deepcopy(self.memory)}; self.turns=0

    def sample(self,state,rng):
        # Uniform next card conditional on public reveals: integrates unknown
        # nine removals without ever using the driver's real remaining deck.
        available=[c for c in range(3,36) if c not in state.seen]
        return state.draw(available[int(rng.integers(len(available)))])

    def choose(self,state,rng,policy):
        if policy=='random':
            names=self.legal(state); return names[int(rng.integers(len(names)))]
        if policy=='tactical':
            return connect_tactical(state,rng) if self.adapter.game=='connect_four' else thanks_tactical(state,rng)
        # Optional expensive continuation follows the known own persona; other
        # players use neutral assumed priorities and independent fictional memory.
        profile=self.profile if state.turn==self.viewer else neutral()
        acts=[action(a,self.adapter.consequence(state,self.step(state,a),state.turn)) for a in self.legal(state)]
        c=self.adapter.context(state,profile,self.seed,min(2**63-1,self.tick+self.turns+1),self.episode,
                               acts,self.memories.get(state.turn))
        d=self.policy.choose(c,stochastic=False)
        self.memories[state.turn]=d['next_state']; self.turns+=1
        return d['action_id']

    def evaluate(self,state):
        if not self.terminal(state): raise ValueError('unfinished game cannot supply a terminal reward')
        row=self.adapter.consequence(self.base,state,self.viewer)
        if self.adapter.game=='connect_four':
            winner=state.winner(); draw=winner is None; won=winner==self.viewer
            share=.5 if draw else float(won); score=None
        else:
            scores=state.scores(); winners=[i for i,s in enumerate(scores) if s==min(scores)]
            won=self.viewer in winners; share=1/len(winners) if won else 0.; draw=False
            score=float(scores[self.viewer])
        # Game-specific terminal objective, explicit fractional credit for tied
        # winners. Preserve other need/value/style/cost meanings from the adapter.
        row['objective']=2*share-1
        row['values']['achievement']=row['objective']
        if self.adapter.game=='connect_four':
            row['values']['power']=row['objective']; row['values']['self_direction']=row['objective']*.5
        return TerminalEvaluation(row,float(share),bool(won),bool(draw),score)


def observe_rollouts(adapter,state,profile,budget,seed,tick,episode,policy_state=None):
    """All roots retained; below the completed-sample floor use reflex unchanged."""
    names=adapter.legal(state)
    if not names: raise ValueError('no decision in terminal/chance state')
    roots={a:adapter.step(state,a) for a in names}
    model=BoardRolloutModel(adapter,state,profile,seed,tick,episode,policy_state)
    packed,stats=evaluate_actions(roots,model,budget,[seed,adapter.game,episode,f'player-{state.turn}',tick])
    acts=[action(a,*(packed[a] if stats['used'] else [dict(adapter.consequence(state,roots[a],state.turn),p=1.)])) for a in names]
    for a in acts: a['target']=adapter.target(state,a['id'])
    c=adapter.context(state,profile,seed,tick,episode,acts,policy_state)
    if stats['used']:
        c['facts']['forecast_horizon']=f"終局までの仮想対戦、各候補{stats['completed_samples']}標本。実勝率ではない"
        c['facts']['rollout_assumption']=budget.rollout_policy+'。相手の実際の人格と本当の未来の山札は入力しない'
    else:
        c['facts']['forecast_horizon']='1手までの仮説。即時の規則効果と将来の推定を区別する'
    stats.update(baseline_transitions=len(roots),goal='terminal win share; No Thanks raw score also measured',
                 opponent_assumption=budget.rollout_policy+' model, not the actual hidden persona',
                 limited_precision=stats['completed_samples']<32)
    return c,stats
