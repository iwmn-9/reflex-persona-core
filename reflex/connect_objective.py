"""Public full-width finite minimax adversary, not a solved-game oracle."""
from functools import lru_cache
from .board_models import connect_features


@lru_cache(maxsize=100000)
def value(s,depth,side):
    winner=s.winner()
    if winner is not None:return (1.+.001*depth)*(1 if winner==side else -1)
    if sum(s.heights)==42:return 0.
    if depth==0:return connect_features(s,side)[0]
    values=[value(s.play(a),depth-1,side) for a in s.legal()]
    return (max if s.turn==side else min)(values)


def choose(s,rng,depth=4):
    if type(depth) is not int or not 1<=depth<=5:raise ValueError('bounded public minimax depth required')
    names=s.legal()
    if not names:raise ValueError('live public decision required')
    values=[value(s.play(a),depth-1,s.turn) for a in names];best=max(values)
    ties=[a for a,v in zip(names,values) if abs(v-best)<1e-12]
    return ties[int(rng.integers(len(ties)))]
