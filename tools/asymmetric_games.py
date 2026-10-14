"""Public asymmetric rule adapters, with no strong/weak labels in actor state.

Authored finite diagnostics, not commercial game replicas. Capabilities,
inventories and the global deadline are public rules. Opponent event probabilities
belong to the experiment, never this adapter. Personality and learner unchanged.
"""
import copy
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

from reflex.adaptive_dynamics import PublicEvent
from reflex.contests import Public, public_from_record, settle
from reflex.core import digest
from reflex.examples import effect
from reflex.purpose_plan import Goal
from tools.open_model_games import OpenGame


GAMES = ('auction', 'resources', 'combat', 'gambling')
CASES = ('balanced', 'poor', 'rich', 'limited', 'powerful', 'specialist',
         'late', 'late_observer')


def board(game, case='balanced', variant=0, deadline=4):
    """Experimental allocation only. Case name is NOT retained in the board."""
    if game not in GAMES or case not in CASES:
        raise ValueError('registered allocation required')
    w = dict(tick=0, episode=0, previous='none', signal=None, deadline=deadline,
             joined=not case.startswith('late'), routes=[True, True, True])
    resource = 2 if case == 'poor' else 10 if case == 'rich' else 6
    power = [1, 1, 1] if case == 'limited' else [3, 3, 3] if case == 'powerful' else [2, 2, 2]
    if case == 'specialist':
        power = [1, 1, 4]
        w['routes'] = [False, True, True]
    elif case == 'limited':
        w['routes'][2] = False
    if game == 'auction':
        prizes = (3, 6, 2, 5, 4, 7) if not variant else (5, 2, 7, 3, 6, 4)
        w['position'] = json.loads(json.dumps(asdict(Public(
            'auction', ((),)*3, (0,)*3, (resource, 6, 6), prizes[0], 'science', prizes[1:]))))
        w['bid_cap'] = 2 if case == 'limited' else 6 if case == 'powerful' else 4
        # Specialisation trades access for a lower price: even bids only,
        # winner pays one less. Public rule, included in every transition.
        w['even_only'] = case == 'specialist'
        w['discount'] = int(case == 'specialist')
    elif game == 'resources':
        w.update(cash=resource, plant=0, rival=6, yields=[p+2 for p in power], costs=[2]*3)
    elif game == 'combat':
        w.update(hp=6, enemy=6, energy=min(6, resource//2), focus=False,
                 damage=power, max_hp=6, energy_cap=6, armor=0)
    else:
        w.update(cash=resource, rival=6, payouts=power, stake_cap=2)
    return w


class AsymmetricGame(OpenGame):
    def __init__(self, game, profile, initial, *, variant=0):
        self.initial = copy.deepcopy(initial)
        super().__init__(game, profile, length=initial['deadline'], variant=variant)
        # All allocations share rules; their public state contains every
        # capability. The base learner identity must follow the new rules.
        self.model_id = digest(['asymmetric-public-v2', game, self.length, variant])
        from reflex.adaptive_dynamics import EventLearner, AdaptiveKernel
        learner = EventLearner(factors=('scene', 'previous'), names=('library-a', 'library-b'),
                               owner='owner', model_id=self.model_id, windows=(8, 32, 128))
        self.kernel = AdaptiveKernel(learner=learner, events=self.events, features=self.features,
            models=self.models, public_record=lambda w:w, model_id=self.model_id)

    def start(self, episode=0, memory=None):
        w = copy.deepcopy(self.initial)
        w['episode'] = episode
        return replace(self.kernel.start(w, memory), tick=w['tick'])

    def terminal(self, s):
        w = s.public
        return (w['tick'] >= w['deadline'] or
                (self.game == 'combat' and (w['hp'] <= 0 or w['enemy'] <= 0)))

    def keys(self, s):
        w = s.public
        if self.terminal(s):
            return ()
        if not w['joined']:
            return ('absent',)
        if self.game == 'auction':
            return tuple('bid:'+str(b) for b in range(min(w['bid_cap'], w['position']['budgets'][0])+1)
                         if not w['even_only'] or b % 2 == 0)
        if self.game == 'resources':
            return ('trade',) + tuple('crop:'+str(i) for i in range(3)
                if w['routes'][i] and w['cash'] >= w['costs'][i]) + (
                ('invest',) if w['cash'] >= 2 and w['plant'] < 2 else ())
        if self.game == 'combat':
            return ('guard',) + (('position',) if w['energy'] and not w['focus'] else ()) + tuple(
                'strike:'+str(i) for i in range(3) if w['energy'] and w['routes'][i])
        return ('pass',) + tuple('bet:'+str(i)+':'+str(stake) for i in range(3) if w['routes'][i]
            for stake in range(1, min(w['stake_cap'], w['cash'])+1))

    def quantities(self, w):
        if self.game == 'auction':
            p = w['position']
            return tuple(s+.25*b for s, b in zip(p['scores'], p['budgets']))
        if self.game == 'resources':
            return (w['cash']+2*w['plant'], w['rival'])
        if self.game == 'combat':
            return (w['hp'], w['enemy'])
        return (w['cash'], w['rival'])

    def goal(self, s):
        q = self.quantities(s.public)
        if self.game == 'combat':
            value = 0. if s.public['hp'] <= 0 else 1. if s.public['enemy'] <= 0 else .5
        else:
            value = 0. if q[0] < max(q) else 1./sum(x == max(q) for x in q)
        return Goal(self.target, 'scored' if self.terminal(s) else 'running', value)

    def events(self, w, key):
        if key not in self.keys(SimpleNamespace(public=w)):
            raise ValueError('unavailable action cannot be simulated')
        rows = []
        before = self.quantities(w)
        for symbol in self.symbols(w):
            z = int(symbol)
            after = copy.deepcopy(w)
            after.update(tick=w['tick']+1, signal=symbol, previous=key.split(':')[0])
            safe = style = 0.
            if self.game == 'auction':
                p = public_from_record(w['position'])
                bid = 0 if key == 'absent' else int(key.split(':')[1])
                nxt, winner = settle(p, (bid, z, min(1, p.budgets[2])))
                if winner == 0 and w['discount']:
                    cash = list(nxt.budgets)
                    cash[0] += min(bid, w['discount'])
                    nxt = replace(nxt, budgets=tuple(cash))
                if nxt.remaining:
                    nxt = replace(nxt, prize=nxt.remaining[0], remaining=nxt.remaining[1:])
                after['position'] = json.loads(json.dumps(asdict(nxt)))
                safe = -(p.budgets[0]-nxt.budgets[0])/12
                style = bid/max(1, p.budgets[0])*.2
            elif self.game == 'resources':
                after['rival'] += z
                if key == 'invest':
                    after['cash'] -= 2
                    after['plant'] += 1
                elif key == 'trade':
                    after['cash'] += 1
                elif key != 'absent':
                    i = int(key.split(':')[1])
                    after['cash'] += (w['yields'][i]+w['plant'] if i == z else 0)-w['costs'][i]
                safe = min(0., (after['cash']-w['cash'])/12)
                style = .2 if key == 'invest' else 0.
            elif self.game == 'combat':
                if key == 'absent':
                    # Before entry the opponent has time to fortify. No damage
                    # is charged to an agent that is not on the battlefield.
                    after['armor'] = min(2, w['armor']+int(z > 0))
                elif key == 'guard':
                    after['hp'] = min(w['max_hp'], w['hp']+1)
                    after['energy'] = min(w['energy_cap'], w['energy']+1)
                elif key == 'position':
                    after['energy'] -= 1
                    after['focus'] = True
                else:
                    i = int(key.split(':')[1])
                    after['energy'] -= 1
                    after['focus'] = False
                    if i == z or w['focus']:
                        after['enemy'] = max(0, w['enemy']-max(1, w['damage'][i]-w['armor']))
                    else:
                        after['hp'] = max(0, w['hp']-2)
                safe = (after['hp']-w['hp'])/6
                style = .2 if key.startswith('strike') else -.1
            else:
                # An opponent stakes one coin on the public current lane. A
                # late entrant does not inherit their cash or their decisions.
                stake = min(1, w['rival'])
                after['rival'] += stake*(2 if z == int(self.scene(w)) else -1)
                if key not in ('pass', 'absent'):
                    _, lane, size = key.split(':')
                    after['cash'] += int(size)*(w['payouts'][int(lane)] if int(lane) == z else -1)
                safe = min(0., (after['cash']-w['cash'])/12)
                style = .1 if key.startswith('bet') else 0.
            q = self.quantities(after)
            gain = ((q[0]-max(q[1:]))-(before[0]-max(before[1:])))/24
            flow = effect(gain, needs={'growth':gain, 'safety':safe},
                values={'achievement':gain, 'security':safe}, style={'openness':style})
            rows.append(PublicEvent(symbol, after, flow))
        return rows

