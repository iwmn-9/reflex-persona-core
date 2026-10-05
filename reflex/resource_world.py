"""Headless four-player economic stress test, not a Civilization implementation.

All stocks/production/board positions are public. Rules and economic evaluation
are game-owned. The shared personality policy has no game-specific changes.
Short forecasts are conditional solo continuations: opponents do not act in the
model, current announced yields persist, and no future event schedule is input.
"""
from dataclasses import dataclass, replace, asdict
import copy
import numpy as np
from .core import NEEDS, TRAITS, Policy, digest
from .examples import action, context, effect

RESOURCES = ('food', 'wood', 'ore', 'gold')
BUILDINGS = ('farm', 'forest', 'mine', 'market', 'lab', 'temple')
ROUTES = ('science', 'culture', 'power')
COSTS = {
    'farm': (0, 3, 0, 1), 'forest': (0, 2, 1, 1),
    'mine': (0, 3, 0, 2), 'market': (0, 3, 1, 2),
    'lab': (0, 3, 2, 3), 'temple': (0, 3, 1, 3),
}


@dataclass(frozen=True)
class Empire:
    stock: tuple = (5, 4, 3, 4)
    buildings: tuple = (1, 1, 1, 0, 0, 0)
    science: int = 0
    culture: int = 0
    army: int = 0
    tech: int = 0
    monuments: int = 0
    land: int = 1
    shortages: int = 0
    overflow: int = 0


@dataclass(frozen=True)
class World:
    empires: tuple
    turn: int = 0
    round: int = 0
    frontier: int = 8
    food_yield: int = 2
    ore_yield: int = 1
    trade_price: int = 3
    routes: tuple = ROUTES
    limit: int = 36

    @classmethod
    def start(cls, players=4, routes=ROUTES, limit=36):
        if type(players) is not int or not 2 <= players <= 6:
            raise ValueError('2..6 players required')
        if not routes or len(set(routes)) != len(routes) or set(routes)-set(ROUTES):
            raise ValueError('distinct known victory routes required')
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('bounded round limit required')
        return cls((Empire(),)*players, frontier=2*players, routes=tuple(routes), limit=limit)


def progress(e):
    """Actual threshold proximity, NOT a probability of winning."""
    return (min(e.tech/3, 1)*.5+min(e.science/14, 1)*.5,
            min(e.monuments/2, 1)*.5+min(e.culture/20, 1)*.5,
            min(e.land/4, 1)*.5+min(e.army/6, 1)*.5)


def achieved(e):
    return tuple(r for r, ok in zip(ROUTES, (
        e.tech >= 3 and e.science >= 14,
        e.monuments >= 2 and e.culture >= 20,
        e.land >= 4 and e.army >= 6)) if ok)


def winners(w):
    return tuple(i for i, e in enumerate(w.empires) if set(achieved(e)) & set(w.routes))


def terminal(w):
    return bool(winners(w)) or w.round >= w.limit


def affordable(e, cost):
    return all(a >= b for a, b in zip(e.stock, cost))


def pay(e, cost, **changes):
    if not affordable(e, cost):
        raise ValueError('cannot pay')
    return replace(e, stock=tuple(a-b for a, b in zip(e.stock, cost)), **changes)


def income(e, w):
    """Production, food/army maintenance, then bounded storage. No free deficit."""
    farm, forest, mine, market, lab, temple = e.buildings
    stock = [a+b for a, b in zip(e.stock, (
        farm*w.food_yield, forest, mine*w.ore_yield, 1+2*market))]
    food_short = max(0, e.land-stock[0])
    stock[0] = max(0, stock[0]-e.land)
    upkeep = (e.army+3)//4
    gold_short = max(0, upkeep-stock[3])
    stock[3] = max(0, stock[3]-upkeep)
    lost = sum(max(0, x-18) for x in stock)
    return replace(e, stock=tuple(min(18, x) for x in stock),
        science=min(40, e.science+2*lab), culture=min(40, e.culture+2*temple),
        army=max(0, e.army-food_short-gold_short),
        shortages=e.shortages+food_short+gold_short, overflow=e.overflow+lost)


def legal(w):
    if terminal(w):
        return ()
    e = w.empires[w.turn]
    choices = ['wait']+[f'gather:{r}' for r in RESOURCES]
    for name, cost in COSTS.items():
        slot = BUILDINGS.index(name)
        if e.buildings[slot] < 3 and affordable(e, cost):
            choices.append('build:'+name)
    if e.tech < 3 and e.science >= 6+4*e.tech and affordable(e, (0, 0, 1+e.tech, 3)):
        choices.append('research')
    if e.monuments < 2 and e.culture >= 8 and affordable(e, (0, 3, 0, 5)):
        choices.append('monument')
    if e.army <= 10 and affordable(e, (3, 0, 3, 2)):
        choices.append('train')
    if w.frontier and affordable(e, (3, 3, 0, 2)):
        choices.append('settle')
    if w.frontier and e.army >= 3 and affordable(e, (2, 0, 1, 1)):
        choices.append('claim')
    for j, rival in enumerate(w.empires):
        if j != w.turn and rival.land > 1 and e.army >= rival.army+3 and affordable(e, (3, 0, 2, 1)):
            choices.append('invade:'+str(j))
    for j, r in enumerate(RESOURCES[:3]):
        if e.stock[j] >= 2:
            choices.append('sell:'+r)
    return tuple(sorted(choices))


def act(w, key):
    """Exact action transition, before end-of-round production."""
    if key not in legal(w):
        raise ValueError('illegal game action')
    actor = w.turn
    e = w.empires[actor]
    roster = list(w.empires)
    frontier = w.frontier
    if key.startswith('gather:'):
        j = RESOURCES.index(key.split(':')[1]); stock = list(e.stock)
        stock[j] += 3
        excess = max(0, stock[j]-18); stock[j] = min(18, stock[j])
        e = replace(e, stock=tuple(stock), overflow=e.overflow+excess)
    elif key.startswith('build:'):
        name = key.split(':')[1]; b = list(e.buildings); b[BUILDINGS.index(name)] += 1
        e = pay(e, COSTS[name], buildings=tuple(b))
    elif key == 'research':
        e = pay(e, (0, 0, 1+e.tech, 3), tech=e.tech+1, science=e.science-(6+4*e.tech))
    elif key == 'monument':
        e = pay(e, (0, 3, 0, 5), monuments=e.monuments+1, culture=e.culture-8)
    elif key == 'train':
        e = pay(e, (3, 0, 3, 2), army=e.army+2)
    elif key in ('settle', 'claim'):
        e = pay(e, (3, 3, 0, 2) if key == 'settle' else (2, 0, 1, 1), land=e.land+1)
        frontier -= 1
    elif key.startswith('invade:'):
        target = int(key.split(':')[1])
        e = pay(e, (3, 0, 2, 1), army=e.army-1, land=e.land+1)
        roster[target] = replace(roster[target], land=roster[target].land-1)
    elif key.startswith('sell:'):
        j = RESOURCES.index(key.split(':')[1]); stock = list(e.stock)
        stock[j] -= 2; stock[3] += w.trade_price
        excess = max(0, stock[3]-18); stock[3] = min(18, stock[3])
        e = replace(e, stock=tuple(stock), overflow=e.overflow+excess)
    roster[actor] = e
    return replace(w, empires=tuple(roster), frontier=frontier)


def step(w, key):
    after = act(w, key)
    # Victory is checked immediately; no production can revoke an achieved win.
    if winners(after):
        return after
    turn = (w.turn+1) % len(w.empires)
    if turn == 0:
        after = replace(after, empires=tuple(income(e, after) for e in after.empires), round=w.round+1)
    return replace(after, turn=turn)


def goal_progress(e, routes):
    return max(progress(e)[ROUTES.index(r)] for r in routes)


def potential(w, viewer):
    """Explicit game-authored economic proxy; not learned or calibrated EV.

    Producing infrastructure has a small proxy value so immediate play can begin
    a prerequisite chain. This makes the rules-only/general-value limitation
    visible rather than concealing an economic strategy in the shared core.
    """
    e = w.empires[viewer]
    score = goal_progress(e, w.routes)
    if set(achieved(e)) & set(w.routes):
        return 1.
    productive = sum(min(x, 2) for x in e.buildings)/12
    liquidity = sum(e.stock)/72
    # Reserve terminal reward: abundant resources/near-completion are not a win.
    return float(.6*(.78*score+.14*productive+.08*liquidity))


def deficits(e, w):
    food_flow = e.buildings[0]*w.food_yield-e.land
    food = np.clip((3-e.stock[0]-2*min(food_flow, 0))/6, 0, 1)
    finance = np.clip(((e.army+3)//4-e.stock[3])/5, 0, 1)
    return {'physiology':float(food), 'safety':float(max(food, finance)),
            'growth':float(.55*(1-goal_progress(e, w.routes)))}


def consequence(base, after, viewer):
    old, new = base.empires[viewer], after.empires[viewer]
    before_needs, after_needs = deficits(old, base), deficits(new, after)
    g = potential(after, viewer)-potential(base, viewer)
    # Research/culture stock is a SPENDABLE project budget, not lost ability.
    # Completed technology/infrastructure remain capability after paying it.
    knowledge = (new.tech-old.tech)/3+(new.buildings[4]-old.buildings[4])/6
    culture = (new.monuments-old.monuments)/2+(new.buildings[5]-old.buildings[5])/6
    military = ((new.land-old.land)/4+(new.army-old.army)/12)/2
    finances = (new.stock[3]-old.stock[3])/18
    supply = (new.stock[0]-old.stock[0])/18
    spending = sum(max(0, a-b) for a, b in zip(old.stock, new.stock))/72
    # All values are proxies on OWN development, with no charity reward for an
    # opponent's race progress. Whole-board interaction remains in exact rules.
    return effect(g, needs={**{n:before_needs[n]-after_needs[n] for n in before_needs},
        'growth':float(np.clip(max(knowledge, culture, g), -1, 1))},
        values={'achievement':g, 'self_direction':float(knowledge),
                # Security means avoided food/upkeep deficit, not an intrinsic
                # reward for hoarding already sufficient capped resources.
                'security':float(before_needs['safety']-after_needs['safety']), 'power':float(military),
                'tradition':float(culture)},
        style={'openness':float(knowledge), 'conscientiousness':float(-spending)},
        cost=float(min(1, spending)))


def forecast(base, key, horizon):
    """Bounded solo model with generic proxy-greedy continuation.

    Every root gets the SAME horizon. Each subsequent model decision retains all
    its legal actions. No real future random values, event time, or rival policy
    enters this function. The greedy modeled continuation is not the actor's
    personality and is not evidence of a correct multiplayer value function.
    """
    if type(horizon) is not int or not 1 <= horizon <= 4:
        raise ValueError('forecast horizon must be 1..4')
    viewer = base.turn
    model = act(base, key); transitions = 1
    effective = min(horizon, base.limit-base.round)
    for depth in range(effective):
        if winners(model):
            break
        own = income(model.empires[viewer], model)
        roster = list(model.empires); roster[viewer] = own
        model = replace(model, empires=tuple(roster), round=model.round+1)
        if depth == effective-1 or terminal(model):
            break
        options = legal(model)
        if not options:
            break
        successors = [(name, act(model, name)) for name in options]
        transitions += len(successors)
        _, model = max(successors, key=lambda pair:(potential(pair[1], viewer), pair[0]))
    return model, transitions


def make_context(w, profile, seed, tick, episode, memory=None, horizon=1):
    viewer = w.turn
    choices = []; transitions = 0
    for key in legal(w):
        after, nodes = forecast(w, key, horizon)
        transitions += nodes
        immediate = act(w,key)
        exact = consequence(w,immediate,viewer)
        projected = consequence(w,after,viewer)
        # Root identity/commitment and persona alignment belong to the selected
        # action, not to unrelated greedy actions in the future model. Only goal
        # and supply projections receive bounded future credit. Weight .65 is
        # an explicit heuristic, not a probability or validated learning gate.
        exact['objective'] += .65*(projected['objective']-exact['objective'])
        for n in ('physiology','safety'):
            exact['needs'][n] += .65*(projected['needs'][n]-exact['needs'][n])
        exact['needs']['growth'] = float(max(exact['values']['self_direction'],
            exact['values']['tradition'],exact['objective']))
        choices.append(action(key,exact))
    c = context(episode, choices, deficits(w.empires[viewer], w), profile['values'],
                dict(zip(TRAITS, profile['traits'])), mode=None)
    for n in NEEDS:
        if n not in deficits(w.empires[viewer], w):
            c['needs'][n] = dict(supported=False, enabled=False, deficit=None)
    c['scope'] = dict(game='resource-world-v1', episode=episode, npc=f'player-{viewer}')
    public = {'public_round':str(w.round), 'public_turn':str(w.turn),
              'public_frontier':str(w.frontier), 'public_routes':str(w.routes),
              'public_food_yield':str(w.food_yield), 'public_ore_yield':str(w.ore_yield),
              'public_trade_price':str(w.trade_price), 'public_limit':str(w.limit)}
    for i, e in enumerate(w.empires):
        for k, v in asdict(e).items():
            public[f'public_player_{i}_{k}'] = str(v)
    c.update(seed=seed, tick=tick, objective='本人が科学・文化・勢力の有効な勝利条件を達成して先着する',
        facts={**public,
            'victory':'science: tech>=3 and science>=14; culture: monuments>=2 and culture>=20; power: land>=4 and army>=6',
            'valuation':'game proxy; root persona effects/cost exact; future goal/supply credit .65, NOT a probability',
            'forecast':'solo greedy continuations, rivals frozen, current announced yields persist; horizon='+str(horizon)})
    if memory is not None:
        c['state'] = copy.deepcopy(memory)
    return c, dict(horizon=horizon, model_transitions=transitions,
                   candidates=len(choices), opponent_actions_modeled=False)


def economic_move(w, horizon=4):
    """Deterministic game-authored comparator; does not share the persona scorer."""
    return max(legal(w), key=lambda key:(potential(forecast(w, key, horizon)[0], w.turn), key))


def world_record(w):
    return asdict(w)


def world_from_record(record):
    fields = set(World.__dataclass_fields__)
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError('complete world record required')
    empires = tuple(Empire(**{**e, 'stock':tuple(e['stock']), 'buildings':tuple(e['buildings'])})
                    for e in record['empires'])
    return World(**{**record, 'empires':empires, 'routes':tuple(record['routes'])})
