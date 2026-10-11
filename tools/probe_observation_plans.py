"""Exact capacity diagnostic, NOT a new production controller or strength trial.

Compare fixed scripts, observation-history policies, and a forbidden world oracle.
Finite public model, no fitted coefficients, sampled real future, or new utility.
The conditional candidate uses the existing flow callback and persona selector.
"""
import argparse
import copy
from dataclasses import dataclass, replace
from itertools import product
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reflex.core import Policy, TRAITS, digest
from reflex.examples import context, action, effect
from reflex.laboratory import profiles
from reflex.flow_rollout import rollout
from reflex.purpose_plan import Goal, goal_forecast
from reflex.continuation_search import search_forecast
from reflex.deliberation import select


@dataclass(frozen=True)
class State:
    tick: int = 0
    root: str | None = None
    world: int | None = None
    choice: str | None = None


class Model:
    """Opportunity known only after commitment; its effect pays out later.

    All decisions remain legal in both worlds. Reflex chooses an immediate
    small payout; investment only pays at settlement. The opaque IDs deliberately
    offer no generic action heuristic. Exact model is a diagnostic assumption.
    """
    ids = ('a0', 'a1', 'b0', 'b1', 'b2', 'n0')

    def __init__(self, horizon, reveal, visible, outside, rename=False):
        self.horizon, self.reveal = horizon, reveal
        self.visible, self.outside = visible, outside
        self.names = dict(zip(self.ids, reversed(self.ids))) if rename else dict(zip(self.ids, self.ids))
        self.target = 'settled-opportunity-return'

    def terminal(self, s): return s.tick == self.horizon

    def legal(self, s):
        if self.terminal(s): return ()
        if s.tick == 0: return tuple(self.names[k] for k in ('a0', 'a1'))
        if s.root == self.names['a1'] and s.tick == self.reveal:
            return tuple(self.names[k] for k in ('b0', 'b1', 'b2'))
        return (self.names['n0'],)

    def public(self, s):
        signal = s.world if self.visible and s.tick >= self.reveal else None
        return (s.tick, s.root, signal, s.choice, self.legal(s))

    def step(self, s, key):
        if key not in self.legal(s): raise ValueError('illegal modeled move')
        return replace(s, tick=s.tick+1, root=key if s.tick == 0 else s.root,
                       choice=key if s.tick == self.reveal and s.root == self.names['a1'] else s.choice)

    def value(self, s):
        if not self.terminal(s): raise ValueError('no cutoff proxy in exact diagnostic')
        if s.root == self.names['a0']: return self.outside
        if s.choice == self.names['b2']: return .2
        return float(s.choice == self.names['b'+str(s.world)])

    def goal(self, s): return Goal(self.target, 'scored', self.value(s)).record()


def observation_solution(model, support, history=()):
    """Exact expectation with one COMMON action per observable history.

    Private modeled states can propagate probability, but never key decisions.
    No assumptions about actor psychology: this is objective-only capacity.
    """
    if all(model.terminal(s) for _, s in support):
        return sum(p*model.value(s) for p, s in support), {}
    groups = {}
    for p, s in support: groups.setdefault(model.public(s), []).append((p, s))
    value, policy = 0., {}
    for obs, group in groups.items():
        mass = sum(p for p, _ in group)
        normalized = [(p/mass, s) for p, s in group]
        node = history+(('observe', obs),)
        candidates = []
        for key in obs[-1]:
            score, tail = observation_solution(model, [(p, model.step(s,key)) for p,s in normalized],
                                               node+(('act',key),))
            candidates.append((score,key,tail))
        score,key,tail = max(candidates,key=lambda row:(row[0],row[1]))
        value += mass*score
        policy[node] = key
        policy.update(tail)
    return value,policy


def script_solution(model, support):
    """Exact best common script, stronger than any bounded script search here."""
    if all(model.terminal(s) for _,s in support):
        return sum(p*model.value(s) for p,s in support), ()
    legal = set.intersection(*(set(model.legal(s)) for _,s in support))
    candidates = []
    for key in sorted(legal):
        score,tail = script_solution(model,[(p,model.step(s,key)) for p,s in support])
        candidates.append((score,(key,*tail)))
    if not candidates: raise ValueError('diagnostic requires common legal script')
    return max(candidates)


def oracle_value(model, s):
    if model.terminal(s): return model.value(s)
    return max(oracle_value(model,model.step(s,key)) for key in model.legal(s))


def independent_check(model, support):
    """Independent forward enumeration at the one genuine decision.

    Includes hidden worlds in the SAME observation group. Does not call either
    optimizer; enumerates every mapping then performs whole trajectories.
    """
    decision = []
    for p,s in support:
        while s.tick < model.reveal: s = model.step(s,model.names['n0'])
        decision.append((p,s))
    observations = sorted({model.public(s) for _,s in decision},key=repr)
    keys = tuple(model.names[k] for k in ('b0','b1','b2'))
    scored = []
    for choices in product(keys,repeat=len(observations)):
        policy = dict(zip(observations,choices)); total = 0.
        for p,s in decision:
            s = model.step(s,policy[model.public(s)])
            while not model.terminal(s): s = model.step(s,model.names['n0'])
            total += p*model.value(s)
        scored.append(total)
    return max(scored),len(scored)


def run_case(model, profile):
    start = State()
    c = context('observation-capacity', [action(k,effect(.1 if k==model.names['a0'] else 0))
                                        for k in model.legal(start)],
                needs={'physiology':0.,'safety':0.},values=profile['values'],
                traits=dict(zip(TRAITS,profile['traits'])))
    c['scope']['npc'] = profile['id']
    original = digest(c)
    actor=Policy(principle_priority='finite')

    def observe(s,memory):
        cc = copy.deepcopy(c); cc['tick']=s.tick; cc['state']=memory
        cc['facts'] = {'public_observation':json.dumps(model.public(s),separators=(',',':'))}
        cc['actions'] = [action(k,effect(.2 if k==model.names['b2'] else -.05
                            if k in (model.names['b0'],model.names['b1']) else 0)) for k in model.legal(s)]
        return cc

    def advance(s,key,seed):
        if s.tick==0: s=replace(s,world=seed%2)
        cc=observe(s,copy.deepcopy(c['state']))
        flow=next(a['outcomes'][0] for a in cc['actions'] if a['id']==key) if s.tick else effect()
        return model.step(s,key),flow

    exact, policies, scripts, forbidden = {}, {}, {}, {}
    for root in model.legal(start):
        support=[(.5,model.step(State(world=world),root)) for world in range(2)]
        exact[root],policies[root]=observation_solution(model,support)
        script_value,scripts[root]=script_solution(model,support)
        forbidden[root]=sum(p*oracle_value(model,s) for p,s in support)
        if root==model.names['a1']:
            checked,count=independent_check(model,support)
            assert abs(checked-exact[root])<1e-12
            assert abs(script_value-.5)<1e-12

    # Existing search, identical horizon and eight balanced model worlds.
    seeds=tuple(range(8))
    fixed=search_forecast(c,start,observe=observe,advance=advance,terminal=model.terminal,
        assess=model.goal,horizon=model.horizon,seeds=seeds,width=2,depth=2,
        target=model.target,max_regret=0.,settlement_weight='absolute',policy=actor)
    fixed_decision,fixed_vote=select([c],fixed,actor)

    # Existing conditional callback, frozen observation policy (not world ID).
    paths,audit={},{};trace_public=[]
    for root in model.legal(start):
        history=()
        def policy_callback(s,cc):
            nonlocal history
            # The world-bearing s argument is deliberately unused: both history
            # reconstruction and policy lookup use only projected public inputs.
            obs=tupleize(json.loads(cc['facts']['public_observation']))
            if obs[0]==1: history=()  # existing rollout starts each detached branch at tick 1
            node=history+(('observe',obs),)
            key=policies[root][node]
            history=node+(('act',key),)
            trace_public.append((root,obs,key))
            return key
        p,a=rollout(c,start,observe=observe,advance=advance,terminal=model.terminal,
            assess=model.goal,horizon=model.horizon,seeds=seeds,roots=(root,),
            continuation=policy_callback,record_choices=True,policy=actor)
        paths[root],audit[root]=p[root],a[root]
    conditional=goal_forecast(c,paths,audit,horizon=model.horizon,unit='public-turns',
        target=model.target,max_regret=0.,settlement_weight='absolute',policy=actor)
    choice,vote=select([c],conditional,actor)
    assert digest(c)==original
    assert abs(vote['selected_purpose']-max(exact.values()))<1e-12
    seen={}
    for root,obs,key in trace_public:
        assert seen.setdefault((root,obs),key)==key
    # Compare real complete transitions for BOTH possible worlds, not just a
    # forecast score. Each actual future reuses the same public policy.
    actual=[]
    for world in range(2):
        root=choice[0]['action_id']; s=model.step(State(world=world),root)
        history=()
        while not model.terminal(s):
            node=history+(('observe',model.public(s)),)
            key=policies[root][node]; history=node+(('act',key),)
            s=model.step(s,key)
        actual.append(model.value(s))
    assert abs(sum(actual)/2-vote['selected_purpose'])<1e-12
    return dict(horizon=model.horizon,reveal=model.reveal,visible=model.visible,outside=model.outside,
        profile=profile['id'],renamed=model.names!=dict(zip(model.ids,model.ids)),
        exact_common_script_commit=.5,exact_observation_commit=exact[model.names['a1']],
        forbidden_oracle_commit=forbidden[model.names['a1']],
        enumerated_maps=count,fixed_root=fixed_decision[0]['action_id'],
        fixed_goal=fixed_vote['selected_purpose'],conditional_root=choice[0]['action_id'],
        conditional_goal=vote['selected_purpose'],actual_complete_worlds=actual,
        conditional_nodes=sum(len(p) for p in policies.values()))


def tupleize(value):
    return tuple(tupleize(v) for v in value) if isinstance(value,(tuple,list)) else value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    settings=[(h,r,v,o) for h,r in ((3,1),(5,1),(5,3),(8,1),(8,6))
              for v in (False,True) for o in (.6,.75,.9)]
    prereg=dict(kind='exact mechanism diagnostic, not independent match sample',
        settings=settings,profiles=[p['id'] for p in profiles()],
        hypotheses=['fixed script commit value .5 independent of horizon',
                    'observed conditional commit 1; hidden conditional commit .5',
                    'world oracle 1 in BOTH variants, forbidden in real selection',
                    'opaque action renaming preserves objective values'],
        source_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'preregister.json').write_text(json.dumps(prereg,indent=2),encoding='utf-8')
    rows=[]
    for h,r,v,o in settings:
        for p in profiles(): rows.append(run_case(Model(h,r,v,o),p))
    renamed=[run_case(Model(5,3,v,.75,True),p) for v in (False,True) for p in profiles()]
    for row in renamed:
        old=next(x for x in rows if all(x[k]==row[k] for k in ('horizon','reveal','visible','outside','profile')))
        for k in ('fixed_goal','conditional_goal','exact_observation_commit','forbidden_oracle_commit'):
            assert abs(row[k]-old[k])<1e-12
    result=dict(preregister=prereg,rows=rows,renaming_checks=renamed,
        production_modified=False,coefficient_fit=False,
        scope='finite exact known model; objective-only policy capacity; no real-game/human strength claim')
    (args.out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(conditions=len(rows),visible=sum(r['visible'] for r in rows),
        visible_improved=sum(r['visible'] and r['conditional_goal']>r['fixed_goal']+1e-12 for r in rows),
        hidden_unchanged=sum(not r['visible'] and abs(r['conditional_goal']-r['fixed_goal'])<1e-12 for r in rows),
        renaming=len(renamed)),indent=2))


if __name__=='__main__': main()
