"""Two tiny executable games verify that the same policy consumes different rules."""
import copy
from .core import NEEDS, VALUES, TRAITS
from .examples import context, action, effect


class ResourceAdapter:
    """Observe only public budget/energy, honor time/resource constraints."""
    def observe(self, world, profile, npc="actor", tick=0):
        acts=[action("WAIT",effect(0))]
        if world["time"]>=1 and world["food"]>=1:
            acts.append(action("REST",effect(.1,{"physiology":.7},cost=.1)))
        if world["time"]>=2:
            acts.append(action("PRACTICE",effect(.2,{"growth":.7},cost=.25)))
            acts.append(action("WORK",effect(.6,{"safety":.4},cost=.2)))
        c=context("resource-session",acts,profile["deficits"],profile["values"],profile["traits"],mode=None)
        c["scope"].update(game="resource_game",npc=npc); c["tick"]=tick
        c["facts"]={"time":f"残り時間={world['time']}","food":f"利用可能な食料={world['food']}"}
        c["state"]=copy.deepcopy(profile.get("state",c["state"]))
        return c

    def transition(self, world, decision):
        result=copy.deepcopy(world); key=decision["action_id"]
        if key=="REST":
            if result["time"]<1 or result["food"]<1: raise ValueError("infeasible REST")
            result["time"]-=1; result["food"]-=1; result["recovered"]+=.7
        elif key in ("WORK","PRACTICE"):
            if result["time"]<2: raise ValueError("infeasible two-unit action")
            result["time"]-=2
            if key=="WORK": result["food"]+=2
            else: result["skill"]+=.7
        elif key!="WAIT": raise ValueError("unknown resource action")
        return result


class ArenaAdapter:
    """Known response probabilities can incorporate short-horizon opponent reading."""
    def observe(self, world, profile, npc="actor", tick=0):
        acts=[action("GUARD",effect(.15,{"safety":.5})),action("ATTACK",effect(.7,p=.7),effect(-.5,p=.3))]
        if world["energy"]>=2: acts.append(action("COUNTER",effect(.3),switch_cost=.05))
        c=context("arena-session",acts,profile["deficits"],profile["values"],profile["traits"],mode=None)
        c["scope"].update(game="arena_game",npc=npc); c["tick"]=tick
        c["facts"]={"energy":f"使えるエネルギー={world['energy']}"}
        c["state"]=copy.deepcopy(profile.get("state",c["state"]))
        if world.get("observed_opponent") is not None:
            # Input must come from observed events, not hidden opponent strategy.
            read=copy.deepcopy(world["observed_opponent"])
            c["facts"]["opponent_evidence"]="観測済みの相手の応答履歴"
            read["evidence_ids"]=["opponent_evidence"]
            read["deltas"]={k:v for k,v in read["deltas"].items() if k in {a["id"] for a in acts}}
            c["opponent"]=read
        return c

    def transition(self, world, decision, rng):
        result=copy.deepcopy(world); key=decision["action_id"]
        if key=="COUNTER":
            if result["energy"]<2: raise ValueError("infeasible COUNTER")
            result["energy"]-=2; result["score"]+=.3
        elif key=="ATTACK": result["score"]+=.7 if rng.random()<.7 else -.5
        elif key=="GUARD": result["energy"]+=1; result["score"]+=.15
        else: raise ValueError("unknown arena action")
        return result


def profile():
    return dict(traits={key:.5 for key in TRAITS},values={key:0 for key in VALUES},
                deficits={key:.1 for key in NEEDS})
