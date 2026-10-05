"""Persistent numeric population: validate JSON once, no per-frame JSON/hash work."""
from dataclasses import replace
import numpy as np
from .core import Policy, compile_batch, counter_rng, digest


class Population:
    """One owner per NPC state; immutable Policy may be shared across populations.

    Updating the same instance concurrently is unsupported. Split populations own
    disjoint NPCs and use the same scope counters, so scheduling cannot affect RNG.
    """
    UPDATES={"needs":(0,1),"effects":(-1,1),"probability":(0,1),"confidence":(0,1),
             "familiarity":(0,1),"switch_cost":(0,1),"reading":(-1,1),"read_confidence":(0,1),
             "read_uncertainty":(0,1),"enabled":None,"legal":None,"known_failure":None,
             "threatened":None,"ahead":None}

    def __init__(self, contexts, policy=None):
        self.batch=compile_batch(contexts); self.policy=policy or Policy()
        self.seeds=np.array([int(digest([c["seed"],c["scope"]])[:16],16) for c in contexts],dtype=np.uint64)
        self.ticks=np.array([c["tick"] for c in contexts],dtype=np.uint64)
        self.actor_keys=tuple((c["scope"]["game"],c["scope"]["episode"],c["scope"]["npc"]) for c in contexts)
        if len(set(self.actor_keys))!=len(self.actor_keys): raise ValueError("duplicate NPC owner in population")
        self.present=np.arange(self.batch.legal.shape[1])[None,:]<np.array([len(ids) for ids in self.batch.ids])[:,None]
        self.supported=np.array([[c["needs"][key]["supported"] for key in ("physiology","safety","belonging","esteem","growth")] for c in contexts])

    def step(self, stochastic=True, **updates):
        """Adapter supplies normalized arrays only; output action indices/state arrays.

        Optional updates replace full field arrays. Changing action identity, traits,
        values or capacity requires creating a new Population at an event boundary.
        Numeric updates use the same constraints as compiled snapshots. Per-frame
        JSON audit hashes are intentionally absent; do not use this path for teachers.
        """
        changes={}
        for key,value in updates.items():
            if key not in self.UPDATES: raise ValueError("unsupported runtime update: "+key)
            bounds=self.UPDATES[key]; a=np.asarray(value)
            if a.shape!=getattr(self.batch,key).shape: raise ValueError("numeric update shape mismatch: "+key)
            if bounds is None:
                if a.dtype!=np.bool_: raise ValueError("boolean array required: "+key)
            else:
                if a.dtype.kind not in "fiu" or not np.isfinite(a).all() or (a<bounds[0]).any() or (a>bounds[1]).any(): raise ValueError("invalid numeric update: "+key)
            changes[key]=a.copy(); changes[key].flags.writeable=False
        b=replace(self.batch,**changes)
        if (b.legal&~self.present).any(): raise ValueError("padded action cannot be enabled")
        if (b.enabled&~self.supported).any() or (b.needs[~b.enabled]!=0).any(): raise ValueError("unsupported/disabled need must remain zero")
        if ((abs(b.probability.sum(2)-1)>1e-8)&self.present).any(): raise ValueError("probabilities must sum to one")
        if (b.effects[:,:,:,-1]<0).any(): raise ValueError("negative cost")
        if (self.ticks>=2**63-1).any(): raise ValueError("tick limit reached")
        rng=counter_rng(self.seeds,self.ticks); rng.flags.writeable=False
        b=replace(b,rng=rng,hashes=tuple(None for _ in b.ids))
        decision=self.policy.decide(b,stochastic)
        age=np.where(b.intent==decision.action,np.minimum(b.age+1,1000000),0)
        fields=dict(primary=decision.primary.copy(),mode=decision.mode.copy(),mode_urgency=decision.mode_urgency.copy(),intent=decision.action.copy(),age=age)
        for value in fields.values(): value.flags.writeable=False
        self.batch=replace(b,**fields); self.ticks+=1
        return decision

    def action_ids(self, decision):
        return [ids[int(j)] for ids,j in zip(self.batch.ids,decision.action)]
