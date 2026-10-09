"""Reference v3 numeric contract; compile at the game boundary, score in batches."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np

VERSION = "reflex-v3"
TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
NEEDS = ("physiology", "safety", "belonging", "esteem", "growth")
VALUES = ("self_direction", "stimulation", "hedonism", "achievement", "power",
          "security", "conformity", "tradition", "benevolence", "universalism")
FEATURES = ("objective",) + NEEDS + VALUES + tuple("style_"+key for key in TRAITS) + ("cost",)
MAX_ACTIONS, MAX_OUTCOMES = 256, 8


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError("finite numeric value required")
    if not low <= value <= high:
        raise ValueError(f"value outside [{low}, {high}]")
    return float(value)


def exact(obj, keys):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise ValueError(f"expected keys: {keys}")


def identifier(value):
    if not isinstance(value, str) or not value or len(value) > 160:
        raise ValueError("nonempty identifier, at most 160 characters, required")
    return value


def counter_rng(seeds, ticks):
    """Vectorized SplitMix64 counter streams; gameplay RNG, not cryptography."""
    seeds=np.asarray(seeds,dtype=np.uint64); ticks=np.asarray(ticks,dtype=np.uint64)
    with np.errstate(over="ignore"):
        x=seeds[:,None]+ticks[:,None]*np.uint64(0x9E3779B97F4A7C15)+np.arange(1,4,dtype=np.uint64)[None,:]*np.uint64(0xD1B54A32D192ED03)
        x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        x=x^(x>>np.uint64(31))
    return (x>>np.uint64(11)).astype(float)*(1.0/2**53)


@dataclass(frozen=True)
class Batch:
    ids: tuple
    targets: tuple
    hashes: tuple
    traits: np.ndarray
    values: np.ndarray
    needs: np.ndarray
    enabled: np.ndarray
    effects: np.ndarray
    probability: np.ndarray
    legal: np.ndarray
    confidence: np.ndarray
    familiarity: np.ndarray
    known_failure: np.ndarray
    switch_cost: np.ndarray
    primary: np.ndarray
    mode: np.ndarray
    intent: np.ndarray
    age: np.ndarray
    mode_urgency: np.ndarray
    rng: np.ndarray
    reading: np.ndarray
    read_confidence: np.ndarray
    read_uncertainty: np.ndarray
    threatened: np.ndarray
    ahead: np.ndarray

    def take(self, indices):
        indices = np.atleast_1d(indices)
        fields = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            fields[key] = tuple(value[int(i)] for i in indices) if isinstance(value, tuple) else value[indices].copy()
            if isinstance(fields[key], np.ndarray): fields[key].flags.writeable=False
        return Batch(**fields)


def compile_batch(contexts):
    """Validate snapshots, canonically order candidates, freeze numeric input arrays.

    Adapter supplies perceived outcome estimates, never hidden true world state.
    Unsupported and disabled needs differ in JSON; both receive zero numeric weight.
    """
    if not contexts:
        raise ValueError("empty batch")
    n = len(contexts)
    traits = np.zeros((n, 5)); values = np.zeros((n, 10)); needs = np.zeros((n, 5))
    enabled = np.zeros((n, 5), bool)
    # Allocate only actual maxima, bounded by the public contract.
    a = max(len(c["actions"]) for c in contexts)
    k = max(len(act["outcomes"]) for c in contexts for act in c["actions"])
    if not 1 <= a <= MAX_ACTIONS or not 1 <= k <= MAX_OUTCOMES:
        raise ValueError("action/outcome capacity exceeded")
    effects = np.zeros((n, a, k, len(FEATURES))); probability = np.zeros((n, a, k))
    legal = np.zeros((n, a), bool); confidence = np.zeros((n, a)); familiarity = confidence.copy()
    known_failure = legal.copy(); switch_cost = confidence.copy(); reading = confidence.copy()
    primary = np.full(n, -1, int); mode = np.full(n, -1, int); intent = primary.copy()
    age = np.zeros(n, int); mode_urgency=np.zeros(n); seeds=np.zeros(n,dtype=np.uint64); ticks=seeds.copy(); read_confidence = np.zeros(n)
    read_uncertainty = np.zeros(n); threatened = np.zeros(n, bool); ahead = threatened.copy()
    ids, targets, hashes = [], [], []
    for i, c in enumerate(contexts):
        exact(c, ("version", "scope", "seed", "tick", "personality", "values", "needs",
                  "state", "objective", "facts", "actions", "opponent"))
        if c["version"] != VERSION:
            raise ValueError("v2 requires explicit re-authoring, not implicit conversion")
        exact(c["scope"], ("game", "episode", "npc"))
        for value in c["scope"].values(): identifier(value)
        for key in ("tick", "seed"):
            if type(c[key]) is not int or not 0 <= c[key] < 2**63: raise ValueError("nonnegative integer seed/tick required")
        exact(c["personality"], TRAITS); exact(c["values"], VALUES); exact(c["needs"], NEEDS)
        traits[i] = [number(c["personality"][key], 0, 1) for key in TRAITS]
        values[i] = [number(c["values"][key], 0, 1) for key in VALUES]
        for j, key in enumerate(NEEDS):
            d = c["needs"][key]; exact(d, ("supported", "enabled", "deficit"))
            if type(d["supported"]) is not bool or type(d["enabled"]) is not bool: raise ValueError("boolean flags required")
            if d["enabled"] and not d["supported"]: raise ValueError("unsupported need enabled")
            if d["enabled"]: needs[i,j] = number(d["deficit"], 0, 1); enabled[i,j] = True
            elif d["deficit"] is not None: raise ValueError("disabled deficit must be null")
        s = c["state"]; exact(s, ("primary_need", "mode", "intent_action", "age", "mode_urgency"))
        if s["primary_need"] is not None:
            primary[i] = NEEDS.index(s["primary_need"])
            if not enabled[i, primary[i]]: raise ValueError("primary need disabled")
        if s["mode"] not in (None, "need", "principle"): raise ValueError("invalid mode")
        mode[i] = {None:-1, "need":0, "principle":1}[s["mode"]]
        if type(s["age"]) is not int or not 0 <= s["age"] <= 1000000: raise ValueError("invalid decision age")
        age[i] = s["age"]
        mode_urgency[i]=number(s["mode_urgency"],0,1)
        identifier(c["objective"])
        if not isinstance(c["facts"], dict) or not c["facts"]: raise ValueError("observed facts required")
        for key, value in c["facts"].items(): identifier(key); identifier(value)
        ordered = sorted(c["actions"], key=lambda action: action["id"])
        action_ids = tuple(identifier(act["id"]) for act in ordered)
        if len(set(action_ids)) != len(action_ids): raise ValueError("duplicate action id")
        ids.append(action_ids); targets.append(tuple(act["target"] for act in ordered)); hashes.append(digest(c))
        if s["intent_action"] is not None:
            identifier(s["intent_action"])
            if s["intent_action"] in action_ids: intent[i] = action_ids.index(s["intent_action"])
        for j, act in enumerate(ordered):
            exact(act, ("id", "target", "legal", "outcomes", "confidence", "familiarity", "known_failure", "switch_cost"))
            if act["target"] is not None: identifier(act["target"])
            for key in ("legal", "known_failure"):
                if type(act[key]) is not bool: raise ValueError("boolean action flag required")
            legal[i,j] = act["legal"]; known_failure[i,j] = act["known_failure"]
            confidence[i,j] = number(act["confidence"],0,1); familiarity[i,j] = number(act["familiarity"],0,1)
            switch_cost[i,j] = number(act["switch_cost"],0,1)
            if not act["outcomes"]: raise ValueError("outcomes required")
            for h, outcome in enumerate(act["outcomes"]):
                exact(outcome, ("p", "objective", "needs", "values", "style", "cost"))
                if set(outcome["needs"]) - set(NEEDS) or set(outcome["values"]) - set(VALUES) or set(outcome["style"]) - set(TRAITS): raise ValueError("unknown effect dimension")
                probability[i,j,h] = number(outcome["p"],0,1)
                effects[i,j,h] = [number(outcome["objective"],-1,1)] + [number(outcome["needs"].get(key,0),-1,1) for key in NEEDS] + [number(outcome["values"].get(key,0),-1,1) for key in VALUES] + [number(outcome["style"].get(key,0),-1,1) for key in TRAITS] + [number(outcome["cost"],0,1)]
            if abs(probability[i,j].sum()-1) > 1e-8: raise ValueError("outcome probabilities must sum to one")
        if not legal[i].any(): raise ValueError("adapter must supply at least one feasible action, including WAIT if needed")
        other = c["opponent"]
        if other is not None:
            exact(other, ("confidence", "uncertainty", "evidence_ids", "threatened", "maintains_advantage", "deltas"))
            if not other["evidence_ids"] or set(other["evidence_ids"]) - set(c["facts"]): raise ValueError("reading requires observed evidence")
            for key in ("threatened", "maintains_advantage"):
                if type(other[key]) is not bool: raise ValueError("boolean prediction flag required")
            read_confidence[i] = number(other["confidence"],0,1); read_uncertainty[i] = number(other["uncertainty"],0,1)
            threatened[i] = other["threatened"]; ahead[i] = other["maintains_advantage"]
            if set(other["deltas"]) - set(action_ids): raise ValueError("unknown reading action")
            reading[i,:len(ordered)] = [number(other["deltas"].get(key,0),-1,1) for key in action_ids]
        # Isolated randomness, independent of batch size/order/thread scheduling.
        seeds[i]=int(digest([c["seed"],c["scope"]])[:16],16); ticks[i]=c["tick"]
    rng=counter_rng(seeds,ticks)
    arrays = locals()
    fields = {key: arrays[key] for key in Batch.__dataclass_fields__}
    for key in ("ids", "targets", "hashes"): fields[key] = tuple(fields[key])
    for value in fields.values():
        if isinstance(value,np.ndarray): value.flags.writeable = False
    return Batch(**fields)


@dataclass(frozen=True)
class Decisions:
    action: np.ndarray
    primary: np.ndarray
    mode: np.ndarray
    mode_urgency: np.ndarray
    reading_used: np.ndarray
    request_reading: np.ndarray
    features: np.ndarray
    scores: np.ndarray
    eligible: np.ndarray

    def records(self, batch):
        result = []
        for i, j in enumerate(self.action):
            result.append({"version":VERSION,"context_hash":batch.hashes[i],
                "action_id":batch.ids[i][j],"target":batch.targets[i][j],
                "reading_used":bool(self.reading_used[i]),"request_reading":bool(self.request_reading[i]),
                "next_state":{"primary_need":NEEDS[self.primary[i]] if self.primary[i]>=0 else None,
                              "mode":"principle" if self.mode[i] else "need",
                              "mode_urgency":float(self.mode_urgency[i]),
                              "intent_action":batch.ids[i][j],"age":int(min(batch.age[i]+1,1000000) if batch.intent[i]==j else 0)}})
        return result


class Policy:
    """Stateless shared policy. Learned residual is optional and bounded; defaults zero."""
    def __init__(self, residual=None, *, principle_priority='lexicographic'):
        if principle_priority not in ('lexicographic','finite'):raise ValueError('known principle priority required')
        self.principle_priority=principle_priority
        self.residual = np.zeros(8) if residual is None else np.asarray(residual,dtype=float).copy()
        if self.residual.shape != (8,) or not np.isfinite(self.residual).all() or np.max(abs(self.residual)) > .08+1e-9:
            raise ValueError("residual must have eight finite coefficients bounded by .08")
        self.residual.flags.writeable=False

    def decide(self, b, stochastic=True, *, appraisal=None, purpose=None, max_social_regret=.15):
        n,a,k,d = b.effects.shape; rows=np.arange(n)
        old=np.maximum(b.primary,0); best=b.needs.argmax(1)
        # Intent switches only for material urgency changes; no rigid Maslow ranking.
        keep=(b.primary>=0)&b.enabled[rows,old]&(b.needs[rows,old]>.05)&(b.needs[rows,best]<b.needs[rows,old]+.15)
        primary=np.where(keep,b.primary,best); primary=np.where(b.enabled.any(1)&(b.needs.max(1)>.05),primary,-1)
        urgency=np.where(primary>=0,b.needs[rows,np.maximum(primary,0)],0)
        top=b.values.argmax(1); strength=b.values[rows,top]
        # Game coefficients, not a claim about psychological causality.
        hold=np.clip(.5+.35*(b.traits[:,1]-.5)-.25*(b.traits[:,4]-.5)+.3*(strength-urgency),.1,.9)
        mode=(b.rng[:,0]<hold).astype(int)
        valid_old=(b.mode>=0)&(primary==b.primary)&(abs(urgency-b.mode_urgency)<.15)
        mode=np.where(valid_old,b.mode,mode)
        mode=np.where(strength==0,0,mode)
        mode_urgency=np.where(valid_old,b.mode_urgency,urgency)
        w=np.zeros((n,d)); w[:,0]=.65
        w[:,1:6]=.12*b.needs*b.enabled
        w[rows,1+np.maximum(primary,0)]+=np.where(primary>=0,1.5*urgency,0)
        w[:,6:16]=.08*b.values
        w[rows,6+top]+=strength*np.where(mode==1,1.6,.3)
        w[:,16:21]=.12*(2*b.traits-1)
        w[:,-1]=-.6
        # Outcome effects are bounded and residual cannot bypass legal/failure masks.
        perceived=b.effects.copy()
        perceived[:,:,:,:-1]*=np.where(perceived[:,:,:,:-1]>0,b.confidence[:,:,None,None],1)
        outcome=np.einsum('nakd,nd->nak',perceived,w)
        means=np.einsum('nak,nakd->nad',b.probability,perceived)
        expected=np.einsum('nak,nak->na',b.probability,outcome)
        loss=np.einsum('nak,nak->na',b.probability,np.maximum(-outcome,0))
        risk=.25+.75*b.traits[:,4]+.25*b.needs[:,1]
        # Growth enters benefit, never a blanket bonus for any risky action.
        risk_adjusted=expected-risk[:,None]*loss
        familiar=.10*(1-b.confidence)*b.familiarity
        switch=b.switch_cost*(np.arange(a)[None,:]!=b.intent[:,None])
        scores=risk_adjusted+familiar-switch
        top_effect=means[rows,:,6+top]
        primary_effect=means[rows,:,1+np.maximum(primary,0)]* (primary>=0)[:,None]
        features=np.stack((means[:,:,0],primary_effect,top_effect,loss,means[:,:,-1],
                           familiar,switch,b.confidence),axis=-1)
        adjustment=np.clip(np.einsum('nad,d->na',features,self.residual),-.08,.08)
        scores+=adjustment
        viable=b.legal & ~b.known_failure
        # If every feasible action has a known failure, adapter must provide recovery.
        if not viable.any(1).all(): raise ValueError("no understood viable action: adapter must supply recovery/WAIT")
        # In principle mode, strongest value has first priority. Compatible secondary
        # values, objective, costs and needs decide within this narrow top-value tier.
        tier=np.max(np.where(viable,top_effect,-np.inf),axis=1)
        eligible=viable & ((mode==0)[:,None] | (top_effect>=tier[:,None]-.02))
        # Finite priority keeps the same strongest-value weights and mode, but
        # does not give a marginal value gain an unlimited veto over all costs.
        if self.principle_priority=='finite':eligible=viable
        own=np.argmax(np.where(eligible,scores,-np.inf),axis=1)
        read_ok=(b.read_confidence>=.6)&(~b.ahead|b.threatened)
        candidate_scores=scores+b.reading*b.read_confidence[:,None]
        candidate=np.argmax(np.where(eligible,candidate_scores,-np.inf),axis=1)
        gain=candidate_scores[rows,candidate]-candidate_scores[rows,own]
        read_used=read_ok&((gain>.05+b.read_uncertainty*.15)|b.threatened)
        scores=np.where(read_used[:,None],candidate_scores,scores)
        maximum=np.max(np.where(eligible,scores,-np.inf),axis=1)
        near=eligible&(scores>=maximum[:,None]-.025)
        # Randomness cannot select materially worse choices; mostly best, rare near ties.
        weights=np.where(near,np.exp(np.clip((scores-maximum[:,None])/.008,-80,0)),0)
        weights/=weights.sum(1,keepdims=True)
        if stochastic:
            chosen=(np.cumsum(weights,axis=1)<b.rng[:,1,None]).sum(1)
            chosen=np.minimum(chosen,a-1)
        else: chosen=np.argmax(np.where(eligible,scores,-np.inf),axis=1)
        request=(b.threatened|(~b.ahead&(maximum<.15)))&(b.read_confidence<.6)
        result=Decisions(chosen,primary,mode,mode_urgency,read_used,request,features,scores,eligible)
        if appraisal is not None:
            from .social import appraised
            return appraised(b,result,appraisal,purpose,max_social_regret,stochastic)
        if purpose is not None:raise ValueError('social purpose requires an appraisal')
        return result

    def choose(self, context, stochastic=True):
        batch=compile_batch([context])
        return self.decide(batch,stochastic).records(batch)[0]
