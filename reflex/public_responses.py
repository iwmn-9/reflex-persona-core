"""Owner-local finite PUBLIC response hypotheses for world-model callbacks.

The adapter supplies legal peer observations and declared hypothetical profiles.
Actual private traits/memory/current choices never belong to those observations.
Predictive adoption is scored prequentially against a declared neutral model.
It does not identify a person's true type or establish decision-value benefit.
"""
import copy
from collections import OrderedDict
from .core import Policy,TRAITS,VALUES,digest,compile_batch,identifier,number
from .opponent_beliefs import HypothesisTracker
from .decision_loop import EvidenceGate
from .prediction_support import gate_record,restored_gate


class PublicResponses:
    def __init__(self,scope,profiles,*,capacity=64,model_seeds=(271,811)):
        if type(capacity) is not int or not 1<=capacity<=1024:raise ValueError('bounded peer memory required')
        if not isinstance(profiles,(tuple,list)) or not 2<=len(profiles)<=15:raise ValueError('finite declared profile hypotheses required')
        if not isinstance(model_seeds,tuple) or not 1<=len(model_seeds)<=8 or len(set(model_seeds))!=len(model_seeds) or any(type(s) is not int or not 0<=s<2**63 for s in model_seeds):
            raise ValueError('bounded declared hypothesis seeds required')
        self.model_seeds=model_seeds
        self.owner=digest(scope);self.capacity=capacity;self.profiles=copy.deepcopy(tuple(profiles))
        names=tuple(p['id'] for p in profiles)
        if len(set(names))!=len(names) or 'neutral' in names:raise ValueError('distinct declared hypotheses required')
        for profile in self.profiles:
            identifier(profile['id'])
            if len(profile['traits'])!=len(TRAITS) or set(profile['values'])-set(VALUES):raise ValueError('known declared preference axes required')
            for value in (*profile['traits'],*profile['values'].values()):number(value,0,1)
            profile['traits']=tuple(profile['traits'])
        self.names=('neutral',*names);self.trackers=OrderedDict();self.gates=EvidenceGate(capacity);self.policy=Policy(principle_priority='finite')

    def models(self,public):
        public=copy.deepcopy(public)
        # Core packets contain fields for their OWN executor. Those fields are
        # not public evidence about a counterpart's hidden random generator,
        # current intent/mode, or privately inferred opponent response.
        public['seed']=self.model_seeds[0];public['opponent']=None
        public['state']=dict(primary_need=None,mode=None,intent_action=None,age=0,mode_urgency=0.)
        compile_batch([public]);legal={a['id'] for a in public['actions'] if a['legal'] and not a['known_failure']}
        templates=(dict(id='neutral',traits=(.5,)*5,values={}),*self.profiles);result={}
        for profile in templates:
            c=copy.deepcopy(public);c.update(personality=dict(zip(TRAITS,profile['traits'])),values={k:profile['values'].get(k,0.) for k in VALUES})
            probs={k:0. for k in sorted(legal)}
            for seed in self.model_seeds:
                c['seed']=seed;chosen=self.policy.choose(c,False)['action_id'];probs[chosen]+=1/len(self.model_seeds)
            result[profile['id']]=probs
        return result

    def forecast(self,peer,public):
        if public['scope']['npc']!=peer:raise ValueError('same observed peer required')
        models=self.models(public);tracker=self.trackers.get(peer)
        if tracker is None:tracker=HypothesisTracker(self.names)
        candidate=tracker.snapshot().predict(models);prior=models['neutral']
        active=self.gates.accepted(peer)
        return copy.deepcopy(candidate if active else prior),dict(active=active,gate=self.gates.status(peer),
            observations=tracker.observations,prior=prior,candidate=candidate,
            semantics='public behavior hypotheses, not revealed true profile')

    def observe(self,peer,public,revealed_action,event_id):
        models=self.models(public);_,trial=self.forecast(peer,public)
        tracker=copy.deepcopy(self.trackers.get(peer,HypothesisTracker(self.names)))
        # Validate and update a detached tracker before modifying gate/history.
        update=tracker.observe(models,revealed_action,event_id)
        gates=copy.deepcopy(self.gates);score=gates.categorical(peer,trial['prior'],trial['candidate'],revealed_action)
        self.trackers.pop(peer,None);self.trackers[peer]=tracker;self.gates=gates
        while len(self.trackers)>self.capacity:
            evicted,_=self.trackers.popitem(last=False);self.gates.entries.pop(evicted,None)
        return dict(update=update,score=score,active=self.gates.accepted(peer))

    def record(self):
        return dict(format='public-response-v2',owner=self.owner,capacity=self.capacity,profiles=copy.deepcopy(self.profiles),model_seeds=list(self.model_seeds),
            peers=[dict(peer=k,tracker=v.record()) for k,v in self.trackers.items()],gates=gate_record(self.gates))

    @classmethod
    def from_record(cls,scope,r):
        if r['format']!='public-response-v2' or r['owner']!=digest(scope):raise ValueError('same owner and public-only estimator checkpoint required')
        bank=cls(scope,r['profiles'],capacity=r['capacity'],model_seeds=tuple(r['model_seeds']))
        if not isinstance(r['peers'],list) or len(r['peers'])>bank.capacity:raise ValueError('bounded peer checkpoint required')
        for row in r['peers']:
            peer=row['peer'];identifier(peer)
            if peer in bank.trackers:raise ValueError('unique peer checkpoint required')
            tracker=HypothesisTracker.from_record(row['tracker'])
            expected=HypothesisTracker(bank.names)
            if (tracker.names,tracker.retention,tracker.smoothing,tracker.lifts.maxlen,tracker.responsive,tracker.prior,tracker.inertia,tracker.assumption_trust)!=(expected.names,expected.retention,expected.smoothing,expected.lifts.maxlen,expected.responsive,expected.prior,expected.inertia,expected.assumption_trust):
                raise ValueError('same declared predictive estimator required')
            bank.trackers[peer]=tracker
        bank.gates=restored_gate(r['gates'])
        if bank.gates.capacity!=bank.capacity or set(bank.gates.entries)-set(bank.trackers):raise ValueError('matching peer evidence checkpoint required')
        return bank
