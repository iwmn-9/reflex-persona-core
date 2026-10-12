"""Detached owner states for conditional future decisions, without new psychology.

The game supplies each future public context. Signed owners may disclose their
own fixed axes/state; a nonparticipant's real private state is never registered.
Every candidate/scenario starts a fresh branch. Predicted decisions are not
observations or training samples for the real owners.
"""
import copy
from .core import Policy, TRAITS, digest


class OwnerBranch:
    def __init__(self, actors, contexts, decisions, roots,*,policy=None):
        if not actors or not len(actors)==len(contexts)==len(decisions)==len(roots):
            raise ValueError('one context, decision and root per declared owner')
        if len(set(actors))!=len(actors):raise ValueError('distinct owners required')
        if (len({c['scope']['npc'] for c in contexts})!=len(contexts) or
                len({(c['scope']['game'],c['scope']['episode'],c['tick']) for c in contexts})!=1):
            raise ValueError('distinct owners in one simultaneous game context required')
        self.owners={};self.states={};self.steps=0;self.policy=policy or Policy(principle_priority='finite')
        for actor,c,d,root in zip(actors,contexts,decisions,roots):
            if d['context_hash']!=digest(c):raise ValueError('decision belongs to owner context')
            if root not in {a['id'] for a in c['actions'] if a['legal'] and not a['known_failure']}:
                raise ValueError('legal counterfactual root required')
            self.owners[actor]=dict(scope=copy.deepcopy(c['scope']),personality=copy.deepcopy(c['personality']),
                values=copy.deepcopy(c['values']),seed=c['seed'],tick=c['tick'])
            state=copy.deepcopy(d['next_state']);old=c['state']
            state.update(intent_action=root,age=min(old['age']+1,1000000) if old['intent_action']==root else 0)
            self.states[actor]=state

    def context(self, actor, world, factory):
        owner=self.owners[actor]
        profile=dict(traits=tuple(owner['personality'][k] for k in TRAITS),values=copy.deepcopy(owner['values']))
        c=factory(world,actor,profile,owner['seed'],copy.deepcopy(self.states[actor]))
        if (c['scope']!=owner['scope'] or c['personality']!=owner['personality'] or
                c['values']!=owner['values'] or c['seed']!=owner['seed'] or c['state']!=self.states[actor] or c['tick']<=owner['tick']):
            raise ValueError('future context must retain owner/axes/state and advance time')
        return c

    def choose(self, actors, world, factory, exchange=None):
        if not actors or any(i not in self.owners for i in actors):raise ValueError('declared living owners required')
        contexts=[self.context(i,world,factory) for i in actors]
        decisions=[self.policy.choose(c,False) for c in contexts]
        audit=None
        if exchange is not None:
            selected,audit=exchange(world,tuple(actors),contexts,decisions)
            if selected is not None:decisions=selected
        if len(decisions)!=len(actors):raise ValueError('one future decision per owner required')
        for i,c,d in zip(actors,contexts,decisions):
            if d['context_hash']!=digest(c) or d['action_id'] not in {a['id'] for a in c['actions'] if a['legal'] and not a['known_failure']}:
                raise ValueError('future decision must belong to the legal owner context')
            if d['next_state']['intent_action']!=d['action_id']:raise ValueError('future memory must name the executed root')
            self.states[i]=copy.deepcopy(d['next_state']);self.owners[i]['tick']=c['tick']
        self.steps+=1
        return {i:d['action_id'] for i,d in zip(actors,decisions)},audit

    def receipt(self):
        return dict(steps=self.steps,owners=list(self.owners),states_sha256=digest(self.states),
            source='detached hypothetical owner state; never a real experience update')
