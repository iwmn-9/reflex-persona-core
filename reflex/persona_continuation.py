"""Experimental public-state self model, never an empirical learner.

A cheap purpose-directed completion is constrained by the existing Policy's
strongest-value tier and its own .025 near-best band. This reconstructs needs
from modeled public HP/threat and carries hypothetical intent only. It does not
predict future experience, progress-watch state, or route replanning, so it is
not an exact simulation of the executing DecisionLoop.
"""
from dataclasses import replace
import copy
import numpy as np
from .core import Policy, TRAITS, compile_batch
from .combat import alive, make_context


class PersonaContinuation:
    """Forecast-local cache; no owner memory, RNG, or persistent state writes."""
    def __init__(self, actors, contexts, reflex, coordinate=True):
        if len(actors)!=len(contexts) or len(actors)!=len(reflex):
            raise ValueError('one owner context and decision per actor required')
        if any('security_model' not in c['facts'] for c in contexts):
            raise ValueError('persona-band requires immediate survival-security combat effects')
        self.templates=dict(zip(actors,contexts));self.coordinate=coordinate
        self.initial={i:copy.deepcopy(d['next_state']) for i,d in zip(actors,reflex)}
        self.cache={};self.scorings=0;self.batches=0;self.cache_hits=0;self.choices=0;self.conflicts=0

    def start(self, first):
        states=copy.deepcopy(self.initial)
        for i,k in first.items():
            old=self.templates[i]['state']
            states[i].update(intent_action=k,age=min(old['age']+1,1000000) if old['intent_action']==k else 0)
        return states

    def contexts(self,w,actors,route,states):
        contexts=[]
        for i in actors:
            c=self.templates[i]
            profile=dict(traits=tuple(c['personality'][t] for t in TRAITS),values=c['values'])
            # The owner's perceived route stays fixed. A plan's support role
            # affects only tactical tie-breaking, never the owner's goal. Needs
            # are reconstructed; no root empirical samples are misrepresented
            # as evidence about hypothetical future conditions.
            route_i=c['facts']['chosen_route']
            cc=make_context(w,i,profile,c['seed'],route_i,memory=states[i],survival_security=True)
            if cc['scope']!=c['scope']:
                raise ValueError('same combat actor/episode required for self continuation')
            contexts.append(cc)
        return contexts

    def choose(self,w,team,route,states):
        from .combat_planning import tactical_scores
        actors=alive(w,team)
        if not actors:return {},{}
        key=(w,tuple((i,route[i] if isinstance(route,dict) else route,tuple(sorted(states[i].items()))) for i in actors))
        if key in self.cache:
            self.cache_hits+=1
            choices,next_states,conflicts=self.cache[key]
        else:
            cs=self.contexts(w,actors,route,states)
            b=compile_batch(cs);d=Policy().decide(b,False);self.scorings+=len(actors);self.batches+=1
            choices={};reserved=set();selected=[];conflicts=0
            for row,i in enumerate(actors):
                best=float(np.max(np.where(d.eligible[row],d.scores[row],-np.inf)))
                permitted=[j for j,k in enumerate(b.ids[row]) if d.eligible[row,j] and d.scores[row,j]>=best-.025-1e-12]
                free=[j for j in permitted if not b.ids[row][j].startswith('move:') or tuple(map(int,b.ids[row][j].split(':')[1:])) not in reserved]
                # Coordination is subordinate to the persona band. If no
                # compatible non-conflicting option remains, retain personality
                # and expose the modeled collision rather than violate a tier.
                pool=free if self.coordinate and free else permitted
                tactic=tactical_scores(w,i,route[i] if isinstance(route,dict) else route)
                j=max(pool,key=lambda j:(tactic[b.ids[row][j]],float(d.scores[row,j]),b.ids[row][j]))
                k=b.ids[row][j];choices[i]=k;selected.append(j)
                if k.startswith('move:'):
                    pos=tuple(map(int,k.split(':')[1:]));conflicts+=int(pos in reserved);reserved.add(pos)
            records=replace(d,action=np.array(selected,dtype=int)).records(b)
            next_states={i:r['next_state'] for i,r in zip(actors,records)}
            self.cache[key]=(choices,next_states,conflicts)
        self.choices+=len(actors);self.conflicts+=conflicts
        return choices.copy(),copy.deepcopy(next_states)
