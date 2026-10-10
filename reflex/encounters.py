"""Simultaneous NPC encounters, driven by the existing common personality policy.

Games own matching, public observations and consequences. No NPC has scripted
cooperation, betrayal or repair. Forecasts use past revealed actions only.
"""
from collections import deque
import copy
import numpy as np
from .core import Policy, TRAITS, compile_batch
from .examples import action, context, effect
from .laboratory import PROFILES
from .social import SocialPopulation, SocialBinding, SocialEvent
from .social_projection import CoarseSocialPopulation


RULES={
    'exchange':dict(cc=(.55,.12),ct=(-.15,-.1),tc=(.8,.08),tt=(.05,-.02),
                    alone=(.3,.06),unmatched=(.12,0),rest=(.08,0),upkeep=.04),
    'watch':dict(cc=(.5,.3),ct=(-.4,-.5),tc=(.6,.1),tt=(-.25,-.3),
                 alone=(.32,.05),unmatched=(.1,0),rest=(.08,0),upkeep=.1),
}
LAYOUTS={'balanced4':(0,1,2,3),'balanced8':(0,1,2,3,0,1,2,3),'social4':(2,2,0,3)}
KINDS=('cooperate','claim')


def options(actors,own):
    return {'alone':('alone',None),'rest':('rest',None),'finished':('finished',None),
            **{f'{kind}:{target}':(kind,target) for target in actors if target!=own for kind in KINDS}}


class EncounterWorld:
    def __init__(self,seed,domain,layout,contact='free'):
        if domain not in RULES or layout not in LAYOUTS or contact not in ('free','public'):
            raise ValueError('known encounter rules/layout/contact required')
        self.seed=seed;self.domain=domain;self.layout=layout;self.contact=contact;self.rules=RULES[domain]
        self.actors=tuple(f'npc-{i}' for i in range(len(LAYOUTS[layout])))
        order=np.random.default_rng(seed).permutation(LAYOUTS[layout])
        self.profiles={actor:copy.deepcopy(PROFILES[int(index)]) for actor,index in zip(self.actors,order)}
        self.energy={a:.8 for a in self.actors};self.health={a:8. for a in self.actors}
        self.stock={a:3. for a in self.actors};self.score={a:0. for a in self.actors}
        self.history={a:deque(maxlen=8) for a in self.actors};self.tick=0
        self.catalog={a:options(self.actors,a) for a in self.actors}

    def available_partner(self,own):
        """Public round-robin contact slots; no forced acceptance or response.

        Four turns per slot allow observations. Every pair gets one opportunity
        in a cycle, independently of personality, scores and relationship state.
        """
        order=list(self.actors)
        for _ in range((self.tick//4)%(len(order)-1)):order=[order[0],order[-1],*order[1:-1]]
        for a,b in zip(order[:len(order)//2],reversed(order[len(order)//2:])):
            if own==a:return b
            if own==b:return a
        raise ValueError('actor must belong to this world')

    def forecast(self,own,target):
        """Joint likelihood of partner selecting this actor and one of two modes.

        Avoid multiplying independently estimated contact and cooperation rates.
        An unseen partner has a broad symmetric contact prior, not its personality.
        Dead partners are public and cannot enter a new matching pair.
        """
        if self.health[target]<=0 or self.energy[target]<.18:return (0.,0.,1.)
        history=self.history[target];prior=1/(len(self.actors)-1)
        if self.contact=='public':
            if target!=self.available_partner(own):return (0.,0.,1.)
            pc=(sum(kind=='cooperate' for kind,_ in history)+1)/(len(history)+3)
            pt=(sum(kind=='claim' for kind,_ in history)+1)/(len(history)+3)
            return pc,pt,1-pc-pt
        cooperative=sum(kind=='cooperate' and recipient==own for kind,recipient in history)
        claim=sum(kind=='claim' and recipient==own for kind,recipient in history)
        denominator=len(history)+2
        pc=(cooperative+prior)/denominator;pt=(claim+prior)/denominator
        return pc,pt,1-pc-pt

    def consequences(self,kind,other=None,matched=False,luck_failure=False):
        if kind=='finished':return (0.,0.)
        if kind in ('alone','rest'):return self.rules[kind]
        if not matched:return self.rules['unmatched']
        if kind=='cooperate' and other=='cooperate' and luck_failure:return (-.1,-.15)
        return self.rules[('c' if kind=='cooperate' else 't')+('c' if other=='cooperate' else 't')]

    def perceived(self,kind,other,matched,p):
        gain,health=self.consequences(kind,other,matched)
        # These are meanings of the game's real action/outcome, not new traits.
        if kind=='cooperate':
            values={'benevolence':(1 if other=='cooperate' else .7) if matched else 0.,'universalism':.3 if matched else 0.}
            needs={'belonging':.3 if matched else 0.,'physiology':-.12,'safety':health}
            style={'agreeableness':.6}
        elif kind=='claim':
            values={'power':.9 if matched else .08,'achievement':.35 if matched else 0.,'benevolence':-.6 if other=='cooperate' and matched else 0.}
            needs={'esteem':.2 if matched else 0.,'physiology':-.12,'safety':health};style={'extraversion':.5}
        elif kind=='alone':
            values={'security':.45,'achievement':.4,'self_direction':.3,'power':.08}
            needs={'growth':.3,'safety':health,'physiology':-.12};style={'conscientiousness':.4}
        elif kind=='rest':
            values={};needs={'physiology':.4};style={}
        else:values={};needs={};style={}
        return effect(gain,needs=needs,values=values,style=style,cost=0 if kind in ('rest','finished') else .05,p=p)

    def contexts(self):
        result=[];bindings=[]
        for own in self.actors:
            acts=[];social={};profile=self.profiles[own]
            for key,(kind,target) in self.catalog[own].items():
                alive=self.health[own]>0
                legal=not alive if kind=='finished' else alive and (kind=='rest' or self.energy[own]>=.18)
                if target is not None:
                    legal &= self.health[target]>0
                    if self.contact=='public':legal &= target==self.available_partner(own)
                    pc,pt,unmatched=self.forecast(own,target)
                    # Joint cooperation can fail visibly with probability .05.
                    # Four outcomes fit the common bounded outcome contract.
                    if kind=='cooperate':
                        outcomes=[self.perceived(kind,'cooperate',True,pc*.95),
                            self.perceived(kind,'claim',True,pt),self.perceived(kind,None,False,unmatched),
                            effect(-.1,p=pc*.05,needs={'safety':-.15,'physiology':-.12},cost=.05)]
                    else:outcomes=[self.perceived(kind,'cooperate',True,pc),self.perceived(kind,'claim',True,pt),self.perceived(kind,None,False,unmatched)]
                    social[key]=SocialBinding(target,1 if kind=='cooperate' else -1,max(0,1-self.health[target]/8))
                else:outcomes=[self.perceived(kind,None,False,1.)]
                a=action(key,*outcomes,legal=bool(legal));a['target']=target;acts.append(a)
            c=context('encounters',acts,traits=dict(zip(TRAITS,profile['traits'])),values=profile['values'],
                needs={'physiology':1-self.energy[own],'safety':max(1-self.health[own]/8,max(0,(2-self.stock[own])/2)),
                       'belonging':.3,'esteem':.2,'growth':.3},mode=None)
            c['scope'].update(game=self.domain,episode=f'{self.layout}-{self.contact}-{self.seed}',npc=own)
            c.update(seed=self.seed,tick=self.tick)
            c['facts']={'public_history':'last eight revealed choices, including recipients; no current choices',
                        'available_partner':self.available_partner(own) if self.contact=='public' else 'all living actors',
                        'purpose':'gain resources and survive using public encounter rules'}
            result.append(c);bindings.append(social)
        return result,bindings

    def resolve(self,chosen):
        if set(chosen)!=set(self.actors) or any(chosen[a] not in self.catalog[a] for a in self.actors):
            raise ValueError('complete available simultaneous choices required')
        decoded={a:self.catalog[a][chosen[a]] for a in self.actors}
        # Check the whole action set before mutating any participant.
        cs,_=self.contexts()
        for c in cs:
            own=c['scope']['npc'];legal={a['id'] for a in c['actions'] if a['legal']}
            if chosen[own] not in legal:raise ValueError('unavailable actual choice')
        before={a:self.state(a) for a in self.actors};events={a:[] for a in self.actors};rows=[]
        # Shared per-pair environmental failure; unrelated to hidden personality.
        luck=np.random.default_rng(np.random.SeedSequence([self.seed,self.tick])).random((len(self.actors),len(self.actors)))
        index={a:i for i,a in enumerate(self.actors)}
        for own,(kind,target) in decoded.items():
            other,recipient=decoded[target] if target else (None,None)
            matched=target is not None and recipient==own
            failure=matched and kind==other=='cooperate' and luck[min(index[own],index[target]),max(index[own],index[target])]<.05
            gain,health=self.consequences(kind,other,matched,failure)
            if kind!='finished':
                self.energy[own]=min(1,self.energy[own]+.4) if kind=='rest' else max(0,self.energy[own]-.18)
                self.energy[own]=max(0,self.energy[own]-.025)
                self.stock[own]=max(0,self.stock[own]+gain-.08)
                self.health[own]=max(0,min(8,self.health[own]+health)-self.rules['upkeep'])
                self.score[own]+=gain
                if matched:
                    benefit=0. if failure else .8 if other=='cooperate' else -1. if kind=='cooperate' else 0.
                    events[own].append(SocialEvent(target,f'{self.tick}-{own}-{target}',self.tick,benefit,agency=not failure))
            rows.append(dict(npc=own,profile=self.profiles[own]['id'],action=chosen[own],kind=kind,target=target,
                matched=matched,other_kind=other if matched else None,luck_failure=bool(failure),gain=gain,
                before=before[own],after=self.state(own)))
        for own in self.actors:self.history[own].append(decoded[own])
        self.tick+=1
        return rows,[events[a] for a in self.actors]

    def state(self,own):
        return dict(energy=float(self.energy[own]),health=float(self.health[own]),stock=float(self.stock[own]),score=float(self.score[own]))


def run(seed,domain,layout,setting,turns=48,*,contact='free',population_factory=None):
    if setting not in ('prediction','continuous','coarse'):raise ValueError('known encounter setting required')
    world=EncounterWorld(seed,domain,layout,contact);initial,bindings=world.contexts()
    cls=CoarseSocialPopulation if setting=='coarse' else SocialPopulation
    pop=(population_factory or cls)(initial,Policy(principle_priority='finite'));trace=[]
    for tick in range(turns):
        cs,bindings=world.contexts();b=compile_batch(cs)
        forecasts={a:{target:list(world.forecast(a,target)) for target in world.actors if target!=a} for a in world.actors}
        selected=pop.step(bindings=bindings if setting!='prediction' else [{} for _ in cs],
            **{key:getattr(b,key) for key in ('needs','effects','probability','legal')})
        actions=dict(zip(world.actors,pop.action_ids(selected)))
        rows,events=world.resolve(actions);pop.observe_batch(events)
        trace.append(dict(tick=tick,rows=rows,appraisal=copy.deepcopy(pop.social_audit),
            forecasts=forecasts,
            fixed_personality=all(m.values==c['values'] and m.personality==c['personality'] for m,c in zip(pop.social,initial))))
    actors=[dict(npc=a,profile=world.profiles[a]['id'],**world.state(a),
        partners=sorted({r['target'] for t in trace for r in t['rows'] if r['npc']==a and r['matched']})) for a in world.actors]
    pairs=sum(r['matched'] for t in trace for r in t['rows'])//2
    cooperation=sum(r['matched'] and r['kind']=='cooperate' and r['other_kind']=='cooperate' for t in trace for r in t['rows'])//2
    exploited=sum(r['matched'] and r['kind']=='cooperate' and r['other_kind']=='claim' for t in trace for r in t['rows'])
    return dict(seed=seed,domain=domain,layout=layout,contact=contact,setting=setting,turns=turns,actors=actors,
        matches=pairs,cooperation_pairs=cooperation,exploited=exploited,
        alive=sum(a['health']>0 for a in actors),mean_score=float(np.mean([a['score'] for a in actors]))),trace
