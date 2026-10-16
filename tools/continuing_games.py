"""Persistent public rules for owned-value integration, not action evaluators.

No episode refill: funds, damage, investment, and the owner survive every turn.
Hidden response processes belong to the runner. The same transition produces
both observed consequences and modeled conditional effects. End values are
declared game scores, not rewards for matching a hand-authored strategy.
"""
import copy
import json
import numpy as np
from reflex.core import TRAITS, NEEDS, FEATURES, character_key, digest, Policy
from reflex.examples import context, action, effect
from reflex.decision_loop import DecisionLoop, Request
from reflex.event_reader import EventReader
from reflex.judgment import Binding, ReadControl
from reflex.pressure import NeedPressure
from reflex.progress import ProgressWatch, PurposeRequest, PurposeFeedback, Activity
from reflex.model_observer import ModelFeedback
from reflex.planning import vector
from reflex.purpose_plan import Goal
from reflex.adaptive_dynamics import PublicEvent


GAMES=('auction','resources','combat','gambling')


class ContinuingGame:
    target='persistent-purpose'
    def __init__(self,game,profile,turns=128,variant=0):
        if game not in GAMES:raise ValueError('declared rule family required')
        self.game=game;self.profile=copy.deepcopy(profile);self.turns=turns;self.variant=variant
        self.scope=dict(game='continuing-'+game,episode='one-world',npc='owner')
        self.model_id=digest(['continuing-v1',game,turns,variant])

    def start(self):
        return dict(tick=0,cash=8.,score=0.,rival=0.,plant=0,hp=12.,ammo=2,
                    investment=0,signal='0',last='none')

    def terminal(self,w):return w['tick']>=self.turns or w['hp']<=0

    def keys(self,w):
        if self.terminal(w):return ()
        if self.game=='auction':return tuple('bid:'+str(b) for b in range(min(4,int(w['cash']))+1))
        if self.game=='resources':
            return ('trade','crop:0','crop:1','crop:2')+(('invest',) if w['cash']>=4 and w['plant']<4 and not w['investment'] else ())
        if self.game=='combat':
            return ('guard','reload')+(('strike:0','strike:1','strike:2') if w['ammo'] else ())
        return ('pass',)+(('bet:0','bet:1','bet:2') if w['cash']>=1 else ())+(('bond',) if w['cash']>=4 and not w['investment'] else ())

    def public_value(self,w):
        if self.game=='auction':return (w['score']-w['rival']+.25*w['cash'])/(8*self.turns)
        if self.game=='resources':return (w['cash']-8)/(6*self.turns)
        if self.game=='combat':return (w['score']+w['hp']-12)/(3*self.turns)
        return (w['cash']-8)/(3*self.turns)

    def events(self,w,key):
        return [PublicEvent(z,*self.transition(w,key,z)) for z in ('0','1','2')]

    def goal(self,w):
        return Goal(self.target,'scored' if self.terminal(w) else 'running',float(np.clip(self.public_value(w),-1,1)))

    def needs(self,w):
        safety=float(np.clip(1.-(w['hp']/12 if self.game=='combat' else w['cash']/8),0,1))
        return dict(growth=.3,safety=safety)

    def transition(self,w,key,symbol):
        if key not in self.keys(w) or symbol not in ('0','1','2'):raise ValueError('legal public transition required')
        z=int(symbol);n=copy.deepcopy(w);n['tick']+=1;n['signal']=symbol;n['last']=key
        before=self.needs(w);style=0.;benevolence=0.
        if self.game=='auction':
            prizes=(2,7,3,8,1,6,4,5) if not self.variant else (8,1,5,2,7,3,6,4)
            prize=prizes[w['tick']%8];bid=int(key[-1]);other=(0,2,3)[z]
            if bid>other:n['score']+=prize;n['cash']-=bid
            elif bid<other:n['rival']+=prize
            n['cash']=min(12.,n['cash']+1.)
            style=bid/4-.5
        elif self.game=='resources':
            if key=='invest':n['cash']-=4;n['investment']=6;style=.4
            elif key=='trade':n['cash']+=1
            else:n['cash']+=(2+w['plant'] if int(key[-1])==z else -1);style=.2
            if w['investment']:
                n['investment']-=1
                if not n['investment']:n['plant']+=1
        elif self.game=='combat':
            if key=='guard':n['hp']=min(12.,w['hp']+2);style=-.3
            elif key=='reload':n['ammo']=3;n['hp']-=.25
            else:
                n['ammo']-=1;style=.3
                if int(key[-1])==z:n['score']+=3
                else:n['hp']-=1
            if w['tick']%8==7:n['hp']-=1
        else:
            if key=='bond':n['cash']-=4;n['investment']=6;style=-.2
            elif key!='pass':n['cash']+=2 if int(key[-1])==z else -1;style=.3
            if w['investment']:
                n['investment']-=1
                if not n['investment']:n['cash']+=7
        after=self.needs(n)
        # Immediate flow is an actual change, not a label for a preferred move.
        gain=float(np.clip((self.public_value(n)-self.public_value(w))*self.turns,-1,1))
        safety=after['safety'];delta=before['safety']-safety
        row=effect(gain,needs={'growth':gain,'safety':delta},
            values={'achievement':gain,'security':delta},style={'openness':style})
        return n,row

    def context(self,w,state=None):
        rows=[action(k,*[dict(self.transition(w,k,z)[1],p=1/3) for z in ('0','1','2')]) for k in self.keys(w)]
        c=context('continuing',rows,self.needs(w),self.profile['values'],dict(zip(TRAITS,self.profile['traits'])),mode=None)
        c.update(scope=copy.deepcopy(self.scope),tick=w['tick'],objective='Maximize the declared persistent game score',
            facts={'world':json.dumps(w,sort_keys=True,separators=(',',':'))})
        # Keep the schema's bounded fact strings; decoding is entirely public.
        encoded=c['facts'].pop('world');c['facts']={f'world{i:03d}':encoded[i:i+128] for i in range(0,len(encoded),128)}
        for k in NEEDS:
            if k not in ('growth','safety'):c['needs'][k]=dict(enabled=False,supported=False,deficit=None)
        if state is not None:c['state']=copy.deepcopy(state)
        return c

    def world(self,c):return json.loads(''.join(c['facts'][k] for k in sorted(c['facts']) if k.startswith('world')))
    def models(self,c):return {str(i):{str(z):(.8 if z==i else .1) for z in range(3)} for i in (0,1)}
    def outcome(self,c,k,z):return self.transition(self.world(c),k,z)[1]

    def request(self,c):
        w=self.world(c);condition=digest({k:v for k,v in w.items() if k not in ('tick','score','rival','signal','last')})[:24]
        acts={k:Activity('uncertain','public-rules',condition=condition) for k in self.keys(w)}
        return Request(c,{k:Binding(k,condition,FEATURES) for k in acts},{k:() for k in acts},
            threatened=True,purpose=PurposeRequest(self.goal(w).value,0.,acts))

    def feedback(self,b,n,k,c,flow):
        return ModelFeedback(PurposeFeedback(self.goal(n).value,0.,completed=self.terminal(n)),
            needs={x:float(flow['needs'].get(x,0.)) for x in ('growth','safety')},
            maintained=('safety',) if flow['needs']['safety']==0 else (),observed=vector(flow),
            event={'revealed_action':n['signal']})

    def actor(self,c,learned=True):
        reader=EventReader(c['scope'],('0','1'),self.models,self.outcome,legal_for=lambda c:('0','1','2'),
            model_id=self.model_id,known_conditionals=True,learned=learned)
        return DecisionLoop(c,policy=Policy(principle_priority='personal',commitments={character_key(c):.5}),
            predictor=reader,read_control=ReadControl(max_nodes=64),pressure=NeedPressure(c['scope']),progress=ProgressWatch(c['scope']))
