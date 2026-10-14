"""Known public rules, unknown behavior: four distinct headless rule families.

Auction reuses contests.settle (three players, capped legal bids). Resources,
combat and gambling are declared diagnostics, not licensed game replicas.
The hidden process lives in the runner; no scenario/seed is stored here.
"""
import copy
import json
from dataclasses import asdict, replace

from reflex.adaptive_dynamics import EventLearner, AdaptiveKernel, PublicEvent
from reflex.closed_loop import ClosedLoop, TreeBudget
from reflex.contests import Public, public_from_record, settle
from reflex.core import TRAITS, NEEDS, digest
from reflex.examples import context, action, effect
from reflex.integrated_trials import request_for
from reflex.model_observer import ModelFeedback
from reflex.purpose_plan import Goal


class OpenGame:
    target = 'declared-purpose'

    def __init__(self, game, profile, arm='adaptive', length=6, variant=0):
        self.game, self.profile, self.length, self.variant = game, copy.deepcopy(profile), length, variant
        self.model_id = digest(['open-game-rules-v1', game, length, variant])
        factors = () if arm=='marginal' else ('scene','previous')
        learner = EventLearner(factors=factors, names=('library-a','library-b'), owner='owner', model_id=self.model_id,
                               learned=arm!='library', windows=(128,) if arm=='long' else (8,32,128))
        self.kernel = AdaptiveKernel(learner=learner, events=self.events, features=self.features,
            models=self.models, public_record=lambda w:w, model_id=self.model_id)

    def start(self, episode=0, memory=None):
        w = dict(tick=0,episode=episode,previous='none',signal=None)
        if self.game=='auction':
            prizes = (3,6,2,5,4,7) if not self.variant else (5,2,7,3,6,4)
            p = Public('auction',((),)*3,(0,)*3,(8,)*3,prizes[0],'science',prizes[1:])
            w['position'] = json.loads(json.dumps(asdict(p)))
        elif self.game=='resources': w.update(cash=6,plant=0)
        elif self.game=='combat': w.update(hp=4,enemy=8)
        elif self.game=='gambling': w.update(cash=6)
        else: raise ValueError('declared game required')
        return self.kernel.start(w,memory)

    def scene(self,w):
        if self.game=='auction': return str(int(w['position']['prize']>=5))
        return str((w['tick']+w['episode']+self.variant)%2)

    def features(self,w,key):
        return (self.scene(w),w['previous']) if self.kernel.learner.factors else ()

    def terminal(self,s):
        w=s.public
        return w['tick']>=self.length or (self.game=='combat' and (w['hp']<=0 or w['enemy']<=0)) or (self.game=='gambling' and w['cash']<=0)

    def keys(self,s):
        w=s.public
        if self.terminal(s): return ()
        if self.game=='auction': return tuple('bid:'+str(b) for b in range(min(4,w['position']['budgets'][0])+1))
        if self.game=='resources': return ('trade','crop:0','crop:1','crop:2')+(('invest',) if w['cash']>=2 and w['plant']<2 else ())
        if self.game=='combat': return ('guard','strike:0','strike:1','strike:2')
        return ('pass','bet:0','bet:1','bet:2')

    def symbols(self,w):
        return tuple(str(i) for i in range(min(3,w['position']['budgets'][1])+1)) if self.game=='auction' else ('0','1','2')

    def models(self,w,key):
        legal=self.symbols(w)
        return {name:{s:float(s==str(min(index,len(legal)-1))) for s in legal}
                for name,index in (('library-a',0),('library-b',1))}

    def goal(self,s):
        w=s.public
        if self.game=='auction':
            p=w['position']; scores=[x+.25*y for x,y in zip(p['scores'],p['budgets'])]
            value=(scores[0]-max(scores[1:]))/27
        elif self.game=='resources': value=(w['cash']+2*w['plant']-6)/30
        elif self.game=='combat': value=(8-w['enemy']-(4-w['hp']))/12
        else: value=(w['cash']-6)/12
        return Goal(self.target,'scored' if self.terminal(s) else 'running',max(-1.,min(1.,value)))

    def events(self,w,key):
        rows=[]
        for symbol in self.symbols(w):
            z=int(symbol); after=copy.deepcopy(w);after['tick']+=1;after['signal']=symbol
            after['previous']=str(int(key.split(':')[-1])%2) if ':' in key else key
            gain=0.; safety=0.; style=0.
            if self.game=='auction':
                p=public_from_record(w['position']); bid=int(key.split(':')[1])
                after_p,win=settle(p,(bid,z,min(1,p.budgets[2])))
                if after_p.remaining: after_p=replace(after_p,prize=after_p.remaining[0],remaining=after_p.remaining[1:])
                after['position']=json.loads(json.dumps(asdict(after_p)))
                gain=(after_p.scores[0]-p.scores[0]-.25*(p.budgets[0]-after_p.budgets[0]))/12
                style=bid/max(1,p.budgets[0])-.5
            elif self.game=='resources':
                if key=='invest':after['cash']-=2;after['plant']+=1
                elif key=='trade':after['cash']+=1
                else:after['cash']+=(3+w['plant'] if int(key[-1])==z else -1)
                gain=(after['cash']-w['cash'])/8;safety=min(0.,gain)
                style=.2 if key=='invest' else 0.
            elif self.game=='combat':
                if key=='guard':after['hp']=min(4,w['hp']+1)
                elif int(key[-1])==z:after['enemy']=max(0,w['enemy']-2)
                else:after['hp']-=1
                safety=(after['hp']-w['hp'])/4
                gain=(w['enemy']-after['enemy'])/4+safety
                style=-.2 if key=='guard' else .2
            else:
                if key!='pass':after['cash']+=2 if int(key[-1])==z else -1
                gain=(after['cash']-w['cash'])/6;safety=min(0.,gain)
                style=0. if key=='pass' else .1
            flow=effect(gain,needs={'growth':gain,'safety':safety},values={'achievement':gain,'security':safety},style={'openness':style})
            rows.append(PublicEvent(symbol,after,flow))
        return rows

    def observe(self,s,state=None):
        choices=[]
        for key in self.keys(s):
            events,_,_,_,p=self.kernel.forecast(s,key);outs=[]
            for row in events:
                out=copy.deepcopy(row.flow);out['p']=p[row.symbol];outs.append(out)
            choices.append(action(key,*outs))
        c=context('open-model',choices,{'growth':.25,'safety':.2},self.profile['values'],dict(zip(TRAITS,self.profile['traits'])),mode=None)
        c.update(tick=s.tick,scope=dict(game='open-model-'+self.game,episode=self.model_id+':'+str(s.public['episode']),npc='owner'),
                 facts=dict(public=digest(s.public),memory=digest(self.kernel.learner.record(s.memory)),model=self.model_id))
        for name in NEEDS:
            if name not in ('growth','safety'):c['needs'][name]=dict(enabled=False,supported=False,deficit=None)
        if state is not None:c['state']=copy.deepcopy(state)
        return c

    def controller(self,horizon=1,update=True):
        k=copy.copy(self.kernel);k.update=update;k.identity=digest([k.state_identity,update])
        return ClosedLoop(observe=self.observe,transitions=k,terminal=self.terminal,assess=lambda s:self.goal(s).record(),
            record=self.kernel.record,request_for=lambda c:request_for(c,'open-model'),feedback=lambda *args:ModelFeedback(None),
            unit='owner-decisions',target=self.target,model_id=k.identity,budget=TreeBudget(horizon=horizon,nodes=100000,outcomes=4096))
