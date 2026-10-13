"""Four declared finite rule families for information-value experiments.

Auction uses the existing three-player first-price settlement. Other families
are authored diagnostic games, not new names for one shared payoff table.
Only this adapter knows game mechanics. The core receives public event laws.
"""
import copy
from dataclasses import asdict, replace
from reflex.belief_planning import BeliefKernel, HiddenOutcome
from reflex.closed_loop import ClosedLoop, TreeBudget
from reflex.contests import Public, settle, public_from_record
from reflex.core import TRAITS, NEEDS, digest
from reflex.examples import context, action, effect
from reflex.integrated_trials import request_for
from reflex.model_observer import ModelFeedback
from reflex.purpose_plan import Goal


class LearningGame:
    names = ('type-a', 'type-b')
    target = 'declared-game-purpose'

    def __init__(self, case, profile):
        self.case, self.profile = copy.deepcopy(case), copy.deepcopy(profile)
        self.game = case['game']; self.limit = case.get('limit', 3)
        self.model_id = digest(['learning-value-rules-v2-consistent-costs', case])
        self.kernel = self.make_kernel()

    def make_kernel(self, update=True):
        return BeliefKernel(names=self.names, model=self.model, public_record=lambda w:w,
                            model_id=self.model_id, owner='experiment-owner', update=update)

    def start(self):
        g=self.game
        if g=='auction':
            prizes=self.case.get('prizes',(2,7,4)); budget=self.case.get('budget',6)
            p=Public('auction',((),)*3,(0,)*3,(budget,)*3,prizes[0],'science',tuple(prizes[1:]))
            public=dict(tick=0,position=asdict(p))
        elif g=='resources':public=dict(tick=0,cash=self.case.get('cash',6),plant=-1,signal=None)
        elif g=='combat':public=dict(tick=0,hp=self.case.get('hp',2),enemy=self.case.get('enemy',4),signal=None)
        elif g=='gambling':public=dict(tick=0,cash=self.case.get('cash',3),signal=None)
        else:raise ValueError('registered diagnostic rules required')
        return self.kernel.start(public,self.case.get('prior',(.5,.5)))

    def terminal(self,s):
        w=s.public
        return w['tick']>=self.limit or (self.game=='combat' and (w['hp']<=0 or w['enemy']<=0)) or (self.game=='gambling' and (w['cash']<=0 or w['cash']>=self.case.get('target',7)))

    def keys(self,s):
        w=s.public
        if self.terminal(s):return ()
        if self.game=='auction':return tuple('bid:'+str(k) for k in public_from_record(w['position']).legal(0))
        if self.game=='combat':return ('feint','strike-a','strike-b','guard')
        if self.game=='resources':
            return ('trade',)+ (('survey',) if w['cash']>=self.case.get('fee',1) else ()) + (('build-a','build-b') if w['cash']>=2 else ()) + (('operate',) if w['plant']>=0 else ())
        return ('pass',)+ (('probe','bet-a','bet-b') if w['cash']>=max(1,self.case.get('fee',1)) else ('bet-a','bet-b'))

    def goal(self,s):
        w=s.public;end=self.terminal(s)
        if self.game=='auction':
            p=public_from_record(w['position']);scores=[x+.25*y for x,y in zip(p.scores,p.budgets)]
            v=max(-1.,min(1.,(scores[0]-max(scores[1:]))/sum(self.case.get('prizes',(2,7,4)))))
        elif self.game=='resources':v=max(-1.,min(1.,(w['cash']-6)/10))
        elif self.game=='combat':
            if end:return Goal(self.target,'success' if w['enemy']<=0 and w['hp']>0 else 'failure',1. if w['enemy']<=0 and w['hp']>0 else -1.)
            v=(self.case.get('enemy',4)-w['enemy'])/max(1,self.case.get('enemy',4))-.1*(self.case.get('hp',2)-w['hp'])
        else:v=max(-1.,min(1.,(w['cash']-self.case.get('cash',3))/6))
        return Goal(self.target,'scored' if end else 'running',max(-1.,min(1.,v)))

    def _events(self,h,w,key):
        """Public event and observed effect distribution before hidden drift."""
        before=copy.deepcopy(w);w=copy.deepcopy(w);w['tick']+=1
        g=self.game;z=self.names.index(h);noise=self.case.get('signal_accuracy',.9)
        fee=self.case.get('fee',1)
        if g=='auction':
            p=public_from_record(before['position']);b=int(key.split(':')[1]);nominal=(1,3)[z]
            bids=(b, min(p.legal(1),key=lambda k:abs(k-nominal)), min(p.legal(2),key=lambda k:abs(k-max(0,nominal-1))))
            after,win=settle(p,bids)
            if after.remaining:after=replace(after,prize=after.remaining[0],remaining=after.remaining[1:])
            w['position']=asdict(after)
            gain=(after.scores[0]-p.scores[0]-.25*(p.budgets[0]-after.budgets[0]))/14
            yield 1.,w,effect(gain,values={'achievement':gain},style={'openness':b/max(1,p.budgets[0])-.5})
        elif key in ('survey','feint','probe'):
            if g!='combat':w['cash']-=fee
            for signal,p in ((z,noise),(1-z,1-noise)):
                if p:
                    v=copy.deepcopy(w);v['signal']=signal
                    yield p,v,self.money_flow(-fee) if g!='combat' else effect()
        elif g=='resources':
            if key.startswith('build'):w['cash']-=2;w['plant']=int(key.endswith('b'))
            if key=='operate':w['cash']+=(4 if w['plant']==z else self.case.get('poor_yield',0))
            if key=='trade':w['cash']+=self.case.get('trade',1)
            yield 1.,w,self.money_flow(w['cash']-before['cash'])
        elif g=='combat':
            if key=='guard':w['hp']=min(self.case.get('hp',2),w['hp']+1)
            elif int(key.endswith('b'))==z:w['enemy']=max(0,w['enemy']-2)
            else:w['hp']=max(0,w['hp']-1)
            gain=(before['enemy']-w['enemy'])/4;loss=before['hp']-w['hp']
            yield 1.,w,effect(gain-loss/4,needs={'safety':-loss/4},values={'achievement':gain,'security':-loss/4},style={'openness':.2 if key.startswith('strike') else -.2})
        else:
            if key=='pass':yield 1.,w,effect();return
            probability=self.case.get('good_odds',.85) if int(key.endswith('b'))==z else self.case.get('bad_odds',.15)
            for p,delta in ((probability,2),(1-probability,-1)):
                if p:
                    v=copy.deepcopy(w);v['cash']+=delta
                    yield p,v,self.money_flow(delta)

    def money_flow(self,delta):
        # Cash changes have the same meaning for productive, speculative, and
        # information actions. No free curiosity/style or missing fee penalty.
        gain=delta/(10 if self.game=='resources' else 6)
        return effect(gain,needs={'growth':gain,'safety':min(0.,gain)},values={'achievement':gain})

    def model(self,h,w,key):
        probe=self.kernel.start(w,(.5,.5))
        if key not in self.keys(probe):raise ValueError('legal live action required')
        flip=self.case.get('flip',0.);other=self.names[1-self.names.index(h)];rows=[]
        for p,public,flow in self._events(h,w,key):
            for chance,next_h in ((1-flip,h),(flip,other)):
                if chance:rows.append(HiddenOutcome(p*chance,public,flow,next_h))
        return rows

    def observe(self,s,state=None):
        choices=[]
        for key in self.keys(s):
            outs=[]
            for t in self.kernel(s,key):
                row=copy.deepcopy(t.flow);row['p']=t.probability;outs.append(row)
            choices.append(action(key,*outs))
        c=context('learning-value',choices,{'growth':.25,'safety':.2},self.profile['values'],dict(zip(TRAITS,self.profile['traits'])),mode=None)
        c.update(tick=s.tick,scope=dict(game='learning-value-'+self.game,episode=self.model_id,npc='experiment-owner'),
                 facts=dict(public=digest(s.public),belief=str(s.weights),model=self.model_id))
        for name in NEEDS:
            if name not in ('growth','safety'):c['needs'][name]=dict(enabled=False,supported=False,deficit=None)
        if state is not None:c['state']=copy.deepcopy(state)
        return c

    def controller(self,arm,*,horizon=None):
        kernel=self.make_kernel(update=arm!='frozen')
        return ClosedLoop(observe=self.observe,transitions=kernel,terminal=self.terminal,
                          assess=lambda s:self.goal(s).record(),record=self.kernel.record,
                          request_for=lambda c:request_for(c,'learning-value'),feedback=lambda *args:ModelFeedback(None),
                          unit='owner-decisions',target=self.target,model_id=kernel.identity,
                          budget=TreeBudget(horizon=horizon or self.limit,nodes=100000,outcomes=4096))
