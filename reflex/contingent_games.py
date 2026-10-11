"""Evaluation bridges, OUTSIDE the generic conditional planning core.

Rules, perception, opponents, endpoint proxies and own-decision intervals are
explicit game responsibilities. This is a coverage suite, not automatic rule
understanding, a universal strength score or full team-plan negotiation.
"""
from dataclasses import dataclass,replace,asdict,is_dataclass,fields
import copy
import numpy as np
from .core import Policy,digest,TRAITS
from .examples import action,effect
from .purpose_plan import Goal
from .public_perception import with_public


def rng(seed,*public):
    return np.random.default_rng(int(digest(['conditional-game-draw',seed,*public])[:16],16))


def settled(target,won,finished,proxy):
    if won is not None:return Goal(target,'success' if won else 'failure',1. if won else -1.)
    return Goal(target,'draw' if finished else 'running',0. if finished else float(np.clip(proxy,-1,1)))


def immutable_position(value):
    if value is None or type(value) in (str,int,float,bool):return True
    if isinstance(value,tuple):return all(immutable_position(v) for v in value)
    return is_dataclass(value) and value.__dataclass_params__.frozen and all(immutable_position(getattr(value,f.name)) for f in fields(value))


@dataclass(frozen=True)
class Frame:
    position: object
    turns: int=0


class Bridge:
    target='game-purpose';unit='owner-decisions';seeds=(401,402);limit=100
    def __init__(self,profile,case):
        self.profile=copy.deepcopy(profile);self.case=copy.deepcopy(case);self.actor_seed=42
        self.episode='conditional-coverage';self._contexts={};self._transitions={}
    def project(self,w):return copy.deepcopy(w)
    def record(self,w):return dict(position=asdict(w.position),turns=w.turns)
    def observe(self,w,state=None):
        # Immutable rule positions can reuse an immediate observation. Actor
        # memory is attached afterwards, so no branch shares mutable intent.
        key=w if immutable_position(w) else None
        if key is None:c=with_public(self.context(w,None),self.record(w))
        else:
            if key not in self._contexts:
                if len(self._contexts)>=4096:self._contexts.clear()
                self._contexts[key]=with_public(self.context(w,None),self.record(w))
            c=copy.deepcopy(self._contexts[key])
        if state is not None:c['state']=copy.deepcopy(state)
        return c
    def actual(self,w,key,seed):return self.advance(w,key,seed)
    def modeled_advance(self,w,key,seed):
        cache_key=(w,key,seed) if immutable_position(w) else None
        if cache_key is None:return self.advance(w,key,seed)
        if cache_key not in self._transitions:
            if len(self._transitions)>=4096:self._transitions.clear()
            self._transitions[cache_key]=self.advance(w,key,seed)
        return copy.deepcopy(self._transitions[cache_key])


class CrossBridge(Bridge):
    def start(self):
        from .cross_games import Navigation,Duel,Project
        cls={'navigation':Navigation,'duel':Duel,'project':Project}[self.case['game']]
        return Frame(cls(self.profile,42,self.case.get('variant',0)))
    def record(self,w):return dict(public=w.position.observation(),history=list(w.position.observations),turns=w.turns)
    def project(self,w):
        q=copy.deepcopy(w)
        if self.case['game']=='duel':q.position.variant=0
        q.position.seed=42
        return q
    def terminal(self,w):return w.position.done(w.position.state) or w.turns>=40
    def context(self,w,state):
        q=copy.deepcopy(w.position);q.memory=copy.deepcopy(state);q.tick=w.turns
        return q.observe()
    def advance(self,w,key,seed):
        q=copy.deepcopy(w.position);before=copy.deepcopy(q.state)
        options=q.branches(key);u=rng(seed,w.turns,'cross').random();j=min(int(np.searchsorted(np.cumsum([p for p,_ in options]),u)),len(options)-1)
        q.state=copy.deepcopy(options[j][1]);q.tick+=1
        if self.case['game']=='duel':q.observations.append(q.moves[j])
        row=q.features(before,q.state,key,1.)
        return Frame(q,w.turns+1),row
    def actual(self,w,key,seed):
        if self.case['game']!='duel':return self.advance(w,key,seed)
        q=copy.deepcopy(w.position);before=copy.deepcopy(q.state);q.seed=seed;q.advance(key)
        q.seed=42
        return Frame(q,w.turns+1),q.features(before,q.state,key,1.)
    def goal(self,w):
        q=w.position;s=q.state
        if self.case['game']=='duel':won=True if s['opponent_hp']<=0 and s['hp']>0 else False if s['hp']<=0 else None
        elif self.case['game']=='project':won=True if q.complete(s) else False if s['time_left']<=0 else None
        else:won=True if q.done(s) else None
        return settled(self.target,won,self.terminal(w),q.progress(s)/2)


class ConnectBridge(Bridge):
    def start(self):
        from .board_models import ConnectPosition
        return Frame(ConnectPosition())
    def terminal(self,w):
        from .board_models import ConnectAdapter
        return ConnectAdapter().terminal(w.position)
    def rival(self,s,seed,turn):
        from .board_models import ConnectAdapter,connect_features
        options=ConnectAdapter().order(s,s.legal())
        # Public tactical baseline: win/block ordering, then one-step geometry.
        winning=[k for k in options if s.play(k).winner()==s.turn]
        unsafe={k for k in options if any(s.play(k).play(r).winner()==1 for r in s.play(k).legal())}
        safe=[k for k in options if k not in unsafe]
        pool=winning or safe or options
        scores=[connect_features(s.play(k),1)[0] for k in pool];best=max(scores)
        ties=[k for k,v in zip(pool,scores) if abs(v-best)<1e-12]
        return ties[int(rng(seed,turn,'connect-rival').integers(len(ties)))]
    def context(self,w,state):
        from .board_models import ConnectAdapter
        s=w.position;a=ConnectAdapter()
        choices=[action(k,a.consequence(s,s.play(k),0)) for k in s.legal()]
        return a.context(s,self.profile,42,w.turns,self.episode,choices,state)
    def advance(self,w,key,seed):
        from .board_models import ConnectAdapter,connect_referee
        s=w.position;after=s.play(key)
        assert connect_referee(after)[0]==after.winner()
        if not ConnectAdapter().terminal(after):after=after.play(self.rival(after,seed,w.turns))
        assert connect_referee(after)[0]==after.winner()
        row=ConnectAdapter().consequence(s,after,0)
        # Existing adapter emits a position stock; a flow must be a delta.
        old=ConnectAdapter().consequence(s,s,0)
        for k,v in list(row['values'].items()):row['values'][k]=float(np.clip(v-old['values'].get(k,0),-1,1))
        row['objective']=float(np.clip(row['objective']-old['objective'],-1,1))
        return Frame(after,w.turns+1),row
    def goal(self,w):
        from .board_models import connect_features
        winner=w.position.winner()
        return settled(self.target,None if winner is None else winner==0,self.terminal(w),connect_features(w.position,0)[0])


class ReversiBridge(Bridge):
    def start(self):
        from .reversi import opening
        s,_=opening(self.case.get('opening',0),6)
        return Frame(s)
    def terminal(self,w):return w.position.terminal()
    def context(self,w,state):
        from .reversi import observe
        return observe(w.position,self.profile,42,self.episode,w.turns,state,depth=1)[0]
    def advance(self,w,key,seed):
        from .reversi import parse_move,baseline,audit_move,consequences
        s=w.position;side=s.side;square=parse_move(key);after=s.play(square);audit_move(s,square,after)
        if not after.terminal():
            move=baseline(after,'minimax2',seed,w.turns);nxt=after.play(move);audit_move(after,move,nxt);after=nxt
        row=consequences(s,after,side);old=consequences(s,s,side)
        row['objective']=float(np.clip(row['objective']-old['objective'],-1,1))
        for k,v in list(row['values'].items()):row['values'][k]=float(np.clip(v-old['values'].get(k,0),-1,1))
        return Frame(after,w.turns+1),row
    def goal(self,w):
        from .reversi import utility
        s=w.position;score=s.black.bit_count()-s.white.bit_count()
        won=(score>0 if score else None) if s.terminal() else None
        return settled(self.target,won,self.terminal(w),utility(s,1))


class QuoridorBridge(Bridge):
    def start(self):
        from .quoridor import Position
        return Frame(Position.start(self.case.get('players',4)))
    def terminal(self,w):return w.position.winner() is not None or w.turns>=60
    def context(self,w,state):
        from .quoridor import snapshot
        return snapshot(w.position,self.profile,42,w.turns,self.episode,state)[0]
    def rival(self,s,seed,turn):
        from .quoridor import consequences
        names=s.legal();scores=[consequences(s,s.play(k,validated=True))[0]['objective'] for k in names]
        best=max(scores);ties=[k for k,v in zip(names,scores) if abs(v-best)<1e-12]
        return ties[int(rng(seed,turn,s.turn,'quoridor-rival').integers(len(ties)))]
    def advance(self,w,key,seed):
        from .quoridor import consequences,audit
        s=w.position;after=s.play(key);audit(s,key,after)
        while after.winner() is None and after.turn!=0:
            k=self.rival(after,seed,w.turns);nxt=after.play(k,validated=True);audit(after,k,nxt);after=nxt
        # Authored flow must be from the owner's decision, not the last rival.
        row=consequences(s,after)[0]
        return Frame(after,w.turns+1),row
    def goal(self,w):
        s=w.position;winner=s.winner();dist=s.distances()
        return settled(self.target,None if winner is None else winner==0,self.terminal(w),(min(dist[1:])-dist[0])/9)


class ThanksBridge(Bridge):
    def __init__(self,profile,case):
        super().__init__(profile,case);self.actual_deck=tuple(int(c) for c in rng(case.get('deal',0),'thanks-deal').permutation(np.arange(3,36))[:24])
    def start(self):
        from .board_models import ThanksPosition
        return Frame(ThanksPosition.start(self.case.get('players',4),self.actual_deck[0]))
    def terminal(self,w):return w.position.card is None and not w.position.remaining
    def rival(self,s,seed,turn):
        from .board_models import card_points
        gain=card_points(s.cards[s.turn]+(s.card,))-card_points(s.cards[s.turn])
        return 'TAKE' if len(s.legal())==1 or gain<=s.pot else 'PASS'
    def context(self,w,state):
        from .board_models import ThanksAdapter
        s=w.position;a=ThanksAdapter()
        choices=[action(k,a.consequence(s,s.play(k),0)) for k in s.legal()]
        return a.context(s,self.profile,42,w.turns,self.episode,choices,state)
    def transition(self,w,key,seed,actual):
        from .board_models import ThanksAdapter,thanks_referee_score
        base=w.position;s=base.play(key);guard=0
        while True:
            if s.card is None:
                if not s.remaining:break
                # Actual order belongs only to the driver. Model draws from
                # the public unseen set, accounting for the nine removals.
                unseen=[c for c in range(3,36) if c not in s.seen]
                card=self.actual_deck[len(s.seen)] if actual else int(rng(seed,s.seen,'thanks-reveal').choice(unseen))
                s=s.draw(card)
            if s.turn==0:break
            s=s.play(self.rival(s,seed,w.turns));guard+=1
            if guard>400:raise AssertionError('finite chip/card ledger failed to return to owner')
        assert s.scores()==tuple(thanks_referee_score(c,m) for c,m in zip(s.cards,s.chips))
        assert sum(s.chips)+s.pot==sum(base.chips)+base.pot
        return Frame(s,w.turns+1),ThanksAdapter().consequence(base,s,0)
    def advance(self,w,key,seed):return self.transition(w,key,seed,False)
    def actual(self,w,key,seed):return self.transition(w,key,seed,True)
    def goal(self,w):
        scores=w.position.scores();own=scores[0];best=min(scores)
        won=own==best if self.terminal(w) and scores.count(best)==1 else False if self.terminal(w) and own>best else None
        return settled(self.target,won,self.terminal(w),(min(scores[1:])-own)/100)


class GoofspielBridge(Bridge):
    def start(self):
        from .goofspiel import Position
        return Frame(Position.start(self.case.get('players',4),self.case.get('cards',7),self.case.get('order','ascending')))
    def terminal(self,w):return w.position.terminal
    def context(self,w,state):
        from .goofspiel import make_context
        return make_context(w.position,0,self.profile,'win_share',42,w.turns,self.episode,state)
    def advance(self,w,key,seed):
        from .goofspiel import reserve_bid,consequence,referee
        s=w.position;bids=(int(key.split(':')[1]),)+tuple(reserve_bid(s,i,rng(seed,s.round,i,'sealed-bid')) for i in range(1,len(s.hands)))
        after=s.play(bids);referee(s,bids,after)
        return Frame(after,w.turns+1),consequence(s,after.scores,0,bids[0],'score',False)
    def goal(self,w):
        s=w.position;best=max(s.scores);won=s.scores[0]==best if s.terminal and s.scores.count(best)==1 else False if s.terminal and s.scores[0]<best else None
        return settled(self.target,won,s.terminal,(s.scores[0]-max(s.scores[1:]))/sum(s.prizes))


class ContestBridge(Bridge):
    def __init__(self,profile,case):
        super().__init__(profile,case)
        values=tuple(case.get('prizes',(-5,-2,1,3,6,8,10) if case['game']=='hagetaka' else (4,12,7,14,5)))
        self.deck=tuple(int(x) for x in rng(case.get('deal',0),'contest-deck').permutation(values))
        self.total=sum(abs(v) for v in values)+1
    def start(self):
        from .contests import Public
        n=self.case.get('players',4);h=tuple(range(1,16)) if self.case['game']=='hagetaka' else ()
        return Frame(Public(self.case['game'],(h,)*n,(0,)*n,(12,)*n,self.deck[0],'science',tuple(sorted(self.deck[1:]))))
    def terminal(self,w):return w.turns==len(self.deck)
    def context(self,w,state):
        from .contests import make_context
        return make_context(w.position,0,self.profile,42,w.turns,state,samples=8)[0]
    def transition(self,w,key,seed,actual):
        from .contests import settle,target,evaluate
        s=w.position;r=rng(seed,s.round,'contest-sealed')
        # Stated rival hypothesis: value-scaled public bidding with near ties.
        bids=[int(key.split(':')[1])]
        for i in range(1,len(s.scores)):
            nominal=target(s,i,'value');h=s.legal(i);pool=[b for b in h if abs(b-nominal)<=1] or [nominal]
            bids.append(int(r.choice(pool)))
        after,_=settle(s,tuple(bids));row=evaluate(s,0,bids[0],np.array([bids,bids]))[0][0];row['p']=1.
        if s.remaining:
            prize=self.deck[w.turns+1] if actual else int(rng(seed,s.round,'contest-reveal').choice(s.remaining))
            remains=list(s.remaining);remains.remove(prize);after=replace(after,prize=prize,remaining=tuple(sorted(remains)))
        return Frame(after,w.turns+1),row
    def advance(self,w,key,seed):return self.transition(w,key,seed,False)
    def actual(self,w,key,seed):return self.transition(w,key,seed,True)
    def goal(self,w):
        s=w.position;scores=tuple(p+(.25*b if s.game=='auction' else 0) for p,b in zip(s.scores,s.budgets))
        best=max(scores);won=scores[0]==best if self.terminal(w) and scores.count(best)==1 else False if self.terminal(w) and scores[0]<best else None
        return settled(self.target,won,self.terminal(w),(scores[0]-max(scores[1:]))/self.total)


class ProceduralBridge(Bridge):
    """The prior public graph diagnostic is also exercised, without refitting."""
    def __init__(self,profile,case):
        super().__init__(profile,case)
        from tools.run_procedural_stability import Trial
        self.rules=Trial(case.get('graph',12100),profile,case.get('length',5))
    def start(self):return Frame((0,0))
    def record(self,w):return dict(public_node=list(w.position),turns=w.turns)
    def terminal(self,w):return self.rules.terminal(w.position)
    def context(self,w,state):return self.rules.observe(w.position,state)
    def advance(self,w,key,seed):
        after,row=self.rules.advance(w.position,key,seed)
        return Frame(after,w.turns+1),row
    def goal(self,w):
        g=self.rules.goal(w.position)
        return Goal(self.target,g.status,g.value)


class TinyBridge(Bridge):
    """Original two integration rules, with an explicit finite scored harness."""
    def start(self):
        resource=self.case['game']=='tiny_resources'
        return Frame(dict(time=6,food=2,recovered=0.,skill=0.) if resource else dict(energy=2,score=0.))
    def record(self,w):return dict(public=copy.deepcopy(w.position),turns=w.turns)
    def terminal(self,w):return w.turns>=12
    def context(self,w,state):
        from .adapters import ResourceAdapter,ArenaAdapter,profile
        p=profile();p.update(traits=dict(zip(TRAITS,self.profile['traits'])),values={k:self.profile['values'].get(k,0.) for k in p['values']})
        if state is not None:p['state']=copy.deepcopy(state)
        adapter=ResourceAdapter() if self.case['game']=='tiny_resources' else ArenaAdapter()
        return adapter.observe(w.position,p,'actor',w.turns)
    def advance(self,w,key,seed):
        from .adapters import ResourceAdapter,ArenaAdapter
        c=self.context(w,None);row=copy.deepcopy(next(a for a in c['actions'] if a['id']==key)['outcomes'][0]);row['p']=1.
        if self.case['game']=='tiny_resources':after=ResourceAdapter().transition(w.position,{'action_id':key})
        else:
            after=ArenaAdapter().transition(w.position,{'action_id':key},rng(seed,w.turns,'tiny-arena'))
            row['objective']=after['score']-w.position['score']
        return Frame(after,w.turns+1),row
    def goal(self,w):
        s=w.position;value=(s['skill']+s['recovered']+.1*s['food'])/6 if self.case['game']=='tiny_resources' else s['score']/12
        return Goal(self.target,'scored' if self.terminal(w) else 'running',float(np.clip(value,-1,1)))


class EmpireBridge(Bridge):
    """Full multi-resource/victory race with public economic opponents."""
    def start(self):
        from .resource_world import World
        return Frame(World.start(players=self.case.get('players',4),limit=36,
            routes=tuple(self.case.get('routes',('science','culture','power')))))
    def terminal(self,w):
        from .resource_world import terminal
        return terminal(w.position)
    def context(self,w,state):
        from .resource_world import make_context
        return make_context(w.position,self.profile,42,w.turns,self.episode,state,1)[0]
    def advance(self,w,key,seed):
        from .resource_world import step,economic_move,terminal,consequence
        s=w.position;after=step(s,key)
        while not terminal(after) and after.turn!=0:after=step(after,economic_move(after))
        return Frame(after,w.turns+1),consequence(s,after,0)
    def actual(self,w,key,seed):
        after,row=self.advance(w,key,seed)
        # A driver-only future change is revealed through the resulting world.
        # The planner receives only the current announced production constants.
        if self.case.get('change') and w.position.round<8<=after.position.round:
            after=replace(after,position=replace(after.position,food_yield=1,ore_yield=2,trade_price=1))
            from .resource_world import consequence
            row=consequence(w.position,after.position,0)
        return after,row
    def goal(self,w):
        from .resource_world import winners,potential
        s=w.position;ws=winners(s);won=(0 in ws) if ws else None
        return settled(self.target,won,self.terminal(w),potential(s,0))


class TrapBridge(Bridge):
    """Existing synthetic rule engines with separate public-history hypotheses.

Synthetic step inputs instantiate a declared current opponent hypothesis;
they never use the actual future driver phase. Exact result/history updates
stay in the original rule engines. Empirical outcome-learning is not enabled.
"""
    def start(self):
        from .judgment_experiment import Arena,Commons,Routes
        return Frame({'cyclic_arena':Arena,'shared_commons':Commons,'branching_delivery':Routes}[self.case['game']]())
    def record(self,w):return dict(public=copy.deepcopy(vars(w.position)),turns=w.turns)
    def terminal(self,w):return w.turns>=60
    def context(self,w,state):
        from .judgment_experiment import snapshot
        return snapshot(w.position,self.profile,42,w.turns,state)
    def advance(self,w,key,seed):
        from .judgment_experiment import Arena,Commons,Routes
        from collections import Counter
        q=copy.deepcopy(w.position);random=rng(seed,w.turns,'public-trap-hypothesis')
        if isinstance(q,Arena):
            counts=Counter(q.history[-4:]);weights=np.array([1+counts[k] for k in q.moves],dtype=float);weights/=sum(weights)
            index=int(random.choice(3,p=weights));row,_=q.step(key,0,index)
        elif isinstance(q,Commons):
            revealed=q.history[-4:];claims=sum(h==('CLAIM',)*3 for h in revealed)
            all_claims=random.random()<(1+claims)/(2+len(revealed));row,_=q.step(key,0 if all_claims else 20,0)
        elif isinstance(q,Routes):
            blocked=q.survey
            if blocked is None:
                blocked=next((k for k,room in reversed(q.history) if k in ('LEFT','RIGHT') and room=='blocked'),None)
            if blocked is None:blocked=('LEFT','RIGHT')[int(random.integers(2))]
            row,_=q.step(key,0 if blocked=='LEFT' else 20,0)
        return Frame(q,w.turns+1),row
    def actual(self,w,key,seed):
        q=copy.deepcopy(w.position);row,_=q.step(key,w.turns+self.case.get('phase_offset',0),seed)
        return Frame(q,w.turns+1),row
    def goal(self,w):
        return Goal(self.target,'scored' if self.terminal(w) else 'running',float(np.clip(w.position.score/60,-1,1)))


class GamblingBridge(Bridge):
    def start(self):
        from .gambling import Table
        return Frame((Table(limit=16),self.case.get('scenario','opportunity')))
    def project(self,w):
        from .gambling import offers
        table,scenario=w.position
        regime='opportunity' if offers(table,scenario)['safe'][2]==.8 else 'unfavorable'
        return Frame((table,regime),w.turns)
    def record(self,w):
        from .gambling import offers
        table,regime=w.position
        return dict(table=asdict(table),current_offers=offers(table,regime),turns=w.turns)
    def terminal(self,w):
        from .gambling import terminal
        return terminal(w.position[0])
    def context(self,w,state):
        from .gambling import make_context
        table,regime=w.position;c=make_context(table,regime,self.profile,42,state)
        c['scope']['episode']=self.episode
        return c
    def advance(self,w,key,seed):
        from .gambling import resolve,offers
        table,regime=w.position;after,delta=resolve(table,regime,key,seed)
        row=effect(delta/12,needs={'growth':float(np.clip(delta/6,-1,1)),'safety':max(0,(6-table.cash)/6)-max(0,(6-after.cash)/6)},
            values={'achievement':float(np.clip(delta/6,-1,1)),'power':float(np.clip(delta/6,-1,1)),
                    'security':max(0,(6-table.cash)/6)-max(0,(6-after.cash)/6)})
        return Frame((after,regime),w.turns+1),row
    def goal(self,w):
        table,_=w.position
        won=True if table.cash>=18 else False if table.cash==0 else None
        return settled(self.target,won,self.terminal(w),(table.cash-12)/18)


class TeamProjectsBridge(Bridge):
    def start(self):
        from .team_projects import Workshop
        return Frame(Workshop.start(self.case.get('scenario','balanced'),self.case.get('members',3),14))
    def terminal(self,w):
        from .team_projects import terminal
        return terminal(w.position)
    def context(self,w,state):
        from .team_projects import make_context
        return make_context(w.position,0,self.profile,42,state)
    def advance(self,w,key,seed):
        from .team_projects import greedy,resolve,consequence
        s=w.position;ours=greedy(s,0,{0:key});theirs=greedy(s,1);after,_=resolve(s,{**ours,**theirs})
        return Frame(after,w.turns+1),consequence(s,after,0,key)
    def goal(self,w):
        from .team_projects import winners,purpose
        s=w.position;ws=winners(s);won=0 in ws if len(ws)==1 else None
        return settled(self.target,won,self.terminal(w),purpose(s,0))


class TeamCombatBridge(Bridge):
    def start(self):
        from .combat import Battle
        return Frame(Battle.start(self.case.get('map','cover'),self.case.get('goal','either'),24))
    def terminal(self,w):
        from .combat import terminal
        return terminal(w.position)
    def context(self,w,state):
        from .combat import make_context
        return make_context(w.position,0,self.profile,42,'eliminate',state)
    def advance(self,w,key,seed):
        from .combat import resolve,reference,transition_effect,terminal
        s=w.position;chosen={i:reference(s,i,seed) for i,p in enumerate(s.units) if p.hp>0};chosen[0]=key
        after,_=resolve(s,chosen,seed)
        # Death ends this actor's decisions, not its team's match. Continue the
        # stated ally/rival policies to the actual shared victory/deadline.
        while after.units[0].hp<=0 and not terminal(after):
            chosen={i:reference(after,i,seed) for i,p in enumerate(after.units) if p.hp>0}
            after,_=resolve(after,chosen,seed)
        return Frame(after,w.turns+1),transition_effect(s,after,0,key,'eliminate')
    def goal(self,w):
        from .combat import victory,potential
        s=w.position;ws=victory(s);won=0 in ws if len(ws)==1 else None
        return settled(self.target,won,self.terminal(w),potential(s,0,'eliminate')/2)


class _ForcedPolicy:
    def __init__(self,key):self.key=key;self.base=Policy()
    def decide(self,b,stochastic=True,**kwargs):
        d=self.base.decide(b,stochastic,**kwargs)
        if stochastic:
            j=b.ids[0].index(self.key)
            if not b.legal[0,j] or b.known_failure[0,j]:raise ValueError('forced laboratory intent is not viable')
            chosen=d.action.copy();chosen[0]=j;d=replace(d,action=chosen)
        return d


class LaboratoryBridge(Bridge):
    def start(self):
        from .laboratory import Laboratory
        from .runtime import Population
        lab=Laboratory(42,self.case.get('scenario','ordinary'),reading=False);lab.roster[0]=copy.deepcopy(self.profile)
        c=lab.context_templates[0];c['personality']=dict(zip(TRAITS,self.profile['traits']));c['values'].update({k:self.profile['values'].get(k,0) for k in c['values']})
        lab.population=Population(lab.context_templates)
        return Frame(lab)
    def record(self,w):
        lab=w.position
        return dict(own=lab.actor_state(0),public_stock=lab.stock.tolist(),public_skill=lab.skill.tolist(),
            known_broken=bool(lab.known_broken[0]),past_claims=lab.observed[0].tolist(),
            past_response=lab.markov[0].tolist(),turns=w.turns)
    def project(self,w):
        from .runtime import Population
        lab=copy.deepcopy(w.position);lab.rules=dict(lab.rules,break_at=None,shock_at=None)
        lab.scenario='ordinary';lab.seed=42
        cs=copy.deepcopy(lab.context_templates)
        for i in range(1,lab.n):
            cs[i]['personality']={k:.5 for k in cs[i]['personality']};cs[i]['values']={k:0. for k in cs[i]['values']}
            lab.known_broken[i]=False
        old=lab.population;lab.population=Population(cs);lab.population.ticks[:]=old.ticks
        lab.environment_seeds=np.array([int(digest(['conditional-model-lab',i])[:16],16) for i in range(lab.n)],dtype=np.uint64)
        return Frame(lab,w.turns)
    def terminal(self,w):return w.turns>=20
    def context(self,w,state):
        lab=copy.deepcopy(w.position);data,*_=lab.inputs();c=lab.snapshot(0,data)
        if state is not None:c['state']=copy.deepcopy(state)
        return c
    def advance(self,w,key,seed):
        from .core import NEEDS
        from .laboratory import sparse_outcome,consequence
        lab=copy.deepcopy(w.position);before=lab.actor_state(0);lab.population.policy=_ForcedPolicy(key)
        lab.environment_seeds=np.array([int(digest(['conditional-lab-world',seed,i])[:16],16) for i in range(lab.n)],dtype=np.uint64)
        rows=lab.step();lab.population.policy=Policy();row=rows[0]
        f=consequence(key,before,success=row['cause'] not in ('risk_failed','observed_workshop_broken'),rival_claim=rows[int(lab.partner[0])]['action']=='CLAIM')
        flow=sparse_outcome(f,1.)
        # Include the actual upkeep/received donation in need FLOW exactly once.
        from .laboratory import deficits
        old=deficits(*(before[k] for k in ('energy','stock','hunger','social','esteem','growth')))
        after=lab.actor_state(0);new=deficits(*(after[k] for k in ('energy','stock','hunger','social','esteem','growth')))
        flow['needs']={k:float(v) for k,v in zip(NEEDS,np.clip(old-new,-1,1))}
        return Frame(lab,w.turns+1),flow
    def goal(self,w):
        lab=w.position;value=float(np.clip((lab.stock[0]/10+lab.skill[0]-.5*lab.hunger[0])/2,-1,1))
        return Goal(self.target,'scored' if self.terminal(w) else 'running',value)


class EncounterBridge(Bridge):
    def start(self):
        from .encounters import EncounterWorld
        w=EncounterWorld(42,self.case.get('domain','exchange'),self.case.get('layout','balanced4'),self.case.get('contact','public'))
        w.profiles['npc-0']=copy.deepcopy(self.profile)
        return Frame(w)
    def record(self,w):
        q=w.position
        return dict(states={a:q.state(a) for a in q.actors},history={a:list(q.history[a]) for a in q.actors},
            domain=q.domain,contact=q.contact,tick=q.tick,turns=w.turns)
    def project(self,w):
        out=copy.deepcopy(w);q=out.position;q.seed=42
        for a in q.actors[1:]:q.profiles[a]=dict(id='neutral',traits=(.5,)*5,values={})
        return out
    def terminal(self,w):return w.turns>=36 or w.position.health['npc-0']<=0
    def context(self,w,state):
        c=w.position.contexts()[0][0]
        if state is not None:c['state']=copy.deepcopy(state)
        return c
    def advance(self,w,key,seed):
        q=copy.deepcopy(w.position);contexts=q.contexts()[0]
        chosen={c['scope']['npc']:Policy().choose(c,stochastic=False)['action_id'] for c in contexts};chosen['npc-0']=key
        q.seed=seed;rows,_=q.resolve(chosen);q.seed=42;own=rows[0]
        row=q.perceived(own['kind'],own['other_kind'],own['matched'],1.)
        row['objective']=own['gain']
        row['needs']['safety']=float(np.clip((own['after']['health']-own['before']['health'])/8,-1,1))
        if own['luck_failure']:row['values']={};row['objective']=-.1
        return Frame(q,w.turns+1),row
    def goal(self,w):
        q=w.position
        if q.health['npc-0']<=0:return Goal(self.target,'failure',-1.)
        return Goal(self.target,'scored' if self.terminal(w) else 'running',float(np.clip(q.score['npc-0']/36,-1,1)))


@dataclass(frozen=True)
class SocialState:
    energy: float=.8
    reserve: float=3.
    health: float=8.
    score: float=0.
    history: tuple=()
    tick: int=0


class SocialProbeBridge(Bridge):
    def start(self):return Frame(SocialState())
    def terminal(self,w):return w.turns>=36 or w.position.health<=0
    def context(self,w,state):
        from .social_experiment import contexts
        from .laboratory import PROFILES
        s=w.position;i=next(i for i,p in enumerate(PROFILES) if p['id']==self.profile['id'])
        c=contexts(42,self.case.get('domain','allocation'),np.full(4,s.energy,dtype=float),np.full(4,s.reserve,dtype=float),
            np.full(4,s.health,dtype=float),[list(s.history) for _ in range(4)],s.tick)[i]
        if state is not None:c['state']=copy.deepcopy(state)
        return c
    def transition(self,w,key,seed,actual):
        s=w.position;domain=self.case.get('domain','allocation');p=(sum(s.history[-8:])+1)/(len(s.history[-8:])+2)
        r=rng(seed,s.tick,'social-outcome');success=bool(r.random()<p)
        if actual:
            schedule=self.case.get('schedule','change_and_repair')
            cooperate=schedule in ('cooperative','bad_luck') or schedule=='change_and_repair' and (s.tick<12 or s.tick>=24)
            success=cooperate and r.random()>=(.3 if schedule=='bad_luck' else .05)
        from .social_experiment import transition
        after,gain=transition(domain,asdict(s),key,success)
        after=SocialState(**after,history=(s.history+(int(success),))[-8:],tick=s.tick+1)
        a=next(a for a in self.context(w,None)['actions'] if a['id']==key)
        row=copy.deepcopy(a['outcomes'][0 if success or key!='aid' else 1]);row['p']=1.;row['objective']=gain
        return Frame(after,w.turns+1),row
    def advance(self,w,key,seed):return self.transition(w,key,seed,False)
    def actual(self,w,key,seed):return self.transition(w,key,seed,True)
    def goal(self,w):
        s=w.position
        if s.health<=0:return Goal(self.target,'failure',-1.)
        return Goal(self.target,'scored' if self.terminal(w) else 'running',float(np.clip(s.score/36,-1,1)))


BRIDGES={'navigation':CrossBridge,'duel':CrossBridge,'project':CrossBridge,
    'connect_four':ConnectBridge,'reversi':ReversiBridge,'quoridor':QuoridorBridge,
    'no_thanks':ThanksBridge,'goofspiel':GoofspielBridge,'hagetaka':ContestBridge,
    'auction_contest':ContestBridge,'gambling':GamblingBridge,'team_projects':TeamProjectsBridge,
    'team_combat':TeamCombatBridge,'laboratory':LaboratoryBridge,
    'encounter_exchange':EncounterBridge,'encounter_watch':EncounterBridge,
    'social_allocation':SocialProbeBridge,'social_patrol':SocialProbeBridge,
    'empire_race':EmpireBridge,'cyclic_arena':TrapBridge,'shared_commons':TrapBridge,'branching_delivery':TrapBridge,
    'tiny_resources':TinyBridge,'tiny_arena':TinyBridge,'public_graph':ProceduralBridge}


def make_bridge(profile,case):
    selected=copy.deepcopy(case)
    if selected['game']=='auction_contest':selected['game']='auction'
    if selected['game'].startswith('encounter_'):selected.setdefault('domain',selected['game'].removeprefix('encounter_'))
    if selected['game'].startswith('social_'):selected.setdefault('domain',selected['game'].removeprefix('social_'))
    return BRIDGES[case['game']](profile,selected)
