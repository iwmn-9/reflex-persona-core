"""Conditional OWN-HAND diagnostic inside actual closed-hand four-player play.

This does NOT forecast a kyoku, opponent turns, yaku, score or ron probability.
Future legal discard sets concern the owner's hypothetical draws only. Actual
roots are always the external engine's closed-protocol legal actions. The
purpose is explicitly a hand-shape proxy, never disguised as winning EV.
"""
from dataclasses import dataclass,replace,asdict
import copy
from .core import Policy,digest
from .examples import effect
from .purpose_plan import Goal,goal_forecast
from .flow_rollout import rollout
from .contingent_search import search_forecast
from .observation_plan import PlanMemory
from .deliberation import select
from .mahjong_adapter import (PublicView,candidates,make_context,action_id,action_type,
    remove_tile,hand_features,engine,reference)
from .contingent_games import rng
from .public_perception import with_public


@dataclass(frozen=True)
class HandState:
    view: PublicView
    rows: tuple
    tick: int
    depth: int=0
    finished: bool=False


class HandModel:
    target='closed-hand-shape-proxy';unit='own-hand-decisions'
    def __init__(self,profile,episode):self.profile=profile;self.episode=episode
    def context(self,s,state=None):
        return with_public(make_context(s.view,s.rows,self.profile,42,s.tick,self.episode,state),asdict(s.view))
    @staticmethod
    def terminal(s):return s.finished or s.depth>=2
    def goal(self,s):
        # The generic 14-tile shape formula can exceed one at shanten=-1.
        # A complete hypothetical shape is bounded proxy credit, not a legal win.
        progress=1. if s.finished else min(1.,max(0.,hand_features(s.view.hand,s.view.unseen())['progress']))
        return Goal(self.target,'scored' if self.terminal(s) else 'running',progress)
    def advance(self,s,key,seed):
        row=next(r for r in s.rows if r['id']==key);v=s.view;before=self.goal(s).value
        if row['type'] in ('Ron','Tsumo'):
            after=replace(s,depth=s.depth+1,finished=True)
            return after,effect(1-before,values={'achievement':1-before})
        draw=None
        if row['type']=='Discard':
            hand=remove_tile(v.hand,row['tile']);rivers=list(v.discards);rivers[v.player]=rivers[v.player]+(row['tile'],)
            visible=set(hand+tuple(v.dora)+tuple(t for river in rivers for t in river))
            unseen=tuple(t for t in range(136) if t not in visible)
            draw=int(rng(seed,s.tick,'mahjong-own-draw').choice(unseen));v=replace(v,hand=tuple(sorted(hand+(draw,))),discards=tuple(rivers))
        elif row['type']=='Riichi':
            riichi=list(v.riichi);riichi[v.player]=True;v=replace(v,riichi=tuple(riichi))
        elif row['type']=='Pass' and len(v.hand)==13:
            draw=int(rng(seed,s.tick,'mahjong-own-draw').choice(v.unseen()))
            v=replace(v,hand=tuple(sorted(v.hand+(draw,))))
        rows=[]
        for tile in sorted(set(v.hand)):
            hand=remove_tile(v.hand,tile);f=hand_features(hand,v.unseen());exposure=v.exposure(tile)
            if v.riichi[v.player] and draw is not None and tile!=draw:continue
            if row['type']=='Riichi' and f['shanten']!=0:continue
            rows.append(dict(id=f'Discard:{tile}:',type='Discard',tile=tile,supported=True,
                features=f,exposure=exposure,outcomes=[effect(f['progress'],values={'achievement':f['progress'],'security':-exposure},cost=.02*exposure)]))
        after=HandState(v,tuple(rows),s.tick+1,s.depth+1)
        gain=self.goal(after).value-before
        return after,effect(gain,values={'achievement':gain,'security':-row['exposure']},cost=.02*row['exposure'])


def play(profile,seed,method='conditional'):
    from .mahjong_experiment import audit
    env=engine().RiichiEnv(game_mode='4p-red-single',seed=seed);observations=env.reset();seat=seed%4
    state=None;memory=None;trace=[];owner_tick=0;initial=env.scores()
    for tick in range(600):
        if env.done():break
        actions={};rows=[]
        for player,obs in sorted(observations.items()):
            if player!=seat:chosen=reference(obs,seed,tick,'efficiency');rows.append(dict(player=player,action=action_id(chosen)));actions[player]=chosen;continue
            view,candidates_=candidates(obs);world=HandState(view,tuple(candidates_),owner_tick)
            model=HandModel(profile,'conditional-mahjong');c=model.context(world,state)
            if memory is None:memory=PlanMemory(c,model.target,model.unit)
            saved=memory.record()
            if method=='conditional':
                retained,prefix=memory.offer(c)
                f=search_forecast(c,world,observe=model.context,advance=model.advance,terminal=model.terminal,
                    assess=lambda w:model.goal(w).record(),horizon=2,seeds=(401,402),width=1,revisions=1,
                    target=model.target,unit=model.unit,retained=retained,prefix=prefix,patch_budget=4)
            elif method=='goal-reflex':
                paths,bank=rollout(c,world,observe=model.context,advance=model.advance,terminal=model.terminal,
                    assess=lambda w:model.goal(w).record(),horizon=2,seeds=(401,402),public_controller=lambda view:None)
                f=goal_forecast(c,paths,bank,horizon=2,unit=model.unit,target=model.target)
            else:raise ValueError('known own-hand comparison required')
            assert memory.record()==saved
            decisions,vote=select([c],f);d=decisions[0];key=d['action_id'];state=d['next_state']
            if method=='conditional':memory.remember(c,f,vote.get('selected_plan'),key)
            chosen=next(a for a in obs.legal_actions() if action_id(a)==key)
            rows.append(dict(player=player,public=asdict(view),action=key,decision=d,
                goal=vote.get('selected_purpose'),conditional=None if method!='conditional' else dict(evaluated=f.audit['contingent_search']['evaluated'],plan=memory.record()),
                semantics='two own-hand decisions; uniform unseen own draws; no kyoku or score forecast'))
            actions[player]=chosen;owner_tick+=1
        start=len(env.mjai_log);observations=env.step(actions);audit(env)
        trace.append(dict(tick=tick,decisions=rows,events=env.mjai_log[start:]))
    else:raise AssertionError('real closed-hand kyoku exceeded finite driver cap')
    scores=env.scores();best=max(scores);credit=(1/scores.count(best) if scores[seat]==best else 0.)
    return dict(case={'game':'mahjong_closed'},profile=profile['id'],seed=seed,method=method,
        steps=len(trace),owner_decisions=owner_tick,finished=env.done(),scores=scores,
        point_delta=scores[seat]-initial[seat],win_share=credit,trace=trace,
        limitation='actual four-player closed-hand kyoku; conditional forecast is ONLY own-hand proxy, not four-player continuation')
