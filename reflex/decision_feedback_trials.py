"""Immediate known-rule transfer labs; no secret opponent state in predictors.

These repeated contests isolate advice adoption. They do not stand in for
long-horizon planning or establish strength in a full commercial game.
"""
import copy
import json
from dataclasses import replace
import numpy as np
from .core import FEATURES,TRAITS,Policy,compile_batch,digest
from .examples import context,action,effect
from .decision_loop import DecisionLoop,Request
from .judgment import Binding,ReadControl
from .judgment_experiment import Arena,Commons
from .loop_predictor import CategoricalReader
from .public_perception import with_public
from .planning import vector
from .contests import Public,settle


GAMES=('arena','commons','auction','exchange')
REGIMES=('stationary','switch','cycle','reactive','noisy')


def public(c):
    return json.loads(''.join(c['facts'][k] for k in sorted(c['facts']) if k.startswith('perception_')))


def names(n):return tuple([f'fixed-{i}' for i in range(n)]+['repeat','cycle','broad'])


def models(c):
    q=public(c);space=list(q['responses']);n=len(space);h=q['history']
    uniform={k:1/n for k in space}
    def favored(k):return {x:.9 if x==k else .1/(n-1) for x in space}
    result={f'fixed-{i}':favored(k) for i,k in enumerate(space)}
    result['repeat']=favored(h[-1]) if h else uniform
    result['cycle']=favored(space[(space.index(h[-1])+1)%n]) if h else uniform
    result['broad']=uniform
    return result


def conditional(c,key,response):return copy.deepcopy(public(c)['table'][key][response])


def auction_position(tick):
    # Independent publicly funded lots, not the finite-budget campaign.
    return Public('auction',((),)*4,(0,)*4,(18,)*4,4+tick%5,('science','culture','power')[tick%3],(),round=tick)


def rule_table(game,tick):
    if game=='arena':return {k:{r:Arena().result(k,r) for r in Arena.moves} for k in Arena.names}
    if game=='commons':return {k:{'crowded':Commons().result(k,('CLAIM',)*3),'open':Commons().result(k,('HELP','SECURE','HELP'))} for k in Commons.names}
    if game=='auction':
        s=auction_position(tick);result={}
        for bid in s.legal(0):
            rows={}
            for code in range(8):
                rivals=tuple(6 if code&(1<<i) else 2 for i in range(3));after,winner=settle(s,(bid,*rivals))
                gain=after.scores[0];paid=18-after.budgets[0];net=(gain-paid)/8
                rows[str(code)]=effect(net,needs={'esteem':gain/8,'safety':-paid/18},
                    values={'achievement':net,'power':(gain-(0 if winner in (None,0) else s.prize))/8,'security':-paid/18},cost=paid/48)
            result[str(bid)]=rows
        return result
    if game=='exchange':
        # The deterministic part of EncounterWorld's exchange rule: public
        # responses cooperate/claim/decline, no unrevealed luck counterfactual.
        from .encounters import EncounterWorld
        w=EncounterWorld(0,'exchange','balanced4')
        return {k:{r:w.perceived(k,r if r!='decline' else None,k in ('cooperate','claim') and r!='decline',1.)
            for r in ('cooperate','claim','decline')} for k in ('cooperate','claim','alone','rest')}
    raise ValueError('declared game required')


def observation(game,profile,seed,tick,history,score,energy):
    table=rule_table(game,tick);responses=list(next(iter(table.values())))
    acts=[]
    for key,rows in table.items():
        outcomes=[dict(copy.deepcopy(r),p=1/len(responses)) for r in rows.values()]
        legal=game!='exchange' or key=='rest' or energy>=.18
        acts.append(action(key,*outcomes,confidence=.8,legal=legal))
    needs={'physiology':1-energy,'belonging':.3,'esteem':.2,'growth':.3} if game=='exchange' else {'esteem':.3,'safety':.1} if game=='auction' else {}
    c=context('decision-feedback',acts,needs=needs,values=profile['values'],traits=dict(zip(TRAITS,profile['traits'])),mode=None)
    for k in c['needs']:
        if k not in needs:c['needs'][k]=dict(supported=False,enabled=False,deficit=None)
    # Public driver seed is NOT the agent's or hypothesis's random generator.
    c.update(scope=dict(game=game+'-response-lab',episode=str(seed),npc=profile['id']),seed=42,tick=tick)
    return with_public(c,dict(responses=responses,history=history[-4:],table=table,score=score,energy=energy))


def revealed(game,regime,seed,tick,history,own):
    space=list(next(iter(rule_table(game,tick).values())));n=len(space)
    rng=np.random.default_rng(int(digest(['actual-response-lab',seed,tick,game])[:16],16))
    i=seed%n
    if regime in ('switch','noisy'):i=(i+tick//16)%n
    elif regime=='cycle':i=(i+tick)%n
    elif regime=='reactive':
        # Opponent sees only PAST actions. No response to this sealed choice.
        i=(sum(str(own[-1]).encode())%n) if own else i
    if regime=='noisy' and rng.random()<.2:i=int(rng.integers(n))
    return space[i]


def play(game,regime,profile,seed,metric,*,turns=64,priority='finite'):
    if game not in GAMES or regime not in REGIMES or metric not in ('purpose','persona'):raise ValueError('declared experiment required')
    history=[];actions=[];score=0.;energy=.8;trace=[]
    c=observation(game,profile,seed,0,history,score,energy)
    reader=CategoricalReader(c['scope'],names(len(public(c)['responses'])),models,conditional,known_conditionals=True)
    actor=DecisionLoop(c,Policy(principle_priority=priority),ReadControl(max_nodes=128,close_gap=1.,downside=0.),predictor=reader,decision_metric=metric)
    for tick in range(turns):
        c=observation(game,profile,seed,tick,history,score,energy)
        # All conditional dimensions may change with a response. Only actual
        # selected feedback trains OutcomeMemory; the rule table is not data.
        req=Request(c,{a['id']:Binding(a['id'],str(tick%5) if game=='auction' else 'same',FEATURES) for a in c['actions']},
            {a['id']:() for a in c['actions']},maintains_advantage=False,threatened=False)
        r=actor.decide(req,False);key=r['decision']['action_id']
        response=revealed(game,regime,seed,tick,history,actions)
        row=conditional(c,key,response)
        # Diagnostic revaluation uses the predecision actor's SAME private
        # state and public facts, plus the now revealed conditional outcome.
        # Only the selected row is a physical/empirical observation.
        frozen=copy.deepcopy(r['context']);audit={}
        for a in frozen['actions']:a['outcomes']=[conditional(c,a['id'],response)]
        bb=compile_batch([frozen]);dd=actor.policy.decide(bb,False,outcome_audit=audit)
        actual_appraisal=float(dd.scores[0,bb.ids[0].index(key)])
        update=actor.observe(r['ticket'],vector(row),{'revealed_action':response})
        history.append(response);actions.append(key);score+=row['objective']
        if game=='exchange':energy=max(0.,min(1.,energy+.4) if key=='rest' else energy-.18);energy=max(0.,energy-.025)
        trace.append(dict(tick=tick,action=key,response=response,flow=row,score=score,energy=energy,
            appraisal=actual_appraisal,reading=r['reading'],comparison=update['decision_comparison'],predictive=update['reading'],
            owner=digest(actor.record())))
    # Independent rule replay is against the saved chosen physical actions.
    h=[];own=[];check=0.
    for t in trace:
        event=revealed(game,regime,seed,t['tick'],h,own);assert event==t['response']
        physical=rule_table(game,t['tick'])[t['action']][event];assert physical==t['flow']
        if game=='auction':
            code=int(event);rivals=tuple(6 if code&(1<<i) else 2 for i in range(3));s=auction_position(t['tick']);after,_=settle(s,(int(t['action']),*rivals))
            assert physical['objective']==(after.scores[0]-(18-after.budgets[0]))/8
        check+=physical['objective'];assert check==t['score'];h.append(event);own.append(t['action'])
    restored=DecisionLoop.from_record(c,json.loads(json.dumps(actor.record())),predictor=reader)
    assert restored.record()==actor.record()
    return dict(game=game,regime=regime,profile=profile['id'],seed=seed,metric=metric,turns=turns,priority=priority,
        score=score,appraisal=sum(t['appraisal'] for t in trace),actions=actions,
        values={v:sum(t['flow']['values'].get(v,0.) for t in trace) for v in c['values']},
        comparisons=sum(t['comparison'] is not None for t in trace),
        model_uses=sum(t['reading'].get('response_model')=='validated_hypotheses' for t in trace),
        revoked=sum(bool(t['comparison'] and t['comparison']['revoked']) for t in trace),
        replay_verified=True,checkpoint_verified=True,trace=trace)
