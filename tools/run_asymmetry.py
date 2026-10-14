"""Frozen matched asymmetric campaigns. Truth is used only by the referee.

The two actual actors use only public results. Reference optimal and same-persona
policies are solved under the referee law for diagnosis, never fed to the actor.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy, character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.closed_loop import reveal
from reflex.integrated_trials import request_for
from reflex.model_observer import ModelFeedback
from reflex.laboratory import profiles
from reflex.preference_basis import PreferenceBasis
from tools.asymmetric_games import AsymmetricGame, board, GAMES, CASES
from tools.opportunity_models import OpenOpportunities, canonical
from tools.run_adaptive_opportunities import reference_value
from tools.run_frontier_value import pool_run


GRAPHS={}
REFERENCES={}
ARMS=('live','reflex')


def world(g,w):
    """Unknown stationary conditional behaviour; no case/profile/seed lookup."""
    symbols=g.symbols(w)
    favored=(int(g.scene(w))+(2 if g.variant else 0))%len(symbols)
    noise=.35 if g.variant else .2
    return {s:noise/len(symbols)+(1-noise)*(i==favored) for i,s in enumerate(symbols)}


def sample(g,w,key,seed,episode,phase):
    rows=g.events(w,key);p=world(g,w)
    u=random.Random(int(digest(['asymmetry',seed,episode,w['tick'],phase])[:16],16)).random()
    mass=0.
    for row in rows:
        mass+=p[row.symbol]
        if u<mass:return row
    return rows[-1]


def entry(game,case,profile,seed,variant,episode):
    g=AsymmetricGame(game,profile,board(game,case,variant),variant=variant)
    initial=g.start(episode).public
    prefix=[]
    if not initial['joined']:
        for _ in range(2):
            event=sample(g,initial,'absent',seed,episode,'prefix')
            prefix.append(dict(public=initial,action='absent',event=event.symbol,after=event.public))
            initial=event.public
        initial=copy.deepcopy(initial);initial['joined']=True
    g.initial=initial
    return g,prefix


def mean_distribution(solution,state):
    return math.fsum(p*v for p,v in solution.distributions[solution.index(state)])


def references(adapter,s,owner,basis):
    g=adapter.g
    key=digest([adapter.graph.identity,owner.character,owner.policy.signature(),s.public['episode']%2])
    if key not in REFERENCES:
        choose,root,regime=adapter.continuation(s,owner,preference_basis=basis)
        law=lambda w,a,syms:world(g,w)
        persona=adapter.graph.solve(law,law_id='referee',choose=choose,chooser_id=key)
        optimal=adapter.graph.solve(law,law_id='referee')
        if persona is None or optimal is None:raise AssertionError('incomplete reference')
        REFERENCES[key]=(persona,optimal)
    return REFERENCES[key]


def run_arm(game,case,profile,seed,variant,arm,episodes):
    memory=None;trace=[];ends=[];starts=[];prefixes=[];entries=[]
    for ep in range(episodes):
        g,prefix=entry(game,case,profile,seed,variant,ep)
        if memory is None:memory=g.kernel.learner.start()
        observations=memory.observations
        if case=='late_observer':
            # These events were publicly visible before participation. No own
            # actions/rewards/DecisionLoop memories are fabricated for them.
            for item in prefix:
                w=item['public'];key=item['action'];syms=g.symbols(w)
                memory=g.kernel.learner.observe(memory,g.features(w,key),syms,g.models(w,key),item['event'])
        s=g.start(ep,memory)
        if s.tick!=s.public['tick']:raise AssertionError('global deadline reset at entry')
        entries.append(dict(public=s.public,observations_before=observations,
            observations_at_entry=memory.observations,memory=digest(g.kernel.learner.record(memory))))
        prefixes.append(prefix)
        c=g.observe(s)
        owner=DecisionLoop(c,policy=Policy(principle_priority='personal',commitments={character_key(c):.5}))
        original=digest([owner.personality,owner.values])
        basis=PreferenceBasis(g.target,purpose=(('values.achievement',1.),))
        cache=digest([g.model_id,canonical(game,s.public)])
        adapter=OpenOpportunities(g,ep,GRAPHS.get(cache));GRAPHS[cache]=adapter.graph
        # Cached graphs contain rules only. All live solves bind fresh memory.
        persona,optimal=references(adapter,s,owner,basis)
        first=True
        while not g.terminal(s):
            before=g.kernel.record(s);owned=owner.record()
            cc=g.observe(s,owner.state);req=request_for(cc,'open-model')
            if arm=='live':
                solved=adapter.solve(s,owner,preference_basis=basis,bound=True)
                ctrl=g.controller(opportunity=solved,preference_basis=basis);cap={}
                if g.kernel.record(s)!=before or owner.record()!=owned:raise AssertionError('imagined evidence leaked')
                d=DecisionLoop.decide_batch([(owner,req)],False,ctrl.planner(owner,req,s,captured=cap))[0]
                if d['decision']['action_id']!=solved.action(s) or not cap.get('audit',{}).get('complete'):
                    raise AssertionError('projected/actual persona mismatch')
                expected=reference_value(adapter.graph,solved.solution,s.public,lambda w:world(g,w))
                predicted=mean_distribution(solved.solution,s.public)
            else:
                d=DecisionLoop.decide_batch([(owner,req)],False)[0]
                expected=predicted=None
            key=d['decision']['action_id']
            if key not in g.keys(s):raise AssertionError('used unavailable capability or resource')
            node=adapter.graph.nodes[optimal.index(s.public)]
            p=world(g,s.public)
            q={a:math.fsum(p[z]*math.fsum(prob*v for prob,v in optimal.distributions[j]) for z,j,_ in es)
               for a,es in node['branches'].items()}
            best=max(q.values());known_persona=mean_distribution(persona,s.public)
            if abs(best-mean_distribution(optimal,s.public))>1e-10:raise AssertionError('reference Bellman mismatch')
            reveal(owner,d['ticket'],ModelFeedback(None))
            if digest([owner.personality,owner.values])!=original:raise AssertionError('persona changed')
            _,_,_,_,prediction=g.kernel.forecast(s,key)
            event=sample(g,s.public,key,seed,ep,'active')
            after=g.kernel.reveal(s,key,event.public,event.flow)
            if after.memory.observations!=s.memory.observations+1:raise AssertionError('incorrect factual update count')
            trace.append(dict(episode=ep,tick=s.tick,public=s.public,action=key,event=event.symbol,
                prediction=prediction,actual=p,best_possible=best,reference_actions=q,
                root_opportunity_loss=best-q[key],same_persona_reference=known_persona,
                continuation_value=expected,predicted_value=predicted,
                personality_gap=best-known_persona,
                prediction_gap=None if expected is None else known_persona-expected,
                reference_persona_action=persona.action(s.public),
                before_memory=digest(g.kernel.learner.record(s.memory)),
                after_memory=digest(g.kernel.learner.record(after.memory))))
            if first:starts.append(key);first=False
            s=after
        memory=s.memory
        ends.append(dict(public=s.public,goal=g.goal(s).value,quantities=g.quantities(s.public)))
    return dict(arm=arm,entries=entries,prefixes=prefixes,starts=starts,ends=ends,trace=trace,
                final_memory=g.kernel.learner.record(memory))


def evaluate(job):
    game,case,profile,seed,variant,episodes=job;start=time.perf_counter()
    arms=[run_arm(game,case,profile,seed,variant,a,episodes) for a in ARMS]
    if arms[0]['prefixes']!=arms[1]['prefixes']:raise AssertionError('unmatched world before entry')
    if [e['public'] for e in arms[0]['entries']]!=[e['public'] for e in arms[1]['entries']]:
        raise AssertionError('unmatched starting resources')
    # Keep memory bounded across disjoint experiments. These caches contain no
    # private trajectories and are only reused within a matched campaign.
    GRAPHS.clear();REFERENCES.clear()
    return dict(game=game,case=case,profile=profile['id'],seed=seed,variant=variant,
                episodes=episodes,arms=arms,elapsed=time.perf_counter()-start)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=('pilot','development','holdout'),default='pilot')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--frozen',action='store_true')
    a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--stage',a.stage,'--workers',str(a.workers)])
    ps=profiles();cases=CASES;seeds=(84100,);variant=0;episodes=6
    if a.stage=='pilot':ps=ps[:1];cases=('poor','specialist','late');seeds=(84000,);episodes=2
    if a.stage=='holdout':seeds=(84300,84301);variant=1
    jobs=[(g,c,p,seed,variant,episodes) for g in GAMES for c in cases for p in ps for seed in seeds]
    registration=dict(version='asymmetry-v2',stage=a.stage,jobs=jobs,arms=ARMS,
        fixed='unchanged personalities, commitment .5, purpose corridor .15, EventLearner 8/32/128; no case names in actor state',
        scope='four finite public diagnostics; global deadline 4, late entry after 2 actual world steps; 6 successive contests',
        comparisons='full terminal persona vs immediate utility; same start and paired random variates; each arm learns only its own public outcomes',
        references='same-persona and purpose-optimal policies under referee probabilities; NEVER supplied to live actor; short finite graph ceiling, not human-level reference',
        metrics='terminal win shares; objective opportunity losses; conditional personality/prediction gaps; illegal actions; ability/resource/entry paired choices; all adverse cases retained',
        limits='fixed needs, no diplomacy or evolving personalities, known rules/capabilities; repeated profiles/seeds not independent game genres; learning omitted inside each solve',
        source=digest(json.loads((root/'source_snapshot.json').read_text(encoding='utf-8'))))
    (root/'preregister.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    rows,errors,elapsed=pool_run(jobs,evaluate,root,a.workers)
    result=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,
        games=sum(len(a['ends']) for r in rows for a in r['arms']),
        decisions=sum(len(a['trace']) for r in rows for a in r['arms']))
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)
    if errors:raise RuntimeError('registered failures retained')


if __name__=='__main__':main()
