"""Frozen, matched loss attribution across public card and four rule families.

No results select openings or parameters. Exact under declared public laws;
this is not a sampled win-rate study or an unknown-game competence claim.
"""
import argparse
import copy
from dataclasses import asdict
from itertools import product
import json
import math
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

from reflex.adaptive_dynamics import PublicEvent, AdaptiveState
from reflex.closed_loop import ClosedLoop, TreeBudget, Transition, reveal
from reflex.core import Policy, character_key, compile_batch, digest
from reflex.decision_loop import DecisionLoop
from reflex.decision_audit import policy_loss_audit
from reflex.goofspiel import Position, consequence, make_context, referee, exact_two_rounds
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from reflex.model_observer import ModelFeedback
from reflex.opportunity_graph import OpportunityGraph, persona_choices
from reflex.preference_basis import PreferenceBasis
from reflex.purpose_plan import Goal
from tools.open_model_games import OpenGame
from tools.run_frontier_value import pool_run


def uniform(s,k,symbols):return dict.fromkeys(symbols,1/len(symbols))


def check_execution(graph,initial,observe,policy,solution,law,basis):
    """Real DecisionLoop root, then three legal revealed continuations.

    Every step reads the same compiled continuation; no actor model truth.
    Public probabilities, feedback ownership and preserved mode are checked.
    """
    c=observe(initial);owner=DecisionLoop(c,policy=policy);original=owner.record()
    def transitions(s,key):
        rows=graph.events(s,key);p=law(s,key,tuple(r.symbol for r in rows))
        return [Transition(p[r.symbol],r.public,r.flow) for r in rows if p[r.symbol]>0]
    ctrl=ClosedLoop(observe=observe,transitions=transitions,terminal=graph.terminal,assess=graph.assess,record=graph.record,
        request_for=lambda c:request_for(c,'preference-study'),feedback=lambda *args:ModelFeedback(None),unit='public-decision',
        target=graph.nodes[graph.root]['goal'].target,model_id=graph.model_id,budget=TreeBudget(horizon=1,nodes=100000,outcomes=4096),
        frontier=solution,preference_basis=basis)
    matches=[]
    for path_index in range(3):
        actor=copy.deepcopy(owner);s=copy.deepcopy(initial);path=[]
        while not graph.terminal(s):
            req=request_for(observe(s,actor.state),'preference-study')
            d=DecisionLoop.decide_batch([(actor,req)],False,ctrl.planner(actor,req,s))[0]
            key=d['decision']['action_id']
            if key!=solution.action(s):raise AssertionError('table differs from actual owned redecision')
            rows=transitions(s,key);row=rows[path_index%len(rows)];path.append(key)
            reveal(actor,d['ticket'],ModelFeedback(None));s=row.state
        matches.append(path)
    assert original==owner.record()
    return matches


def compare(graph,initial,observe_raw,profile,law,reference,law_id,instrumental):
    c=observe_raw(initial);policy=Policy(principle_priority='personal',commitments={character_key(c):.5})
    # Fixed needs, no switch fees/familiarity effects: mode is the only retained
    # behavioral state here. Resolve it ONCE, as the actual owner does.
    batch=compile_batch([c]);regime=policy.decide(batch,False).records(batch)[0]['next_state']
    regime.update(intent_action=None,age=0)
    def observe(s,state=None):return observe_raw(s,regime if state is None else state)
    target=graph.nodes[graph.root]['goal'].target
    bases=dict(legacy=None,deduplicated=PreferenceBasis(target,instrumental),
        endpoint_only=PreferenceBasis(target,purpose=(('values.achievement',1.),)),
        repaired=PreferenceBasis(target,instrumental,(('values.achievement',1.),)))
    solutions={};root_audits={}
    for name,basis in bases.items():
        audits={}
        def choose(states,alternatives):
            records=[]
            keys=persona_choices([observe(s) for s in states],alternatives,policy,
                preference_basis=basis,target=target,audit=records)
            for s,r in zip(states,records):
                if graph.record(s)==graph.record(initial):audits['root']=r
            return keys
        solutions[name]=graph.solve(law,law_id=law_id,choose=choose,chooser_id=digest([profile,basis.identity if basis else 'legacy',regime]))
        if solutions[name] is None:raise RuntimeError(graph.last)
        root_audits[name]=audits['root']
    informed_root={}
    def informed_choose(states,alts):
        records=[]
        keys=persona_choices([observe(s) for s in states],alts,policy,preference_basis=bases['repaired'],target=target,audit=records)
        for s,r in zip(states,records):
            if graph.record(s)==graph.record(initial):informed_root.update(r)
        return keys
    informed=graph.solve(reference,law_id='declared-reference',choose=informed_choose,chooser_id=digest(['informed',profile,regime]))
    informed_legacy=graph.solve(reference,law_id='declared-reference',choose=lambda states,alts:persona_choices([observe(s) for s in states],alts,policy),
        chooser_id=digest(['informed-legacy',profile,regime]))
    assert informed is not None
    audit=policy_loss_audit(graph,initial,reference,reference_id='declared-reference',informed=informed,repaired=solutions['repaired'],legacy=solutions['legacy'],informed_legacy=informed_legacy)
    arms={}
    for name,sol in solutions.items():
        fixed=graph.solve(reference,law_id='declared-reference',choose=lambda states,alts,sol=sol:tuple(sol.action(s) for s in states),chooser_id=sol.identity)
        arms[name]=dict(action=sol.action(initial),actor_value=math.fsum(p*v for p,v in sol.evaluate(initial).outcomes),
            reference_value=math.fsum(p*v for p,v in fixed.evaluate(initial).outcomes),root=root_audits[name])
    paths=check_execution(graph,initial,observe,policy,solutions['repaired'],law,bases['repaired'])
    # Personality/commitment coefficients are an invariant, never a fit target.
    assert policy.signature()==Policy(principle_priority='personal',commitments={character_key(c):.5}).signature()
    return dict(profile=profile['id'],arms=arms,attribution=audit,execution_paths=paths,
        regime=regime,character=character_key(c),informed_root=informed_root,basis={k:None if v is None else asdict(v) for k,v in bases.items()})


def cards(job):
    _,players,order,seed=job;rng=random.Random(seed);start=Position.start(players=players,cards=13,order=order)
    while len(start.hands[0])>3:
        bids=tuple(rng.choice(h) for h in start.hands);after=start.play(bids);referee(start,bids,after);start=after
    out=[];graphs=[]
    for goal in ('score','win_share'):
        target='goofspiel-'+goal;denominator=sum(start.prizes[start.round:]);model=digest(['opportunity-cards-v1',asdict(start),goal])
        def assess(s):
            v=(s.scores[0]-start.scores[0])/denominator if goal=='score' else 2*s.share(0)[0]-1
            return Goal(target,'scored' if s.terminal else 'running',v).record()
        def events(s,key):
            own=int(key.split(':')[1]);rows=[]
            for others in product(*s.hands[1:]):
                bids=(own,*others);after=s.play(bids);referee(s,bids,after)
                rows.append(PublicEvent(str(others),after,consequence(s,after.scores,0,own,goal,terminal=after.terminal)))
            return rows
        graph=OpportunityGraph(actions=lambda s:tuple(f'BID:{b}' for b in s.hands[0]),events=events,
            terminal=lambda s:s.terminal,assess=assess,record=asdict,model_id=model)
        assert graph.compile(start);graphs.append(copy.deepcopy(graph.last))
        optimal=graph.solve(uniform,law_id='uniform')
        # Independent pre-existing two-round oracle, not this graph algorithm.
        for i,n in enumerate(graph.nodes):
            s=n['state']
            if len(s.hands[0])!=2:continue
            targets=exact_two_rounds(s,0)
            expected=max((t['mean_score']-start.scores[0])/denominator if goal=='score' else 2*t['win_share']-1 for t in targets.values())
            assert abs(math.fsum(p*v for p,v in optimal.distributions[i])-expected)<1e-12
        for profile in profiles():
            def observe(s,state=None):return make_context(s,0,profile,goal,91,s.round,'opportunity-cards:'+model,memory=state)
            # Card safety/security were authored solely as lost future bidding
            # ability. Immediate prestige, social gain and bold/frugal style stay.
            row=compare(graph,start,observe,profile,uniform,uniform,'uniform',('needs.safety','values.security'))
            out.append(dict(goal=goal,**row))
    return dict(job=job,game='cards',public_start=asdict(start),graphs=graphs,cases=out,oracle=True)


def open_game(job):
    _,name,variant,seed=job;rng=random.Random(seed);out=[]
    g=OpenGame(name,profiles()[0],variant=variant);s=g.start(0)
    def state(w):return AdaptiveState(w,s.memory,g.kernel.state_identity,w['tick'])
    # Three legal public turns; never discard an opening for poor performance.
    for _ in range(3):
        if g.terminal(s):break
        key=rng.choice(g.keys(s));row=rng.choice(g.events(s.public,key));s=state(row.public)
    # Dead early states are reported; the registered sampled opening is not replaced.
    if g.terminal(s):return dict(job=job,game=name,public_start=s.public,cases=[],terminal_opening=True)
    graph=OpportunityGraph(actions=lambda w:g.keys(state(w)),events=g.events,
        terminal=lambda w:g.terminal(state(w)),assess=lambda w:g.goal(state(w)).record(),
        record=lambda w:w,model_id=digest(['preference-open-v1',g.model_id,s.public]))
    assert graph.compile(s.public)
    def law(s,key,symbols):
        # Public fixed estimate. Same law for every persona and treatment.
        return {z:.55*(z==symbols[-1])+.45/len(symbols) for z in symbols}
    def reference(s,key,symbols):
        # Separate reference controller, unavailable to acting policies. Nonlinear
        # public conditional response also tests changing future own actions.
        pick=symbols[-1] if (g.scene(s)=='1') != (s['previous']=='0') else symbols[0]
        return {z:.8*(z==pick)+.2/len(symbols) for z in symbols}
    for profile in profiles():
        gp=OpenGame(name,profile,variant=variant)
        def observe(w,memory=None):return gp.observe(state(w),memory)
        # Actual injuries/lost cash remain intrinsic safety effects. Only the
        # short-run achievement estimate is replaced; no game-specific weights.
        out.append(dict(goal='declared-purpose',**compare(graph,s.public,observe,profile,law,reference,'declared-actor',())))
    return dict(job=job,game=name,public_start=s.public,graphs=[graph.last],cases=out)


def evaluate(job):return cards(job) if job[0]=='cards' else open_game(job)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cohort',choices=('pilot','development','holdout','diagnostic'),required=True);p.add_argument('--frozen',action='store_true')
    a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--workers',str(a.workers),'--cohort',a.cohort])
    seeds=(82600,82601) if a.cohort=='development' else (82610,82611,82612,82613)
    jobs=[('cards',n,o,s) for n in (3,4,5) for o in ('ascending','descending') for s in seeds]
    jobs += [('open',g,v,s) for g in ('auction','resources','combat','gambling') for v in (0,1) for s in seeds]
    if a.cohort=='pilot':jobs=[('cards',4,'descending',82601),('open','resources',0,82600)]
    if a.cohort=='diagnostic':jobs=[('cards',4,'descending',82601),('cards',3,'ascending',82600),('open','combat',0,82600)]
    registration=dict(jobs=jobs,cohort=a.cohort,source=digest(json.loads((root/'source_snapshot.json').read_text(encoding='utf-8'))),
        fixed='four original personas, commitment .5, corridor .15; same public graph, law and personality across proxy treatments',
        controls='legacy; remove declared instrumental estimates; replace achievement by terminal purpose; both; informed persona and purpose-optimal reference',
        semantics='card safety/security are future-ability proxies; open games physical safety intrinsic; esteem/power/benevolence/style retained',
        attribution='optimal - informed + informed - repaired + repaired - legacy; all policies evaluated under same reference law; signed, order-dependent',
        acceptance='correct ownership, exact decomposition/oracle, intrinsic control preserved, no game-specific core branches; report all adverse cases; no promise all scores improve')
    (root/'preregister.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    rows,errors,elapsed=pool_run(jobs,evaluate,root,a.workers)
    summary=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,cases=sum(len(r['cases']) for r in rows))
    (root/'evaluation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary),flush=True)
    if errors:raise RuntimeError('failed cases retained')


if __name__=='__main__':main()
