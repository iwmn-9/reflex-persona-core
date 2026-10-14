"""Transfer to existing public-bid Goofspiel, with an independent two-round oracle.

All remaining public bids are enumerated under declared uniform opponents.
These are finite model evaluations, not sampled matches against learned rivals.
"""
import argparse
import copy
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.adaptive_dynamics import PublicEvent
from reflex.closed_loop import ClosedLoop,TreeBudget,Transition
from reflex.core import Policy,character_key,compile_batch,digest
from reflex.decision_loop import DecisionLoop
from reflex.goofspiel import Position,consequence,make_context,referee,exact_two_rounds
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from reflex.model_observer import ModelFeedback
from reflex.opportunity_graph import OpportunityGraph,persona_choices
from reflex.purpose_plan import Goal
from tools.run_frontier_value import pool_run


def evaluate(job):
    players,order,seed=job;rng=random.Random(seed);start=Position.start(players=players,cards=13,order=order)
    while len(start.hands[0])>3:
        before=start;start=start.play(tuple(rng.choice(h) for h in start.hands));referee(before,tuple(next(c for c in h if c not in after) for h,after in zip(before.hands,start.hands)),start)
    denominator=sum(start.prizes[start.round:]);out=[];counts=[]
    for goal in ('score','win_share'):
        target='goofspiel-'+goal;model=digest(['opportunity-cards-v1',asdict(start),goal])
        def assess(s):
            value=(s.scores[0]-start.scores[0])/denominator if goal=='score' else 2*s.share(0)[0]-1
            return Goal(target,'scored' if s.terminal else 'running',float(value)).record()
        def events(s,key):
            rows=[];own=int(key.split(':')[1])
            for others in product(*s.hands[1:]):
                bids=(own,*others);after=s.play(bids);referee(s,bids,after)
                rows.append(PublicEvent(str(others),after,consequence(s,after.scores,0,own,goal,terminal=after.terminal)))
            return rows
        def keys(s):return tuple(f'BID:{x}' for x in s.hands[0])
        graph=OpportunityGraph(actions=keys,events=events,terminal=lambda s:s.terminal,assess=assess,record=asdict,model_id=model)
        assert graph.compile(start);counts.append(dict(goal=goal,**graph.last))
        law=lambda s,k,symbols:dict.fromkeys(symbols,1/len(symbols))
        ideal=graph.solve(law,law_id='declared-independent-uniform');assert ideal is not None
        # The legacy helper does not use OpportunityGraph or persona_choices.
        oracle={}
        for i,n in enumerate(graph.nodes):
            s=n['state']
            if len(s.hands[0])!=2:continue
            targets=exact_two_rounds(s,0)
            expected=max((t['mean_score']-start.scores[0])/denominator if goal=='score' else 2*t['win_share']-1 for t in targets.values())
            predicted=sum(p*v for p,v in ideal.distributions[i]);assert abs(predicted-expected)<1e-12
            oracle[digest(asdict(s))]=expected
        root_values={k:sum(oracle[digest(asdict(e.public))] for e in events(start,k))/len(events(start,k)) for k in keys(start)}
        assert abs(sum(p*v for p,v in ideal.evaluate(start).outcomes)-max(root_values.values()))<1e-12
        for profile in profiles():
            regime=None
            def observe(s,state=None):return make_context(s,0,profile,goal,91,s.round,'opportunity-cards:'+model,memory=regime if state is None else state)
            c=observe(start);policy=Policy(principle_priority='personal',commitments={character_key(c):.5})
            # These needs are constant and switching has no fee. The owned mode
            # persists after the first choice; do not reroll it in future nodes.
            b=compile_batch([c]);regime=policy.decide(b,False).records(b)[0]['next_state']
            regime.update(intent_action=None,age=0)
            def immediate(states,alternatives):
                b=compile_batch([observe(s) for s in states]);records=policy.decide(b,False).records(b)
                return tuple(r['action_id'] for r in records)
            def planned(states,alternatives):return persona_choices([observe(s) for s in states],alternatives,policy)
            solutions={'reflex':graph.solve(law,law_id='declared-independent-uniform',choose=immediate,chooser_id=digest(['reflex',profile])),
                       'planned_persona':graph.solve(law,law_id='declared-independent-uniform',choose=planned,chooser_id=digest(['persona',profile])),
                       'terminal_goal':ideal}
            for name,solution in solutions.items():assert solution is not None
            actor=DecisionLoop(c,policy=policy);before=actor.record();request=request_for(c,'goofspiel')
            def transitions(s,key):
                rows=events(s,key)
                return [Transition(1/len(rows),r.public,r.flow) for r in rows]
            ctrl=ClosedLoop(observe=observe,transitions=transitions,terminal=lambda s:s.terminal,assess=assess,record=asdict,
                request_for=lambda c:request_for(c,'goofspiel'),feedback=lambda *args:ModelFeedback(None),unit='public-round',target=target,model_id=model,
                budget=TreeBudget(horizon=1,nodes=100000,outcomes=4096),frontier=solutions['planned_persona'])
            replica=copy.deepcopy(actor);d=DecisionLoop.decide_batch([(replica,request)],False,ctrl.planner(replica,request,start))[0]
            assert d['decision']['action_id']==solutions['planned_persona'].action(start) and before==actor.record()
            out.append(dict(goal=goal,profile=profile['id'],actual_root_matches=True,
                arms={name:dict(action=solution.action(start),expected_goal=sum(p*v for p,v in solution.evaluate(start).outcomes),distribution=solution.evaluate(start).outcomes)
                      for name,solution in solutions.items()}))
    return dict(players=players,order=order,seed=seed,public_start=asdict(start),graphs=counts,cases=out,independent_two_round_oracle=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--frozen',action='store_true')
    a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--workers',str(a.workers)])
    jobs=[(players,order,seed) for players in (3,4,5) for order in ('ascending','descending') for seed in (82600,82601)]
    reg=dict(jobs=jobs,source=digest(json.loads((root/'source_snapshot.json').read_text(encoding='utf-8'))),
        selection='all 3/4/5-player x two prize orders x two seeded legal openings, last three of thirteen rounds, four personas and two objectives',
        purpose='transfer and finite exact evaluation; no learned opponent or human-level claim',
        arms='reflex uses existing immediate Policy every round; planned_persona uses current flow plus terminal purpose every round; terminal_goal maximizes purpose every round')
    (root/'preregister.json').write_text(json.dumps(reg,indent=2)+'\n',encoding='utf-8')
    rows,errors,elapsed=pool_run(jobs,evaluate,root,a.workers)
    result=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,cases=sum(len(r['cases']) for r in rows),
        all_existing_oracle_checks=all(r['independent_two_round_oracle'] for r in rows))
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
    if errors:raise RuntimeError('failed transfer cases retained')


if __name__=='__main__':main()
