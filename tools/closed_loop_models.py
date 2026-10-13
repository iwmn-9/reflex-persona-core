"""Evaluation adapters. Exact probabilities belong to these rules, not the core."""
import copy
from dataclasses import replace
from reflex.closed_loop import ClosedLoop,TreeBudget,Transition,SampledKernel
from reflex.contingent_games import Frame
from reflex.model_observer import ModelFeedback
from reflex.integrated_trials import request_for


def graph_kernel(bridge):
    def transitions(world,key):
        rows=[];rules=bridge.rules
        effects=next(a['outcomes'] for a in rules.observe(world.position)['actions'] if a['id']==key)
        for node,row in zip(rules.edges[*world.position,key],effects):
            flow=copy.deepcopy(row);p=flow['p'];flow['p']=1.
            rows.append(Transition(p,Frame((world.position[0]+1,node),world.turns+1),flow))
        return rows
    return transitions


def gambling_kernel(bridge):
    """Enumerate public current odds, without assuming future regime changes."""
    from reflex.gambling import offers,legal,terminal
    from reflex.examples import effect
    def transitions(world,key):
        table,regime=world.position
        if terminal(table) or key not in legal(table,regime):raise ValueError('legal live bet required')
        if key=='pass':deltas=((1.,0),)
        else:
            stake,profit,p=offers(table,regime)[key];deltas=((p,profit),(1-p,-stake))
        rows=[];safety=lambda cash:max(0.,(6-cash)/6)
        for probability,delta in deltas:
            after=replace(table,cash=table.cash+delta,tick=table.tick+1)
            growth=max(-1.,min(1.,delta/6));safe=safety(table.cash)-safety(after.cash)
            flow=effect(delta/12,needs={'growth':growth,'safety':safe},
                values={'achievement':growth,'power':growth,'security':safe})
            rows.append(Transition(probability,Frame((after,regime),world.turns+1),flow))
        return rows
    return transitions


def controller(bridge,*,horizon=4,exact=False,seeds=(401,402),future_choices='planned',nodes=4096,outcomes=256,max_regret=.15,learn=False,frontier=None):
    from reflex.integrated_trials import TRAPS
    from reflex.planning import vector
    kernels={'public_graph':graph_kernel,'gambling':gambling_kernel}
    if exact and bridge.case['game'] not in kernels:raise ValueError('declared public enumeration unavailable for this game')
    if learn and bridge.case['game'] not in TRAPS:raise ValueError('explicit immediate learning rules required')
    return ClosedLoop(observe=bridge.observe,transitions=kernels[bridge.case['game']](bridge) if exact else SampledKernel(bridge.modeled_advance,seeds),
        terminal=bridge.terminal,assess=lambda w:bridge.goal(w).record(),record=bridge.record,
        request_for=lambda c:request_for(c,bridge.case['game'],learn=learn),
        feedback=lambda before,after,key,c,row:ModelFeedback(None,observed=vector(row) if learn and key!='CHECK' else None),
        unit=bridge.unit,target=bridge.target,model_id=('public-graph-enumeration-v1' if bridge.case['game']=='public_graph' else 'public-gambling-enumeration-v1') if exact else 'public-samples-'+str(seeds),
        budget=TreeBudget(horizon=horizon,nodes=nodes,outcomes=outcomes),max_regret=max_regret,future_choices=future_choices,frontier=frontier)
