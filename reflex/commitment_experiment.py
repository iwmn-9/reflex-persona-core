"""Game-owned diagnostic: keep a pledge, revise a stochastic means.

This authored rule set is not a validated model of psychology. The driver knows
the regime; the actor sees only its past selected outcomes, never future types.
"""
import copy
from .core import Policy,character_key as owner_key,digest
from .examples import context,action,effect
from .opponent_beliefs import HypothesisTracker
from .contingent_games import rng


def pledge_episode(seed,rigidity,inertia,*,changed=True,turns=40):
    belief=HypothesisTracker(('old','new'),inertia=inertia,
        prior={'old':.95,'new':.05},assumption_trust=.9)
    health=1.;state=None;trace=[];policy=None;fixed=None
    for tick in range(turns):
        models={'old':{'success':.9,'failure':.1},'new':{'success':.1,'failure':.9}}
        estimate=belief.snapshot().predict(models)['success']
        choices=[action('leave',effect(.8))]
        for name,prob in (('old-way',estimate),('other-way',.65)):
            success=effect(-1. if health<=.015 else .1,values={'benevolence':.5},cost=.05,p=prob)
            failure=effect(-1. if health<=.07 else -.3,values={'benevolence':0.},cost=.1,p=1-prob)
            choices.append(action(name,success,failure))
        c=context('pledge-and-method',choices,values={'benevolence':1.},
            needs={'physiology':0.,'safety':1-health,'belonging':.1,'esteem':0.,'growth':0.},mode='principle')
        # The actor's counter stream and event IDs never disclose world RNG.
        c['seed']=42;c['tick']=tick;c['facts'].update(health=f'{health:.6f}',
            belief='subjective prediction from prior and observed own outcomes')
        if state is not None:c['state']=copy.deepcopy(state)
        else:c['state'].update(primary_need='belonging',mode_urgency=.1)
        if policy is None:
            policy=Policy(principle_priority='personal',commitments={owner_key(c):rigidity});fixed=owner_key(c)
        assert fixed==owner_key(c)
        prior=belief.record();d=policy.choose(c,False);key=d['action_id'];state=d['next_state']
        selected_models=models if key=='old-way' else {k:{'success':.65,'failure':.35} for k in models}
        forecast=belief.snapshot().predict(selected_models)
        # The future shift and actual Bernoulli outcome belong ONLY to driver.
        true_probability=(.1 if changed and tick>=12 else .9) if key=='old-way' else .65
        observed=None;feedback=None
        if key!='leave':
            observed='success' if rng(seed,tick,key,'actual-pledge').random()<true_probability else 'failure'
            health=max(0.,health-(.015 if observed=='success' else .07))
            feedback=belief.observe(selected_models,observed,f'pledge:{tick}',forecast=forecast)
        assert HypothesisTracker.from_record(belief.record()).record()==belief.record()
        trace.append(dict(tick=tick,context_hash=digest(c),owner=fixed,action=key,
            estimate=estimate,health=health,observed=observed,decision=d,belief_before=prior,
            feedback=feedback,belief_after=belief.record()))
        if health<=0:break
    return dict(seed=seed,rigidity=rigidity,inertia=inertia,changed=changed,steps=len(trace),
        died=health<=0,health=health,protected=sum(t['observed']=='success' for t in trace),
        leave=sum(t['action']=='leave' for t in trace),old_after_change=sum(t['tick']>=12 and t['action']=='old-way' for t in trace),
        other_after_change=sum(t['tick']>=12 and t['action']=='other-way' for t in trace),trace=trace,
        limitation='authored stochastic pledge diagnostic; same stated benevolence, no human psychological validation')
