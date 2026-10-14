import copy
import unittest

from reflex.adaptive_dynamics import EventLearner, AdaptiveKernel, PublicEvent
from reflex.adaptive_opportunities import AdaptiveOpportunity
from reflex.core import Policy, character_key
from reflex.decision_loop import DecisionLoop
from reflex.examples import context, action, effect
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from reflex.opportunity_graph import OpportunityGraph, persona_choices
from reflex.preference_basis import PreferenceBasis
from reflex.purpose_plan import Goal
from tools.open_model_games import OpenGame
from tools.opportunity_models import OpenOpportunities


def simple(rename=lambda x:x):
    events=lambda s,a:[PublicEvent(str(i),(s[0]+1,float(a==rename(str(i)))),effect()) for i in range(2)]
    learner=EventLearner(owner='owner',model_id='rules')
    kernel=AdaptiveKernel(learner=learner,events=events,features=lambda *a:(),models=lambda *a:{},public_record=lambda s:s,model_id='rules')
    graph=OpportunityGraph(actions=lambda s:tuple(rename(str(i)) for i in range(2)),events=events,
        terminal=lambda s:s[0]==1,assess=lambda s:Goal('x','scored' if s[0] else 'running',s[1]).record(),record=lambda s:s,model_id='rules')
    graph.compile((0,0.))
    c=context('owner',[action(rename(str(i)),effect()) for i in range(2)])
    owner=DecisionLoop(c)
    def choose(states,alts):
        return persona_choices([copy.deepcopy(c) for s in states],alts,owner.policy)
    return kernel,graph,owner,choose


class AdaptiveOpportunityTests(unittest.TestCase):
    def test_real_update_invalidates_old_valuation_but_imagined_branch_does_not_learn(self):
        k,g,o,choose=simple();s=k.start((0,0.));before=k.record(s);owned=o.record()
        result=AdaptiveOpportunity(g,k,s,o,choose=choose,chooser_id='persona')
        for t in result(s,'0'):
            self.assertEqual(t.state.memory,s.memory)
            result.evaluate(t.state,o)
        self.assertEqual(k.record(s),before);self.assertEqual(o.record(),owned)
        row=k.events(s.public,'0')[0];after=k.reveal(s,'0',row.public,row.flow)
        self.assertEqual(after.memory.observations,1)
        with self.assertRaisesRegex(ValueError,'prediction changed'):result.evaluate(after,o)
        with self.assertRaisesRegex(ValueError,'prediction changed'):result.action(k.start((0,0.),after.memory))

    def test_live_revisions_and_isolated_outlier_with_arbitrary_names(self):
        for rename in (lambda x:x,lambda x:{'0':'shield','1':'advance'}[x]):
            k,g,o,choose=simple(rename);m=k.learner.start()
            for i in range(32):m=k.learner.observe(m,(),('0','1'),{},'0')
            def choice(memory):
                s=k.start((0,0.),memory)
                return AdaptiveOpportunity(g,k,s,o,choose=choose,chooser_id='persona').action(s)
            self.assertEqual(choice(m),rename('0'))
            after=k.learner.observe(m,(),('0','1'),{},'1')
            self.assertEqual(choice(after),rename('0'))
            for i in range(32):after=k.learner.observe(after,(),('0','1'),{},'1')
            self.assertEqual(choice(after),rename('1'))
            self.assertEqual(choice(m),rename('0'))  # factual old checkpoint unmodified

    def test_checkpoint_restores_identical_long_horizon_choice(self):
        k,g,o,choose=simple();s=k.start((0,0.))
        row=k.events(s.public,'0')[1];after=k.reveal(s,'0',row.public,row.flow)
        restored=k.restore(k.record(after),lambda s:s)
        roots=[k.start((0,0.),x.memory) for x in (after,restored)]
        results=[AdaptiveOpportunity(g,k,x,o,choose=choose,chooser_id='persona') for x in roots]
        self.assertEqual(results[0].identity,results[1].identity)
        self.assertEqual(results[0].action(roots[0]),results[1].action(roots[1]))

    def test_full_persona_execution_matches_bound_prediction_in_all_four_rule_families(self):
        for game in ('auction','resources','combat','gambling'):
            for p in profiles():
                g=OpenGame(game,p,length=2);s=g.start();c=g.observe(s)
                o=DecisionLoop(c,policy=Policy(principle_priority='personal',commitments={character_key(c):.5}))
                adapter=OpenOpportunities(g)
                basis=PreferenceBasis(g.target,purpose=(('values.achievement',1.),))
                table=adapter.solve(s,o,preference_basis=basis,bound=True)
                ctrl=g.controller(opportunity=table,preference_basis=basis);req=request_for(c,'open-model')
                before=g.kernel.record(s);capture={}
                d=DecisionLoop.decide_batch([(o,req)],False,ctrl.planner(o,req,s,captured=capture))[0]
                self.assertEqual(d['decision']['action_id'],table.action(s))
                self.assertTrue(capture['audit']['complete'])
                self.assertEqual(g.kernel.record(s),before)
                self.assertEqual(o.values,c['values']);self.assertEqual(o.personality,c['personality'])

    def test_different_persona_or_recompiled_graph_cannot_reuse_endpoint(self):
        k,g,o,choose=simple();s=k.start((0,0.));result=AdaptiveOpportunity(g,k,s,o,choose=choose,chooser_id='persona')
        other=DecisionLoop(context('other',[action('0',effect()),action('1',effect())],values={'benevolence':1.}))
        with self.assertRaises(ValueError):result.evaluate(s,other)
        g.compile((1,0.))
        with self.assertRaises(ValueError):result.action(s)


if __name__=='__main__':unittest.main()
