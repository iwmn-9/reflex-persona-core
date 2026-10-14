import copy
import unittest

from reflex.core import Policy, character_key
from reflex.examples import context, action, effect
from reflex.preference_basis import PreferenceBasis
from reflex.opportunity_graph import Alternative, Outcome, OpportunityGraph, persona_choices
from reflex.adaptive_dynamics import PublicEvent
from reflex.purpose_plan import Goal, goal_forecast
from reflex.intertemporal import Branch
from reflex.deliberation import select
from reflex.decision_audit import policy_loss_audit


class PreferenceBasisTests(unittest.TestCase):
    def test_only_explicit_estimates_change_and_input_remains_intact(self):
        f=effect(.1,needs={'safety':-.3,'esteem':.2},values={'security':-.8,'benevolence':.7,'achievement':.1},style={'openness':.2},cost=.1,p=.4)
        before=copy.deepcopy(f)
        basis=PreferenceBasis('x',('needs.safety','values.security'),(('values.achievement',1.),))
        out=basis.resolve(f,.6,target='x',complete=True)
        self.assertEqual(f,before)
        self.assertEqual(out,effect(.1,needs={'esteem':.2},values={'benevolence':.7,'achievement':.6},style={'openness':.2},cost=.1,p=.4))
        # Same numeric signal deliberately declared intrinsic remains present.
        self.assertEqual(PreferenceBasis('x').resolve(f,.6,target='x',complete=True),f)

    def test_unknown_mixed_or_duplicate_channels_are_not_silently_reinterpreted(self):
        for args in (('x',('cost',)),('x',('values.security','values.security')),
                     ('x',('values.security',),(('values.security',1.),)),('x',(),(('values.achievement',2.),))):
            with self.assertRaises(ValueError):PreferenceBasis(*args)
        basis=PreferenceBasis('x')
        for target,complete in (('y',True),('x',False),('x',1)):
            with self.assertRaises(ValueError):basis.resolve(effect(),.2,target=target,complete=complete)

    def test_intrinsic_principle_can_still_cost_purpose(self):
        c=context('care',[action('keep',effect()),action('share',effect())],values={'benevolence':1.},mode='principle')
        p=Policy(principle_priority='personal',commitments={character_key(c):1.})
        alts=(Alternative('keep',(Outcome(1.,.9,effect(values={'benevolence':-.8})),)),
              Alternative('share',(Outcome(1.,.1,effect(values={'benevolence':.8})),)))
        audit=[];before=copy.deepcopy(c)
        choice=persona_choices([c],[alts],p,preference_basis=PreferenceBasis('x',purpose=(('values.achievement',1.),)),target='x',audit=audit)
        self.assertEqual(choice,('share',));self.assertEqual(c,before)
        self.assertFalse(audit[0]['actions']['keep']['principle_eligible'])
        self.assertTrue(audit[0]['actions']['share']['corridor_eligible'])

    def test_same_security_trait_distinguishes_tactical_advice_from_intrinsic_preference(self):
        c=context('security',[action('hold',effect()),action('use',effect())],values={'security':1.},mode='principle')
        policy=Policy(principle_priority='personal',commitments={character_key(c):1.})
        alts=(Alternative('hold',(Outcome(1.,.7,effect()),)),
              Alternative('use',(Outcome(1.,.8,effect(values={'security':-.9})),)))
        old=copy.deepcopy(c);signature=policy.signature()
        intrinsic=persona_choices([c],[alts],policy,preference_basis=PreferenceBasis('x'),target='x')
        instrumental=persona_choices([c],[alts],policy,preference_basis=PreferenceBasis('x',('values.security',)),target='x')
        self.assertEqual(intrinsic,('hold',));self.assertEqual(instrumental,('use',))
        self.assertEqual(c,old);self.assertEqual(policy.signature(),signature)

    def test_arbitrary_game_and_action_names_do_not_determine_repair(self):
        for names in (('hold','use'),('attack','heal'),('buy','sell'),('zeta','alpha')):
            c=context('semantic-control',[action(k,effect()) for k in names],values={'security':1.},mode='principle')
            c['scope']['game']=names[0]
            policy=Policy(principle_priority='personal',commitments={character_key(c):1.})
            alts=tuple(sorted((Alternative(names[0],(Outcome(1.,.7,effect()),)),
                              Alternative(names[1],(Outcome(1.,.8,effect(values={'security':-.9})),))),key=lambda a:a.key))
            self.assertEqual(persona_choices([c],[alts],policy,preference_basis=PreferenceBasis('x',('values.security',)),target='x'),(names[1],))

    def test_incomplete_or_wrong_goal_frontier_cannot_erase_preferences(self):
        from reflex.closed_loop import ClosedLoop, TreeBudget, Transition
        from reflex.frontier_value import EndpointEstimate
        from reflex.decision_loop import DecisionLoop
        from reflex.integrated_trials import request_for
        from reflex.model_observer import ModelFeedback
        c=context('incomplete',[action('a',effect(values={'security':-.4}))])
        owner=DecisionLoop(c)
        class Incomplete:
            identity='partial'
            def evaluate(self,s,o):return EndpointEstimate(((1.,.5),),'partial',.5)
        args=dict(observe=lambda s,state=None:c,transitions=lambda s,a:[Transition(1.,s+1,effect(values={'security':-.4}))],
            terminal=lambda s:False,assess=lambda s:Goal('x','running',0.).record(),record=lambda s:s,
            request_for=lambda c:request_for(c,'test'),feedback=lambda *a:ModelFeedback(None),unit='step',target='x',model_id='test',
            budget=TreeBudget(horizon=1),frontier=Incomplete(),preference_basis=PreferenceBasis('x',('values.security',)))
        ctrl=ClosedLoop(**args);req=request_for(c,'test');before=owner.record()
        with self.assertRaises(ValueError):DecisionLoop.decide_batch([(owner,req)],False,ctrl.planner(owner,req,0))
        self.assertEqual(owner.record(),before)
        with self.assertRaises(ValueError):ClosedLoop(**dict(args,budget=TreeBudget(horizon=2)))
        with self.assertRaises(ValueError):ClosedLoop(**dict(args,preference_basis=PreferenceBasis('wrong')))

    def test_resolved_batch_matches_execution_forecast_including_losses(self):
        from reflex.laboratory import profiles
        for profile in profiles():
            for confidence in (.6,1.):
                c=context('match',[action('a',effect(),confidence=confidence),action('b',effect())],
                    {'safety':.2},profile['values'],dict(zip(('openness','conscientiousness','extraversion','agreeableness','neuroticism'),profile['traits'])),mode=None)
                p=Policy(principle_priority='personal',commitments={character_key(c):.5})
                alts=(Alternative('a',(Outcome(.3,-.7,effect(values={'security':-.8,'achievement':.8})),Outcome(.7,.8,effect()))),
                      Alternative('b',(Outcome(1.,.2,effect(values={'benevolence':.3})),)))
                basis=PreferenceBasis('x',('values.security',),(('values.achievement',1.),))
                paths={a.key:[Branch(o.probability,(o.flow,),(1.,)) for o in a.outcomes] for a in alts}
                audit={a.key:[dict(actions=[a.key],terminal=False,terminal_coverage=True,assessment=Goal('x','running',o.value).record()) for o in a.outcomes] for a in alts}
                f=goal_forecast(c,paths,audit,horizon=1,unit='decision',target='x',policy=p,preference_basis=basis)
                decisions,_=select((c,),f,p)
                self.assertEqual(persona_choices([c],[alts],p,preference_basis=basis,target='x'),(decisions[0]['action_id'],))
                uncertain={k:[Branch(row.probability,row.effects,(.6,)) for row in rows] for k,rows in paths.items()}
                with self.assertRaises(ValueError):goal_forecast(c,uncertain,audit,horizon=1,unit='decision',target='x',policy=p,preference_basis=basis)
                for rows in audit.values():
                    for row in rows:row.pop('terminal_coverage')
                with self.assertRaises(ValueError):goal_forecast(c,paths,audit,horizon=1,unit='decision',target='x',policy=p,preference_basis=basis)

    def test_signed_loss_decomposition_uses_fixed_policies_and_reference_law(self):
        graph=OpportunityGraph(actions=lambda s:('a','b'),events=lambda s,a:[PublicEvent('ok',a+'+',effect()),PublicEvent('bad',a+'-',effect())],
            terminal=lambda s:s!='root',assess=lambda s:Goal('x','running' if s=='root' else 'scored',{'root':0.,'a+':1.,'a-':-1.,'b+':.3,'b-':.3}[s]).record(),record=lambda s:s,model_id='audit')
        graph.compile('root')
        actor_law=lambda s,a,syms:{'ok':.9,'bad':.1}
        ref_law=lambda s,a,syms:{'ok':.4,'bad':.6}
        informed=graph.solve(ref_law,law_id='reference')
        repaired=graph.solve(actor_law,law_id='actor')
        legacy=graph.solve(actor_law,law_id='actor',choose=lambda ss,aa:('b',)*len(ss),chooser_id='biased')
        calls=[]
        def captured_reference(s,a,symbols):
            calls.append((s,a));return ref_law(s,a,symbols)
        result=policy_loss_audit(graph,'root',captured_reference,reference_id='reference',informed=informed,repaired=repaired,legacy=legacy)
        self.assertEqual(sorted(calls),[('root','a'),('root','b')])
        self.assertAlmostEqual(result['loss']['persona'],0.)
        self.assertAlmostEqual(result['loss']['prediction'],.5)
        self.assertAlmostEqual(result['loss']['proxy'],-.5)  # stale proxy happened to help
        self.assertAlmostEqual(result['total'],0.)
        # Unlike evaluating the policies under their own (incomparable) laws.
        self.assertAlmostEqual(result['values']['repaired'],-.2)

    def test_current_compromise_and_future_cost_are_not_the_same_corridor(self):
        transitions={'root':{'a':'middle','b':'end-b'},'middle':{'c':'end-c','d':'end-d'}}
        graph=OpportunityGraph(actions=lambda s:tuple(transitions[s]),
            events=lambda s,k:[PublicEvent('seen',transitions[s][k],effect())],terminal=lambda s:s.startswith('end'),
            assess=lambda s:Goal('x','scored' if s.startswith('end') else 'running',{'end-b':.6,'end-c':.4,'end-d':1.}.get(s,0.)).record(),
            record=lambda s:s,model_id='two-choices')
        graph.compile('root');law=lambda s,a,syms:{'seen':1.}
        def fixed(root):return graph.solve(law,law_id='certain',choose=lambda ss,aa:tuple(root if s=='root' else 'c' for s in ss),chooser_id=root)
        informed=fixed('a');repaired=fixed('b')
        result=policy_loss_audit(graph,'root',law,reference_id='certain',informed=informed,repaired=repaired,legacy=informed,informed_legacy=repaired)
        self.assertAlmostEqual(result['loss']['persona'],.6)
        self.assertAlmostEqual(result['persona_parts']['current'],.2)
        self.assertAlmostEqual(result['persona_parts']['continuation'],.4)
        self.assertAlmostEqual(result['loss']['prediction'],-.2)
        self.assertAlmostEqual(result['interaction']['proxy_with_reference_law'],-.2)
        self.assertAlmostEqual(result['interaction']['proxy_with_actor_law'],.2)
        self.assertAlmostEqual(result['interaction']['difference'],.4)


if __name__=='__main__':unittest.main()
