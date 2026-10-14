import copy
import unittest

from reflex.adaptive_dynamics import PublicEvent
from reflex.core import Policy, character_key, digest
from reflex.examples import context, action, effect
from reflex.opportunity_graph import OpportunityGraph, Alternative, Outcome, persona_choices
from reflex.purpose_plan import Goal, goal_forecast
from reflex.intertemporal import Branch
from reflex.deliberation import select


def resource_graph(expensive_late=True, nodes=100, rename=lambda x:x):
    # Same resource may be spent now, reserved for a larger opportunity, or used
    # immediately when there is no later payback. No stock-price feature exists.
    def actions(s):return (rename('save'),rename('spend')) if s[1] else (rename('save'),)
    def events(s,a):
        t,stock,score=s;spend=a==rename('spend')
        gain=(.8 if expensive_late else .1) if t else .3
        return [PublicEvent('public',(t+1,stock-int(spend),score+(gain if spend else 0)),effect(gain if spend else 0))]
    g=OpportunityGraph(actions=actions,events=events,terminal=lambda s:s[0]==2,
        assess=lambda s:Goal('purpose','scored' if s[0]==2 else 'running',s[2]).record(),
        record=lambda s:s,model_id='resource',max_nodes=nodes)
    return g


LAW=lambda s,a,events:dict.fromkeys(events,1/len(events))


class OpportunityTests(unittest.TestCase):
    def test_resource_value_comes_from_available_future_opportunities(self):
        for late,expected in ((True,'save'),(False,'spend')):
            g=resource_graph(late);self.assertTrue(g.compile((0,1,0.)))
            result=g.solve(LAW,law_id='known')
            self.assertEqual(result.action((0,1,0.)),expected)
            self.assertEqual(result.action((1,1,0.)),'spend')

    def test_names_do_not_price_resources(self):
        g=resource_graph(rename=lambda x:{'save':'advance','spend':'rest'}[x]);g.compile((0,1,0.))
        self.assertEqual(g.solve(LAW,law_id='known').action((0,1,0.)),'advance')

    def test_changed_public_prediction_revalues_same_graph(self):
        g=OpportunityGraph(actions=lambda s:('safe','risk'),terminal=lambda s:s!='start',
            assess=lambda s:Goal('goal','running' if s=='start' else 'scored',{'start':0.,'safe':.2,'win':1.,'lose':-1.}[s]).record(),
            events=lambda s,a:[PublicEvent('certain','safe',effect())] if a=='safe' else [PublicEvent('win','win',effect()),PublicEvent('lose','lose',effect())],record=lambda s:s,model_id='law')
        g.compile('start');before=copy.deepcopy(g.nodes)
        def law(p):return lambda s,a,events:({'certain':1.} if a=='safe' else {'win':p,'lose':1-p})
        a=g.solve(law(.9),law_id='first');b=g.solve(law(.2),law_id='second')
        self.assertEqual(a.action('start'),'risk');self.assertEqual(b.action('start'),'safe')
        self.assertNotEqual(a.identity,b.identity);self.assertEqual(g.nodes,before)
        self.assertIn(-1.,[v for p,v in a.evaluate('start').outcomes])

    def test_reference_continuation_can_sacrifice_purpose(self):
        g=resource_graph();g.compile((0,1,0.))
        solution=g.solve(LAW,law_id='known',choose=lambda states,alternatives:tuple('save' for _ in states),chooser_id='never-spend')
        self.assertEqual(solution.evaluate((0,1,0.)).outcomes,((1.,0.),))

    def test_capacity_declines_all_roots_and_invalidates_old_graph(self):
        g=resource_graph(nodes=4);self.assertFalse(g.compile((0,1,0.)));self.assertIsNone(g.root);self.assertFalse(g.last['complete'])
        with self.assertRaises(ValueError):g.solve(LAW,law_id='known')
        g=resource_graph();g.compile((0,1,0.));solution=g.solve(LAW,law_id='known')
        g.compile((1,1,0.))
        with self.assertRaises(ValueError):solution.evaluate((0,1,0.))

    def test_invalid_law_and_callback_do_not_produce_partial_solution(self):
        g=resource_graph();g.compile((0,1,0.))
        for law in (lambda *a:{'public':float('nan')},lambda *a:{'hidden':1.},lambda *a:{'public':.5}):
            with self.assertRaises(ValueError):g.solve(law,law_id='broken')
            self.assertFalse(g.last['solve_complete'])
        with self.assertRaises(ValueError):g.solve(LAW,law_id='known',choose=lambda *a:('unknown',),chooser_id='bad')

    def test_cycle_and_false_settlement_rejected(self):
        g=OpportunityGraph(actions=lambda s:('wait',),events=lambda s,a:[PublicEvent('same',s,effect())],terminal=lambda s:False,
            assess=lambda s:Goal('x','running',0).record(),record=lambda s:s,model_id='loop')
        with self.assertRaises(ValueError):g.compile(0)
        g.terminal=lambda s:True
        with self.assertRaises(ValueError):g.compile(0)

    def test_only_current_reachable_continuation_is_solved(self):
        g=resource_graph();g.compile((0,1,0.))
        result=g.solve(LAW,law_id='known',initial=(1,1,0.))
        self.assertEqual(result.action((1,1,0.)),'spend')
        with self.assertRaises(ValueError):result.action((0,1,0.))
        self.assertLess(g.last['solve_nodes'],len(g.nodes))

    def test_distribution_overflow_declines_instead_of_erasing_bad_outcomes(self):
        g=OpportunityGraph(actions=lambda s:('play',),events=lambda s,a:[PublicEvent(str(i),i,effect()) for i in (-1,0,1)],
            terminal=lambda s:s is not None,assess=lambda s:Goal('x','running' if s is None else 'scored',0. if s is None else float(s)).record(),
            record=lambda s:s,model_id='three-outcomes')
        g.compile(None)
        self.assertIsNone(g.solve(LAW,law_id='known',max_atoms=2));self.assertFalse(g.last['solve_complete'])

    def test_selector_cannot_mutate_rules_or_return_atoms(self):
        g=resource_graph();g.compile((0,1,0.));before=copy.deepcopy(g.nodes)
        def chooser(states,alternatives):
            for alts in alternatives:
                for alt in alts:
                    for atom in alt.outcomes:atom.flow['objective']=99
            return tuple('save' for _ in states)
        g.solve(LAW,law_id='known',choose=chooser,chooser_id='mutator')
        self.assertEqual(g.nodes,before)

    def test_two_resources_can_be_converted_for_a_later_opportunity(self):
        def keys(s):
            t,a,b,won=s
            return ('wait',)+(('convert',) if a else ())+(('finish',) if a and b else ())
        def events(s,k):
            t,a,b,won=s
            return [PublicEvent('seen',(t+1,a-(k in ('convert','finish')),b+(k=='convert')-(k=='finish'),won or k=='finish'),effect())]
        g=OpportunityGraph(actions=keys,events=events,terminal=lambda s:s[0]==2 or s[3],
            assess=lambda s:Goal('x','scored' if s[0]==2 or s[3] else 'running',float(s[3])).record(),record=lambda s:s,model_id='conversion')
        g.compile((0,2,0,False));result=g.solve(LAW,law_id='known')
        self.assertEqual(result.action((0,2,0,False)),'convert')
        self.assertEqual(result.evaluate((0,2,0,False)).outcomes,((1.,1.),))

    def test_persona_batch_matches_existing_one_step_endpoint_chooser(self):
        from reflex.laboratory import profiles
        cases=[];alts=[];commitments={};expected=[]
        for i,p in enumerate(profiles()):
            for j in range(6):
                c=context(f'case-{i}-{j}',[action('a',effect(),confidence=.6+j*.08),action('b',effect(),switch_cost=.1)],
                    {'growth':.3,'safety':.2},p['values'],dict(zip(('openness','conscientiousness','extraversion','agreeableness','neuroticism'),p['traits'])),mode=None)
                c['state']['intent_action']='a'
                rows=(Alternative('a',(Outcome(.3,-.6+j*.1,effect(values={'achievement':.3},style={'openness':.4})),Outcome(.7,.5,effect()))),
                      Alternative('b',(Outcome(1.,.1+j*.1,effect(values={'security':.3},cost=.05)),)))
                cases.append(c);alts.append(rows);commitments[character_key(c)]=i/3
        policy=Policy(principle_priority='personal',commitments=commitments)
        for c,choices in zip(cases,alts):
            paths={a.key:[Branch(o.probability,(o.flow,),(1.,)) for o in a.outcomes] for a in choices}
            audit={a.key:[dict(actions=[a.key],terminal=False,assessment=Goal('x','running',o.value).record()) for o in a.outcomes] for a in choices}
            forecast=goal_forecast(c,paths,audit,horizon=1,unit='decision',target='x',policy=policy)
            decisions,_=select((c,),forecast,policy);expected.append(decisions[0]['action_id'])
        self.assertEqual(persona_choices(cases,alts,policy),tuple(expected))

    def test_persona_continuation_matches_actual_replanning_under_fixed_belief(self):
        from reflex.laboratory import profiles
        from reflex.decision_loop import DecisionLoop
        from reflex.closed_loop import reveal
        from reflex.model_observer import ModelFeedback
        from reflex.integrated_trials import request_for
        from tools.open_model_games import OpenGame
        from tools.opportunity_models import OpenOpportunities,canonical
        for name in ('auction','resources','combat','gambling'):
            g=OpenGame(name,profiles()[2],length=2);s=g.start(3);c=g.observe(s)
            actor=DecisionLoop(c,policy=Policy(principle_priority='personal',commitments={character_key(c):.5}))
            adapter=OpenOpportunities(g,3);solution=adapter.solve(s,actor)
            before=(g.kernel.record(s),actor.record())
            other=copy.deepcopy(actor);other.character='different-person'
            with self.assertRaises(ValueError):solution.evaluate(s,other)
            def execute(world,owner):
                if g.terminal(world):return g.goal(world).value
                own=copy.deepcopy(owner);cc=g.observe(world,own.state);req=request_for(cc,'open-model')
                plan=adapter.solve(world,own);ctrl=g.controller();ctrl.frontier=plan
                decision=DecisionLoop.decide_batch([(own,req)],False,ctrl.planner(own,req,world))[0]
                key=decision['decision']['action_id'];self.assertEqual(key,plan.action(world))
                reveal(own,decision['ticket'],ModelFeedback(None))
                rows,_,_,_,p=g.kernel.forecast(world,key)
                return sum(p[r.symbol]*execute(adapter.state(r.public,world.memory),own) for r in rows)
            actual=execute(s,actor);predicted=sum(p*v for p,v in solution.evaluate(s,actor).outcomes)
            self.assertAlmostEqual(actual,predicted);self.assertEqual(before,(g.kernel.record(s),actor.record()))
            # The declared public projection preserves rules, goal and features.
            for n in adapter.graph.nodes:
                w=n['state'];projected=canonical(name,w)
                self.assertEqual(g.goal(adapter.state(w)),g.goal(adapter.state(projected)))
                for key in g.keys(adapter.state(w)):
                    self.assertEqual(g.features(w,key),g.features(projected,key))
                    left=[(r.symbol,canonical(name,r.public),r.flow) for r in g.events(w,key)]
                    right=[(r.symbol,canonical(name,r.public),r.flow) for r in g.events(projected,key)]
                    self.assertEqual(left,right)


if __name__=='__main__':unittest.main()
