import copy
from dataclasses import replace
import functools
import json
import math
import unittest

from reflex.core import Policy, character_key
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import profiles
from reflex.preference_basis import PreferenceBasis
from tools.asymmetric_games import AsymmetricGame, board, GAMES, CASES
from tools.opportunity_models import OpenOpportunities
from tools.run_asymmetry import entry, world, run_arm


class AsymmetryTests(unittest.TestCase):
    def test_resources_and_capabilities_restrict_legal_actions_and_survive_checkpoint(self):
        p=profiles()[0]
        for game in GAMES:
            for case in CASES[:6]:
                g=AsymmetricGame(game,p,board(game,case,deadline=2));s=g.start()
                restored=g.kernel.restore(g.kernel.record(s),lambda w:w)
                self.assertEqual(g.keys(s),g.keys(restored))
                self.assertFalse({'case','scenario','seed','world_law'} & s.public.keys())
                for key in g.keys(s):
                    for row in g.events(s.public,key):
                        if game in ('resources','gambling'):self.assertGreaterEqual(row.public['cash'],0)
                        if game=='combat':self.assertGreaterEqual(row.public['energy'],0)
                        if game=='auction':self.assertGreaterEqual(min(row.public['position']['budgets']),0)
                with self.assertRaises(ValueError):g.events(s.public,'unowned-ability')
        poor=AsymmetricGame('resources',p,board('resources','poor')).start()
        w=copy.deepcopy(poor.public);w['cash']=0
        g=AsymmetricGame('resources',p,w)
        self.assertEqual(g.keys(g.start()),('trade',))

    def test_late_entry_preserves_world_deadline_and_does_not_invent_personal_experience(self):
        for game in GAMES:
            args=(game,'late',profiles()[0],84000,0)
            g,prefix=entry(*args,0);s=g.start()
            self.assertEqual(len(prefix),2)
            self.assertEqual(s.tick,2);self.assertEqual(s.public['deadline'],4)
            self.assertEqual(s.memory.observations,0)
            expected=copy.deepcopy(prefix[-1]['after']);expected['joined']=True
            self.assertEqual(s.public,expected)
            row=run_arm(*args,'live',1)
            self.assertEqual(len(row['trace']),2)
            self.assertEqual(row['entries'][0]['observations_at_entry'],0)
            observer=run_arm(game,'late_observer',profiles()[0],84000,0,'live',1)
            self.assertEqual(observer['entries'][0]['public'],row['entries'][0]['public'])
            self.assertEqual(observer['entries'][0]['observations_at_entry'],2)
            self.assertEqual(observer['final_memory']['observations'],4)

    def test_changed_capabilities_are_not_collapsed_by_graph_projection(self):
        p=profiles()[0]
        for game in GAMES:
            g=AsymmetricGame(game,p,board(game,'balanced',deadline=2));s=g.start();c=g.observe(s)
            owner=DecisionLoop(c,policy=Policy(principle_priority='personal',commitments={character_key(c):.5}))
            adapter=OpenOpportunities(g)
            result=adapter.solve(s,owner,bound=True)
            other=AsymmetricGame(game,p,board(game,'powerful',deadline=2)).start()
            self.assertEqual(s.identity,other.identity)  # same rules, different public ability
            with self.assertRaises(KeyError):result.action(other)

    def test_terminal_reference_matches_independent_recursive_expectimax(self):
        # This oracle traverses raw rules directly, without the graph's state
        # projection, merge, ordering or stored terminal distributions.
        for game in GAMES:
            g=AsymmetricGame(game,profiles()[0],board(game,'specialist',deadline=3),variant=1)
            s=g.start();adapter=OpenOpportunities(g)
            @functools.lru_cache(None)
            def reference(encoded):
                w=json.loads(encoded);state=replace(g.kernel.start(w),tick=w['tick'])
                if g.terminal(state):return g.goal(state).value
                p=world(g,w)
                return max(math.fsum(p[e.symbol]*reference(json.dumps(e.public,sort_keys=True))
                    for e in g.events(w,a)) for a in g.keys(state))
            table=adapter.graph.solve(lambda w,a,syms:world(g,w),law_id='reference')
            value=math.fsum(p*v for p,v in table.distributions[table.index(s.public)])
            self.assertAlmostEqual(value,reference(json.dumps(s.public,sort_keys=True)),places=12)

    def test_combat_partial_damage_is_not_mislabeled_a_win(self):
        g=AsymmetricGame('combat',profiles()[0],board('combat'))
        w=copy.deepcopy(g.initial);w.update(tick=4,hp=6,enemy=1)
        self.assertEqual(g.goal(replace(g.kernel.start(w),tick=4)).value,.5)
        w['enemy']=0
        self.assertEqual(g.goal(replace(g.kernel.start(w),tick=4)).value,1.)


if __name__=='__main__':unittest.main()
