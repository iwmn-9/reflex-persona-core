import copy
import unittest
from reflex.core import compile_batch
from reflex.encounters import EncounterWorld, run


class EncounterTests(unittest.TestCase):
    def test_public_slots_cover_pairs_without_forcing_actions(self):
        w=EncounterWorld(4000,'exchange','balanced8','public');pairs=set()
        for tick in range(0,28,4):
            w.tick=tick
            for own in w.actors:
                target=w.available_partner(own);self.assertNotEqual(own,target)
                self.assertEqual(w.available_partner(target),own);pairs.add(tuple(sorted((own,target))))
            cs,_=w.contexts();compile_batch(cs)
            for c in cs:
                legal={a['id'] for a in c['actions'] if a['legal']}
                own=c['scope']['npc'];target=w.available_partner(own)
                self.assertEqual(legal,{'alone','rest',f'cooperate:{target}',f'claim:{target}'})
        self.assertEqual(len(pairs),28)

    def test_simultaneous_opposing_choices_have_real_asymmetric_consequences(self):
        w=EncounterWorld(4000,'exchange','balanced4','public');a=w.actors[0];b=w.available_partner(a)
        choices={x:'alone' for x in w.actors};choices[a]=f'cooperate:{b}';choices[b]=f'claim:{a}'
        rows,events=w.resolve(choices);by={r['npc']:r for r in rows}
        self.assertEqual(by[a]['gain'],-.15);self.assertEqual(by[b]['gain'],.8)
        self.assertEqual(w.score[a],-.15);self.assertEqual(w.score[b],.8)
        self.assertEqual(events[w.actors.index(a)][0].benefit,-1)
        self.assertEqual(events[w.actors.index(b)][0].benefit,.8)

    def test_forecasts_use_past_public_choices_not_other_personalities(self):
        w=EncounterWorld(4000,'exchange','balanced4','public');own=w.actors[0];target=w.available_partner(own)
        before=w.contexts()[0][0];p=w.forecast(own,target)
        for probability in p:self.assertAlmostEqual(probability,1/3)
        w.profiles[target]['values']={};w.profiles[target]['traits']=(0,)*5
        self.assertEqual(w.contexts()[0][0],before)
        w.history[target].extend([('claim',own)]*8)
        pc,pt,_=w.forecast(own,target);self.assertLess(pc,.1);self.assertGreater(pt,.8)
        self.assertEqual(len(w.history[target]),8)

    def test_random_joint_failure_has_shared_luck_and_no_betrayal_feedback(self):
        w=EncounterWorld(4000,'watch','balanced4','public');a=w.actors[0];b=w.available_partner(a)
        found=False
        for tick in range(80):
            w.tick=tick;w.energy={x:.8 for x in w.actors};w.health={x:8. for x in w.actors}
            b=w.available_partner(a);choices={x:'alone' for x in w.actors}
            choices[a]=f'cooperate:{b}';choices[b]=f'cooperate:{a}'
            rows,events=w.resolve(choices);by={r['npc']:r for r in rows}
            self.assertEqual(by[a]['luck_failure'],by[b]['luck_failure'])
            if by[a]['luck_failure']:
                found=True
                for own in (a,b):
                    e=events[w.actors.index(own)][0];self.assertFalse(e.agency);self.assertEqual(e.benefit,0)
                break
        self.assertTrue(found)

    def test_invalid_batch_and_dead_actors_do_not_mutate_state(self):
        w=EncounterWorld(4000,'watch','balanced4','public');saved=copy.deepcopy(w.__dict__)
        choices={a:'alone' for a in w.actors};choices[w.actors[-1]]='invalid'
        with self.assertRaises(ValueError):w.resolve(choices)
        for key in ('energy','stock','health','score','tick','history'):self.assertEqual(w.__dict__[key],saved[key])
        dead=w.actors[0];w.health[dead]=0;before=w.state(dead)
        choices={a:'alone' for a in w.actors};choices[dead]='finished'
        rows,events=w.resolve(choices)
        self.assertEqual(w.state(dead),before);self.assertEqual(events[0],[])
        self.assertEqual(rows[0]['gain'],0)

    def test_npc_actions_are_reproducible_and_fixed_axes_remain(self):
        a,t=run(4000,'exchange','social4','coarse',12,contact='public')
        b,u=run(4000,'exchange','social4','coarse',12,contact='public')
        self.assertEqual(a,b);self.assertEqual(t,u)
        self.assertTrue(all(x['fixed_personality'] for x in t))
        self.assertGreater(a['matches'],0)


if __name__=='__main__':unittest.main()
