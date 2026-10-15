import copy
import unittest
from reflex.encounters import EncounterWorld, after_action
from reflex.core import compile_batch
from tools.cooperation_experiment import catalog, forecast_options, configure, run_one
from tools.cooperation_projects import run


class CooperationAdapterTests(unittest.TestCase):
    def test_exact_conditional_outcomes_match_real_public_physics(self):
        for domain in ('exchange','watch'):
            w=EncounterWorld(92000,domain,'balanced4');w.health['npc-0']=.2
            cs,_=w.contexts();offers,methods=catalog(w);saved=copy.deepcopy(w.state('npc-0'))
            opts,roots,_,_=forecast_options(w,'npc-0',cs[0],offers,methods,1)
            key='joint-npc-0-npc-1-cooperate-cooperate'
            model=next(a for a in opts if a['id']==key)
            self.assertEqual([o['p'] for o in model['outcomes']],[.95,.05])
            for flow,failed in zip(model['outcomes'],(False,True)):
                gain,hp=w.consequences('cooperate','cooperate',True,failed)
                actual=after_action(saved,'cooperate',gain,hp,w.rules['upkeep'])
                self.assertAlmostEqual(flow['objective'],gain if actual['health']>0 else -1.)
                self.assertAlmostEqual(flow['needs']['physiology'],actual['energy']-saved['energy'])
            self.assertEqual(saved,w.state('npc-0'));self.assertEqual(w.tick,0)
            c=copy.deepcopy(cs[0]);c['actions']=opts;compile_batch([c])

    def test_synchronized_recovery_does_not_enable_illegal_current_work(self):
        w=EncounterWorld(92000,'watch','balanced4');w.energy['npc-1']=.01
        cs,_=w.contexts();offers,methods=catalog(w)
        key='joint-npc-0-npc-1-cooperate-cooperate'
        offer=next(o for o in offers if o.key==key)
        self.assertEqual(dict(offer.roots),{'npc-0':'rest','npc-1':'rest'})
        opts,roots,_,bindings=forecast_options(w,'npc-0',cs[0],offers,methods,1)
        self.assertNotIn(key,bindings)  # no imaginary one-step relationship gain
        a=next(a for a in opts if a['id']==key)
        b=next(a for a in opts if a['id']=='outside-rest')
        self.assertEqual(a['outcomes'],b['outcomes'])

    def test_missing_future_announcement_is_not_read_before_rule_change(self):
        a=EncounterWorld(92000,'watch','balanced4');b=copy.deepcopy(a)
        configure(a,'standard');configure(b,'change')
        self.assertEqual(a.rules,b.rules)
        cs,_=a.contexts();offers,methods=catalog(a)
        self.assertEqual(forecast_options(a,'npc-0',cs[0],offers,methods,4),
                         forecast_options(b,'npc-0',cs[0],offers,methods,4))
        b.tick=160;configure(b,'change');self.assertNotEqual(a.rules,b.rules)

    def test_recovery_rollouts_are_bounded_and_do_not_mutate_history(self):
        for domain in ('exchange','watch'):
            w=EncounterWorld(92001,domain,'balanced8');w.energy['npc-0']=.02
            cs,_=w.contexts();offers,methods=catalog(w);before=copy.deepcopy(w.__dict__)
            opts,_,_,_=forecast_options(w,'npc-0',cs[0],offers,methods,8)
            c=copy.deepcopy(cs[0]);c['actions']=opts;compile_batch([c])
            self.assertEqual(w.__dict__,before)

    def test_public_contact_schedule_limits_future_joint_work(self):
        w=EncounterWorld(92000,'watch','balanced4','public');cs,_=w.contexts()
        offers,methods=catalog(w);target=w.available_partner('npc-0')
        key=f'joint-npc-0-{target}-cooperate-cooperate'
        public=forecast_options(w,'npc-0',cs[0],offers,methods,8)[0]
        free=copy.deepcopy(w);free.contact='free'
        unrestricted=forecast_options(free,'npc-0',cs[0],offers,methods,8)[0]
        score=lambda opts:sum(o['p']*o['values'].get('benevolence',0) for a in opts if a['id']==key for o in a['outcomes'])
        self.assertLess(score(public),score(unrestricted))

    def test_connected_execution_has_real_consent_and_personality_ownership(self):
        r,trace=run_one((None,92000,'watch','balanced4','free','standard','combined',16,8))
        self.assertTrue(r['personality_fixed']);self.assertGreater(r['agreements'],0)
        for t in trace:
            for agreed in t['negotiation']['agreements']:
                for own in agreed['members']:
                    audit=t['evaluations'][own];key=agreed['offer']
                    self.assertIn(key,audit['accepted'])
                    self.assertGreater(audit['scores'][key],audit['scores'][audit['outside_option']])

    def test_same_protocol_reduces_conflicting_material_reservations(self):
        old,_=run(92000,'balanced',2,False);new,trace=run(92000,'balanced',2,True)
        self.assertLess(new['conflicts'],old['conflicts']);self.assertGreater(new['agreements'],0)
        for t in trace:
            for negotiation in t['negotiations']:
                for agreed in negotiation['agreements']:
                    self.assertEqual(len(agreed['members']),2)


if __name__=='__main__':unittest.main()
