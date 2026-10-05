import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.combat import Battle,make_context,incoming_counts,preview,transition_effect,opponent
from reflex.laboratory import profiles
from reflex.core import Policy,compile_batch,FEATURES
from reflex.validation_experiment import Session,remember,combat,replay


def scene(changes):
    w=Battle.start('open');units=list(w.units)
    for i,fields in changes.items():units[i]=replace(units[i],**fields)
    return replace(w,units=tuple(units))


class PurposeTests(unittest.TestCase):
    def test_current_volley_has_bounded_probability_mass_and_batch_identity(self):
        w=scene({0:dict(x=3,y=2),3:dict(x=5,y=2)})
        for p in profiles():
            c=make_context(w,0,p,70,'eliminate',survival_security=True)
            for a in c['actions']:
                self.assertLessEqual(len(a['outcomes']),8)
                self.assertAlmostEqual(sum(o['p'] for o in a['outcomes']),1)
            b=compile_batch([c]);self.assertEqual(Policy().choose(c),Policy().decide(b).records(b)[0])

    def test_newly_visible_actor_cannot_be_targeted_retroactively(self):
        w=scene({0:dict(x=2,y=2),3:dict(x=7,y=2),4:dict(ammo=0),5:dict(ammo=0)})
        after=preview(w,0,'move:3:2')
        self.assertEqual(incoming_counts(w,after,0),((0,1.),))

    def test_killed_rival_still_has_committed_volley(self):
        w=scene({0:dict(x=3,y=2),3:dict(x=4,y=2,hp=3),4:dict(ammo=0),5:dict(ammo=0)})
        killed=preview(w,0,'shoot:3',True)
        self.assertEqual(killed.units[3].hp,0)
        self.assertGreater(sum(k*p for k,p in incoming_counts(w,killed,0)),0)

    def test_security_distinguishes_vital_loss_from_injury_need(self):
        w=scene({0:dict(x=3,y=2)})
        hurt=scene({0:dict(x=3,y=2,hp=6)})
        dead=scene({0:dict(x=3,y=2,hp=0)})
        a=transition_effect(w,hurt,0,'guard','eliminate',survival_security=True)
        b=transition_effect(w,dead,0,'guard','eliminate',survival_security=True)
        self.assertEqual(a['values']['security'],0)
        self.assertEqual(b['values']['security'],-1)
        self.assertLess(a['needs']['physiology'],0)
        self.assertLess(a['needs']['safety'],0)

    def test_observations_require_selected_one_turn_target_and_route_isolation(self):
        w=Battle.start();p=profiles()[0];s=Session(False,learn=True)
        c=make_context(w,0,p,70,'eliminate',survival_security=True)
        s.initialize(c);a=s.request(c)
        other=make_context(w,0,p,70,'secure',survival_security=True);b=s.request(other)
        self.assertNotEqual(a.bindings['guard'].situation,b.bindings['guard'].situation)
        self.assertEqual(a.bindings['guard'].estimated,tuple(f for f in FEATURES if f!='cost'))
        s.choose(remember(s,c))
        with self.assertRaises(ValueError):s.feedback({'esteem':0})
        s.loop.abandon(s.pending['ticket'])
        with self.assertRaises(ValueError):combat(p,70,'open','either','forecast',record=False,learn=True)

    def test_known_future_exposure_cannot_use_immediate_survival_contract(self):
        with self.assertRaises(ValueError):make_context(Battle.start(),0,profiles()[0],70,'eliminate',exposure_horizon=3,survival_security=True)

    def test_actual_switch_is_not_a_forecast_controller_label(self):
        w=scene({0:dict(x=3,y=2),3:dict(x=5,y=2)})
        for i in (0,1,2):self.assertEqual(opponent(w,i,70,'switch'),opponent(w,i,70,'reference'))
        later=replace(w,tick=8)
        for i in (0,1,2):self.assertEqual(opponent(later,i,70,'switch'),opponent(later,i,70,'raider'))
        c=make_context(later,0,profiles()[0],70,'eliminate',survival_security=True)
        self.assertNotIn('rival',c['facts']);self.assertNotIn('controller',c['facts'])

    def test_complete_learning_episode_replays_and_keeps_personality(self):
        p=profiles()[0];before=copy.deepcopy(p)
        r=combat(p,60,'open','both','baseline',learn=True,survival_security=True,rival='switch')
        self.assertGreater(replay(r),0)
        self.assertGreater(r['learned_uses'],0)
        self.assertEqual(before,p)


if __name__=='__main__':unittest.main()
