import copy
import unittest
from dataclasses import replace
from reflex.laboratory import profiles
from reflex.core import Policy,digest
from reflex.gambling import Table,offers,legal,expected,make_context,resolve,terminal
from reflex.validation_experiment import gambling,auction,resources,combat,replay
from reflex.combat import Battle,make_context as combat_context,exposure,preview


class ValidationTests(unittest.TestCase):
    def test_known_odds_and_no_borrowing(self):
        w=Table(cash=1)
        self.assertEqual(legal(w,'opportunity'),('pass','safe'))
        self.assertAlmostEqual(expected(w,'opportunity','safe'),.6)
        self.assertLess(expected(w,'opportunity','trap'),0)
        with self.assertRaises(ValueError):resolve(w,'opportunity','swing',1)
        self.assertEqual(resolve(w,'opportunity','pass',1)[0].cash,1)

    def test_public_change_not_future_schedule_in_context(self):
        c=make_context(Table(), 'change',profiles()[0],2)
        self.assertEqual(c['facts']['odds'],str(offers(Table(),'opportunity')))
        self.assertNotIn('seed',c['facts']);self.assertNotIn('change',str(c['facts']))
        self.assertEqual(offers(Table(tick=8),'change'),offers(Table(),'unfavorable'))

    def test_unfavorable_waits_without_forced_gambling_all_personas(self):
        for p in profiles():
            for mode in ('baseline','pressure'):
                r=gambling(p,40,'unfavorable',mode)
                self.assertEqual(r['score'],12);self.assertEqual(r['negative_ev_bets'],0)
                self.assertEqual(r['actions'],{'pass':16});self.assertEqual(r['max_need_pressure'],0)
                self.assertEqual(replay(r),16)

    def test_actual_draw_namespace_and_determinism(self):
        w=Table();first=resolve(w,'opportunity','safe',8)
        self.assertEqual(first,resolve(w,'opportunity','safe',8))
        c=make_context(w,'opportunity',profiles()[0],8);original=copy.deepcopy(c)
        Policy().choose(c);self.assertEqual(original,c)
        self.assertFalse(terminal(w));self.assertTrue(terminal(Table(cash=0)))

    def test_small_cross_genre_runs_restore_and_real_rules_replay(self):
        p=profiles()[0]
        for mode in ('baseline','pressure'):
            for runner,args in [(auction,dict(scenario='pattern')),(resources,dict(scenario='mixed')),(combat,dict(map_name='open',goal='either'))]:
                r=runner(p,40,mode=mode,**args);self.assertGreater(replay(r),0)
                self.assertGreater(r['decisions'],0)

    def test_combat_forecast_is_bounded_and_guard_one_tick_only(self):
        w=Battle.start('open','eliminate');units=list(w.units)
        units[0]=replace(units[0],x=3,y=2);units[3]=replace(units[3],x=5,y=2)
        w=replace(w,units=tuple(units));g=preview(w,0,'guard')
        for h in (1,3,8):self.assertTrue(0<=exposure(g,0,h)<=1)
        self.assertGreater(exposure(g,0,3),exposure(g,0,1))
        with self.assertRaises(ValueError):exposure(w,0,9)
        with self.assertRaises(ValueError):exposure(w,0,True)
        old=combat_context(w,0,profiles()[0],0,'eliminate')
        self.assertEqual(old,combat_context(w,0,profiles()[0],0,'eliminate',goal_need=False,exposure_horizon=1))
        for kw in ({'exposure_horizon':True},{'exposure_horizon':0},{'goal_need':1}):
            with self.assertRaises(ValueError):combat_context(w,0,profiles()[0],0,'eliminate',**kw)

    def test_legacy_combat_baseline_decisions_and_results_preserved(self):
        from reflex.combat_experiment import run
        p=profiles()[1];old=run(p,40,'open','either',record=False);new=combat(p,40,'open','either','baseline',record=False)
        for k in ('won','lost','ticks','health','movement_conflicts'):self.assertEqual(old[k],new[k])
        self.assertEqual(old['team_actions'],new['actions'])

    def test_legacy_auction_learned_baseline_preserved(self):
        from reflex.contests_experiment import run
        p=profiles()[1];old=run('auction',p,40,'pattern','learned',episodes=1,record=False);new=auction(p,40,'pattern','baseline',record=False)
        for k in ('won','win_credit','score','margin','payment','unused_budget'):self.assertEqual(old['episodes'][0][k],new[k])


if __name__=='__main__':unittest.main()
