import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from reflex.opponent_beliefs import HypothesisTracker,BeliefSnapshot
from reflex.goofspiel import Position,Pending,RolloutModel,observe,immediate_outcomes
from reflex.goofspiel_beliefs import PublicBidBeliefs,Forecast,NAMES,hypotheses
from reflex.goofspiel_reading import decide,use_gate
from reflex.laboratory import profiles
from reflex.monte_carlo import RolloutBudget
from reflex.opponent_reading_experiment import play,prediction_probe


class OpponentBeliefTests(unittest.TestCase):
    def test_actual_caller_forecast_controls_surprise_and_invalid_input_is_atomic(self):
        t=HypothesisTracker(('uniform','high','low'));models=self.distributions()
        self.assertEqual(t.snapshot().predict(models)['A'],.5)
        r=t.observe(models,'A','unexpected',forecast={'A':.1,'B':.9})
        self.assertEqual(r['predicted_probability'],.1)
        self.assertAlmostEqual(r['effective_retention'],t.retention*.25)
        before=copy.deepcopy(t.__dict__)
        for bad in ({'A':.5},{'A':0.,'B':1.},{'A':float('nan'),'B':.5},{'A':.6,'B':.6}):
            with self.assertRaises(ValueError):t.observe(models,'A','invalid',forecast=bad)
            self.assertEqual(t.__dict__,before)

    def distributions(self):
        return dict(uniform={'A':.5,'B':.5},high={'A':0.,'B':1.},low={'A':1.,'B':0.})

    def test_cold_start_and_minimum_evidence(self):
        t=HypothesisTracker(('uniform','high','low')); models=self.distributions()
        self.assertEqual(t.snapshot().predict(models),{'A':.5,'B':.5})
        for i in range(2): t.observe(models,'B',str(i))
        self.assertEqual(t.snapshot().confidence,0.)
        t.observe(models,'B','2'); self.assertGreater(t.snapshot().confidence,0.)
        self.assertGreater(t.snapshot().predict(models)['B'],.5)
        self.assertGreater(t.snapshot().predict(models)['A'],0.)

    def test_pre_reveal_probability_and_invalid_updates_atomic(self):
        t=HypothesisTracker(('uniform','high','low')); models=self.distributions()
        row=t.observe(models,'B','first'); self.assertEqual(row['predicted_probability'],.5)
        before=t.snapshot()
        for action,id in (('X','second'),('A','first')):
            with self.assertRaises(ValueError): t.observe(models,action,id)
            self.assertEqual(t.snapshot(),before)
        bad=copy.deepcopy(models); bad['high']['B']=.8
        with self.assertRaises(ValueError): t.observe(bad,'B','second')
        self.assertEqual(t.snapshot(),before)

    def test_forced_actions_do_not_teach_a_behavior(self):
        t=HypothesisTracker(('uniform','high','low'))
        row=t.observe({k:{'A':1.} for k in t.names},'A','forced')
        self.assertTrue(row['forced']); self.assertEqual(t.snapshot().observations,0)
        self.assertEqual(t.snapshot().confidence,0.)
        for i in range(8): t.observe(self.distributions(),'B',str(i))
        forced={k:{'A':1.} for k in t.names}
        self.assertEqual(t.snapshot().predict(forced),{'A':1.})

    def test_switch_reduces_trust_and_old_evidence_is_forgotten(self):
        t=HypothesisTracker(('uniform','high','low')); models=self.distributions()
        for i in range(12): t.observe(models,'B',str(i))
        confident=t.snapshot().confidence
        t.observe(models,'A','switch')
        self.assertLess(t.snapshot().confidence,confident)
        for i in range(20): t.observe(models,'A',f'new-{i}')
        self.assertGreater(t.snapshot().weights[2],t.snapshot().weights[1])
        self.assertEqual(len(t.lifts),6)

    def test_snapshots_and_observers_are_isolated(self):
        t=PublicBidBeliefs(4,0); other=PublicBidBeliefs(4,1); snapshot=t.snapshot(); s=Position.start(4,3)
        t.reveal(s,(3,1,1,2))
        self.assertEqual(snapshot.players[1].observations,0)
        self.assertEqual(other.snapshot().players[0].observations,0)
        self.assertEqual(t.snapshot().players[0].observations,0)
        with self.assertRaises(ValueError): t.reveal(s,(3,1,1,2))

    def test_bad_public_joint_bids_do_not_partially_update_players(self):
        t=PublicBidBeliefs(4,0); before=t.snapshot()
        with self.assertRaises(ValueError): t.reveal(Position.start(4,3),(1,2,1,9))
        self.assertEqual(before,t.snapshot())

    def test_hypotheses_change_with_available_cards_and_prize(self):
        s=Position.start(3,3)
        models=hypotheses(s,1)
        self.assertEqual(models['reserve'],{'BID:1':.5,'BID:2':.5,'BID:3':0.})
        s=s.play((1,1,1)); models=hypotheses(s,1)
        self.assertNotIn('BID:1',models['reserve'])
        self.assertEqual(models['high']['BID:3'],1.)

    def trained_state(self):
        t=PublicBidBeliefs(3,0); s=Position.start(3,5)
        for _ in range(3):
            bids=(min(s.hands[0]),max(s.hands[1]),min(s.hands[2]))
            t.reveal(s,bids); s=s.play(bids)
        return s,t

    def test_conditional_analytic_probability_matches_enumeration(self):
        s,t=self.trained_state(); snapshot=t.snapshot()
        p1=snapshot.probabilities(s,1); p2=snapshot.probabilities(s,2)
        for bid in s.hands[0]:
            expected=0.
            for a,pa in p1.items():
                for b,pb in p2.items():
                    bids=(bid,int(a.split(':')[1]),int(b.split(':')[1]))
                    if bid==max(bids) and bids.count(bid)==1: expected+=pa*pb*s.prizes[s.round]/sum(s.prizes[s.round:])
            rows=immediate_outcomes(s,0,bid,'score',snapshot)
            self.assertAlmostEqual(sum(r['p'] for r in rows),1.)
            self.assertAlmostEqual(sum(r['p']*r['objective'] for r in rows),expected)

    def test_cold_model_uses_identical_common_random_trials(self):
        s=Position.start(3,3); snap=PublicBidBeliefs(3,0).snapshot(); p=profiles()[0]
        budget=RolloutBudget(samples=8,rollout_policy='random',max_steps=3)
        base,stats=observe(s,0,p,'score',9,0,'same',budget=budget,common_random=True)
        read,other=observe(s,0,p,'score',9,0,'same',budget=budget,beliefs=snap,common_random=True)
        self.assertEqual(base['actions'],read['actions']); self.assertEqual(stats['actions'],other['actions'])

    def test_root_commitment_is_hidden_and_virtual_trials_do_not_train(self):
        s,t=self.trained_state(); snap=t.snapshot(); original=t.snapshot()
        model=RolloutModel(s,0,profiles()[0],'score','random',9,0,'x',beliefs=snap,common_random=True)
        model.begin_trial(); a=model.joint_bids(Pending(s,s.hands[0][0]),np.random.default_rng(8))
        model.begin_trial(); b=model.joint_bids(Pending(s,s.hands[0][-1]),np.random.default_rng(8))
        self.assertEqual(a[1:],b[1:])
        observe(s,0,profiles()[0],'score',9,0,'x',budget=RolloutBudget(samples=8,rollout_policy='random'),beliefs=snap)
        self.assertEqual(t.snapshot(),original)

    def test_reading_budget_fallback_and_observer_validation(self):
        s,t=self.trained_state(); p=profiles()[0]
        base,d,stats=decide(s,0,p,'score',9,0,'x',budget=RolloutBudget(max_nodes=0),beliefs=t.snapshot())
        self.assertFalse(stats['reading_applied']); self.assertIn(stats['reason'],('no_reliable_public_evidence','learned_rollout_incomplete'))
        with self.assertRaises(ValueError): observe(s,1,p,'score',9,0,'x',beliefs=t.snapshot())
        _,_,stats=decide(s,0,p,'score',9,0,'cap',budget=RolloutBudget(samples=8,max_nodes=40,rollout_policy='random'),beliefs=t.snapshot())
        self.assertLessEqual(stats['total_nodes'],40)

    def test_advantage_gate_and_gain_threshold(self):
        self.assertEqual(use_gate(True,False,True,False,True,.8,.05),(False,'advantage_maintained_under_model'))
        self.assertFalse(use_gate(False,False,False,False,True,.04,.05)[0])
        self.assertTrue(use_gate(True,False,False,False,True,.1,.05)[0])
        self.assertTrue(use_gate(False,False,False,False,False,0.,.05)[0])
        self.assertFalse(use_gate(True,True,True,False,True,.8,.05)[0])

    def test_confident_uniform_forecast_skips_duplicate_evaluation(self):
        p=BeliefSnapshot(NAMES,(.99,.004,.003,.003),12,.8,.08)
        snapshot=Forecast(0,(p,p,p)); s=Position.start(3,3)
        self.assertLess(snapshot.nonuniform_mass,.05)
        _,_,stats=decide(s,0,profiles()[0],'score',9,0,'x',beliefs=snapshot)
        self.assertEqual(stats['reason'],'near_uniform_forecast'); self.assertFalse(stats['model_evaluated'])

    def test_replay_threads_use_same_frozen_beliefs(self):
        s,t=self.trained_state(); snap=t.snapshot()
        def run(_): return decide(s,0,profiles()[0],'score',9,0,'x',beliefs=snap)
        expected=run(None)
        with ThreadPoolExecutor(3) as pool: results=list(pool.map(run,range(6)))
        self.assertTrue(all(r==expected for r in results))

    def test_closed_loop_replay_and_transition_caps(self):
        a=play(4,0,'win_share','read_mc16','mixed',cards=7)
        b=play(4,0,'win_share','read_mc16','mixed',cards=7)
        self.assertTrue(a['finished']); self.assertEqual(a['trace'],b['trace'])
        self.assertTrue(all(t['reading']['total_nodes']<=t['reading']['node_cap'] for t in a['trace']))
        self.assertEqual(sum(a['scores'])+a['discarded'],28)

    def test_fixed_history_prediction_probe_includes_shift_and_unseen(self):
        summary,histories=prediction_probe(1)
        self.assertEqual(summary['histories'],18)
        self.assertEqual(len(histories),18)
        self.assertEqual({g['controller'] for g in summary['groups']},{'random','reserve','high','switch','unseen','mixed'})
        self.assertTrue(all(g['predictions']>0 for g in summary['groups']))


if __name__=='__main__': unittest.main()
