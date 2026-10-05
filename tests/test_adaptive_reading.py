from dataclasses import replace
from itertools import permutations,product
import unittest
import numpy as np
from reflex.core import Policy,compile_batch
from reflex.goofspiel import Position,RolloutModel,Pending,make_context
from reflex.goofspiel_beliefs import PublicBidBeliefs,NAMES,CONTEXT_NAMES,hypotheses,under_pressure
from reflex.opponent_beliefs import HypothesisTracker,BeliefSnapshot
from reflex.goofspiel_beliefs import Forecast
from reflex.adaptive_reading_experiment import (prediction_probe,own_adversity_probe,score_sensitive_positions,
                                               adversity_positions,exact_small_actions,play)
from reflex.laboratory import profiles
from reflex.planning import from_vector,vector


class AdaptiveReadingTests(unittest.TestCase):
    def models(self):
        return {'uniform':{'A':1/3,'B':1/3,'C':1/3},'high':{'A':0.,'B':0.,'C':1.},
                'low':{'A':1.,'B':0.,'C':0.}}

    def test_surprise_immediately_reduces_trust_and_old_weight(self):
        old=HypothesisTracker(('uniform','high','low'),responsive=False)
        new=HypothesisTracker(('uniform','high','low'),responsive=True)
        for i in range(8):
            old.observe(self.models(),'C',str(i)); new.observe(self.models(),'C',str(i))
        before=new.snapshot().confidence
        a=old.observe(self.models(),'A','changed'); b=new.observe(self.models(),'A','changed')
        self.assertLess(b['after']['confidence'],before)
        self.assertEqual(a['effective_retention'],.9); self.assertEqual(b['effective_retention'],.225)
        self.assertGreater(new.snapshot().predict(self.models())['A'],old.snapshot().predict(self.models())['A'])

    def test_forced_moves_preserve_surprise_and_evidence(self):
        t=HypothesisTracker(('uniform','high','low'))
        for i in range(4):t.observe(self.models(),'C',str(i))
        t.observe(self.models(),'A','shock'); before=t.snapshot()
        t.observe({n:{'A':1.} for n in t.names},'A','forced')
        self.assertEqual(before,t.snapshot())

    def test_situation_hypotheses_use_public_score_only(self):
        s=Position.start(3,7); leading=replace(s,scores=(8,0,0)); trailing=replace(s,scores=(0,8,0))
        self.assertFalse(under_pressure(leading,0)); self.assertTrue(under_pressure(trailing,0))
        a=hypotheses(leading,0,CONTEXT_NAMES); b=hypotheses(trailing,0,CONTEXT_NAMES)
        self.assertEqual(a['pressure_attack'],a['reserve']); self.assertEqual(b['pressure_attack'],b['high'])
        self.assertEqual(a['pressure_conserve'],a['high']); self.assertEqual(b['pressure_conserve'],b['reserve'])
        self.assertEqual(len(a),6)

    def test_pressure_threshold_and_tie(self):
        s=Position.start(3,7)
        self.assertFalse(under_pressure(replace(s,scores=(0,3,0)),0))
        self.assertTrue(under_pressure(replace(s,scores=(0,4,0)),0))
        self.assertFalse(under_pressure(replace(s,scores=(4,4,0)),0))

    def test_forecast_cache_distinguishes_scores_with_identical_hands(self):
        s=Position.start(3,7); p=BeliefSnapshot(CONTEXT_NAMES,(0.,0.,0.,0.,1.,0.),8,.9,.08)
        snap=Forecast(0,(p,p,p)); model=RolloutModel(s,0,profiles()[0],'score','random',2,0,'cache',beliefs=snap)
        ahead=replace(s,scores=(0,8,0)); behind=replace(s,scores=(8,0,0))
        model.begin_trial(); a=model.joint_bids(Pending(ahead,2),np.random.default_rng(4))
        model.begin_trial(); b=model.joint_bids(Pending(behind,2),np.random.default_rng(4))
        self.assertNotEqual(a[1],b[1]); self.assertEqual(ahead.hands,behind.hands)

    def test_contextual_and_legacy_are_explicit(self):
        old=PublicBidBeliefs(3,0,contextual=False,responsive=False).snapshot()
        new=PublicBidBeliefs(3,0).snapshot()
        self.assertEqual(old.players[1].names,NAMES); self.assertFalse(old.players[1].responsive)
        self.assertEqual(new.players[1].names,CONTEXT_NAMES); self.assertTrue(new.players[1].responsive)
        s=Position.start(3,7)
        self.assertEqual(old.probabilities(s,1),new.probabilities(s,1))

    def test_legal_counterfactuals_preserve_hands_prizes_and_personality(self):
        for states in (adversity_positions(3),score_sensitive_positions()):
            a,b=states; self.assertEqual(a.hands,b.hands); self.assertEqual(a.prizes,b.prizes)
            self.assertEqual(a.round,b.round); self.assertEqual(sum(a.scores)+a.discarded,sum(b.scores)+b.discarded)
            self.assertGreater(a.scores[0],max(a.scores[1:])); self.assertLess(b.scores[0],max(b.scores[1:]))
            ca=make_context(a,0,profiles()[0],'win_share',2,4,'same')
            cb=make_context(b,0,profiles()[0],'win_share',2,4,'same')
            for key in ('personality','values','needs','state','seed','scope'):self.assertEqual(ca[key],cb[key])

    def test_exact_goal_reference_matches_independent_list_referee(self):
        base=score_sensitive_positions()[1]; packed,values=exact_small_actions(base)
        for root in base.hands[0]:
            shares=[]
            for rest,a,b in product(permutations(c for c in base.hands[0] if c!=root),permutations(base.hands[1]),permutations(base.hands[2])):
                scores=list(base.scores); orders=((root,)+rest,a,b)
                for t in range(3):
                    bids=tuple(h[t] for h in orders)
                    if bids.count(max(bids))==1:scores[bids.index(max(bids))]+=base.prizes[base.round+t]
                winners=[i for i,p in enumerate(scores) if p==max(scores)]
                shares.append(1/len(winners) if 0 in winners else 0.)
            self.assertAlmostEqual(values[f'BID:{root}'],np.mean(shares))
        c=make_context(base,0,profiles()[0],'win_share',2,4,'same',outcomes=packed)
        self.assertEqual(Policy().choose(c,False)['action_id'],'BID:6')

    def test_own_adversity_changes_without_mode_or_randomness_change(self):
        r=own_adversity_probe(); self.assertEqual(r['pairs'],32)
        self.assertTrue(all(row['leading']['mode']==row['trailing']['mode'] for row in r['rows']))
        row=next(x for x in r['rows'] if x['case']=='score_sensitive_diagnostic' and x['profile']=='growth' and x['seed']==2)
        self.assertEqual(row['leading']['action'],'BID:5'); self.assertEqual(row['trailing']['action'],'BID:6')
        self.assertGreater(row['trailing']['goal_values']['BID:6'],row['trailing']['goal_values']['BID:5'])

    def test_roundoff_fix_does_not_hide_materially_invalid_effects(self):
        v=np.zeros(22); v[0]=1+2e-16
        row=from_vector(v,1.)
        self.assertEqual(row['objective'],1.)
        v[0]=1.01; bad=from_vector(v,1.)
        self.assertEqual(bad['objective'],1.01)
        packed,_=exact_small_actions(adversity_positions(3)[0])
        c=make_context(adversity_positions(3)[0],0,profiles()[0],'win_share',0,4,'bounded',outcomes=packed)
        compile_batch([c])

    def test_score_triggered_histories_and_scheduled_shift_are_separate(self):
        r,h=prediction_probe(1)
        self.assertEqual(r['histories'],15)
        scheduled=next(g for g in r['groups'] if g['controller']=='scheduled' and g['phase']=='first_two_after_change' and g['model']=='legacy4')
        fast=next(g for g in r['groups'] if g['controller']=='scheduled' and g['phase']=='first_two_after_change' and g['model']=='responsive4')
        self.assertLess(fast['log_loss'],scheduled['log_loss'])
        for history in h:
            if history['controller'] not in ('state_attack','state_conserve'):continue
            for t in history['trace']:
                pressure=max(t['scores_before'])-t['scores_before'][1]>=6.5
                expected=('high' if pressure else 'reserve') if history['controller']=='state_attack' else ('reserve' if pressure else 'high')
                self.assertEqual(t['target_mode'],expected)

    def test_closed_loop_replay_cap_and_reveals(self):
        a=play(3,0,'state_attack','responsive6'); b=play(3,0,'state_attack','responsive6')
        self.assertTrue(a['finished']); self.assertEqual(a['trace'],b['trace'])
        self.assertEqual(sum(a['scores'])+a['discarded'],91)
        self.assertTrue(all(t['reading']['total_nodes']<=20000 for t in a['trace']))
        self.assertTrue(all(t['observations']['player-1']['before']['observations']==t['round'] for t in a['trace']))


if __name__=='__main__':unittest.main()
