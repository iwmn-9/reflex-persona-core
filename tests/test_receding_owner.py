import copy
import unittest
from reflex.receding_owner import RecedingOwner
from reflex.closed_loop import owned_rollout
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame,GAMES


class RecedingOwnerTests(unittest.TestCase):
    def test_every_decision_matches_fresh_manual_controller_after_feedback(self):
        for game in GAMES:
            g=ContinuingGame(game,PROFILES[0],turns=4);w=g.start();own=g.actor(g.context(w),learning='segmented')
            args=dict(observe=g.context,events=g.events,terminal=g.terminal,
                assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
                feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=1,seeds=(701,),world_model='posterior')
            continuous=RecedingOwner(own,**args)
            for symbol in ('2','2','0','1'):
                reference=copy.deepcopy(own);c=g.context(w,own.state);req=g.request(c)
                ctrl=owned_rollout(reference,**args)
                expected=DecisionLoop.decide_batch([(reference,req)],False,ctrl.planner(reference,req,w))[0]
                actual=continuous.decide(w)
                self.assertEqual(actual['decision'],expected['decision'])
                self.assertTrue(actual['deliberation']['adopted'])
                self.assertIsNotNone(continuous.last['node'])
                with self.assertRaises(ValueError):continuous.decide(w)
                key=actual['decision']['action_id'];after,flow=g.transition(w,key,symbol)
                fb=g.feedback(w,after,key,c,flow)
                continuous.resolve(actual['ticket'],fb);resolve_feedback(reference,expected['ticket'],fb)
                self.assertEqual(own.record(),reference.record())
                self.assertEqual(own.personality,g.context(w)['personality'])
                w=after
            with self.assertRaises(ValueError):continuous.decide(w)

    def test_restore_rebuilds_same_controller(self):
        g=ContinuingGame('auction',PROFILES[1],turns=3);w=g.start();c=g.context(w);own=g.actor(c,learning='segmented')
        args=dict(observe=g.context,events=g.events,terminal=g.terminal,
            assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
            feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=1,seeds=(701,),world_model='posterior')
        loop=RecedingOwner(own,**args);r=loop.decide(w);k=r['decision']['action_id'];n,f=g.transition(w,k,'2')
        loop.resolve(r['ticket'],g.feedback(w,n,k,c,f))
        restored=DecisionLoop.from_record(g.context(n),own.record(),predictor=g.actor(g.context(n),learning='segmented').predictor)
        other=RecedingOwner(restored,**args)
        self.assertEqual(loop.decide(n)['decision'],other.decide(n)['decision'])


if __name__=='__main__':unittest.main()
