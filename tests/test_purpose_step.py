import copy
import unittest
from reflex.closed_loop import owned_rollout
from reflex.purpose_step import purpose_step
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame, GAMES


class PurposeStepTests(unittest.TestCase):
    def test_exact_one_step_equivalence_and_learning_lifecycle(self):
        for game in GAMES:
            for profile in PROFILES[:4]:
                g=ContinuingGame(game,profile,turns=3);w=g.start();actor=g.actor(g.context(w),learning='segmented')
                for symbol in ('2','0','1'):
                    req=g.request(g.context(w,actor.state));c=actor.preview(req)[0];saved=actor.record()
                    slow=owned_rollout(actor,observe=g.context,events=g.events,terminal=g.terminal,
                        assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
                        feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=0)
                    fast=purpose_step(actor,w,events=g.events,terminal=g.terminal,
                        assess=lambda w:g.goal(w).record(),target=g.target)
                    a=slow.planner(actor,req,w)([c],[]);b=fast([c],[])
                    self.assertEqual(a.contexts,b.contexts)
                    self.assertEqual(a.purpose,b.purpose)
                    self.assertEqual(a.roots,b.roots)
                    self.assertEqual(a.return_frame,b.return_frame)
                    self.assertEqual(actor.record(),saved)
                    left=copy.deepcopy(actor);right=copy.deepcopy(actor)
                    x=DecisionLoop.decide_batch([(left,req)],False,lambda cs,ds:a)[0]
                    y=DecisionLoop.decide_batch([(right,req)],False,lambda cs,ds:b)[0]
                    self.assertEqual(x['decision'],y['decision'])
                    key=x['decision']['action_id'];after,flow=g.transition(w,key,symbol)
                    for person,result in ((left,x),(right,y)):
                        resolve_feedback(person,result['ticket'],g.feedback(w,after,key,c,flow))
                    self.assertEqual(left.record(),right.record())
                    actor=right;w=after

    def test_stale_planner_and_missing_response_are_rejected(self):
        g=ContinuingGame('auction',PROFILES[0],turns=3);w=g.start();own=g.actor(g.context(w))
        req=g.request(g.context(w));c=own.preview(req)[0]
        bad=purpose_step(own,w,events=lambda w,k:g.events(w,k)[:-1],terminal=g.terminal,
            assess=lambda w:g.goal(w).record(),target=g.target)
        with self.assertRaises(ValueError):bad([c],[])
        planner=purpose_step(own,w,events=g.events,terminal=g.terminal,assess=lambda w:g.goal(w).record(),target=g.target)
        out=DecisionLoop.decide_batch([(own,req)],False,planner)[0];key=out['decision']['action_id']
        after,flow=g.transition(w,key,'2');resolve_feedback(own,out['ticket'],g.feedback(w,after,key,c,flow))
        with self.assertRaises(ValueError):planner([c],[])


if __name__=='__main__':unittest.main()
