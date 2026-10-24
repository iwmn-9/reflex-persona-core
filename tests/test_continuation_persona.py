"""Intrinsic values must remain operative after long-purpose evaluation.

The four persistent resource worlds do not exercise benevolence. This small
public rule fixture creates an explicit material/other-regarding tradeoff;
it is a contract check, not a new quality benchmark or a controller special case.
"""
import copy
import unittest
from reflex.core import character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.examples import effect
from reflex.laboratory import PROFILES
from reflex.model_observer import resolve_feedback
from tools.horizon_consensus import consensus_planner
from tools.preparation_world import DelayedYield
from tools.run_owned_value import cumulative_basis


class IntrinsicTradeoff(DelayedYield):
    def __init__(self, profile, names=('x','y'), turns=64, mode='principle'):
        super().__init__(1, turns=turns, profile=profile)
        self.names = names
        self.initial_mode = mode

    def context(self, world, state=None):
        context = super().context(world, state)
        if state is None:
            context['state'].update(mode=self.initial_mode, primary_need='growth', mode_urgency=.3)
        return context

    def keys(self, world):
        return () if self.terminal(world) else self.names

    def transition(self, world, key, symbol):
        if key not in self.keys(world):
            raise ValueError('legal public action required')
        private = key == self.names[0]
        after = copy.deepcopy(world)
        after.update(tick=world['tick']+1, last=key, signal='0',
            cash=world['cash']+(2 if private else 1))
        gain = (after['cash']-world['cash'])/12
        return after, effect(gain, needs={'growth':gain, 'safety':0.},
            values={'achievement':gain, 'benevolence':-.5 if private else .5})


def choose(game, minimum=32):
    world=game.start();actor=game.actor(game.context(world),learning='segmented')
    request=game.request(game.context(world,actor.state));before=actor.record();captured={}
    planner=consensus_planner(actor,request,world,observe=game.context,events=game.events,
        terminal=game.terminal,assess=lambda w:game.goal(w).record(),request_for=game.request,
        feedback=game.feedback,model_id=game.model_id,target=game.target,horizon=32,
        minimum_horizon=minimum,seeds=(701,),cumulative_preference=cumulative_basis(game,True),
        include_retry=True,purpose_estimator='conditional',tail_recovery=2,recovery_cache=64,
        coherent_cumulative_mean=True,captured=captured)
    result=DecisionLoop.decide_batch([(actor,request)],False,planner)[0]
    after,flow=game.transition(world,result['decision']['action_id'],'0')
    resolve_feedback(actor,result['ticket'],game.feedback(world,after,result['decision']['action_id'],result['context'],flow))
    factual=game.actor(game.context(world),learning='segmented')
    replay=DecisionLoop.decide_batch([(factual,request)],False,
        _counterfactual_roots=(result['decision']['action_id'],))[0]
    resolve_feedback(factual,replay['ticket'],game.feedback(world,after,result['decision']['action_id'],replay['context'],flow))
    assert actor.record()==factual.record()
    assert character_key(result['context'])==character_key(game.context(world))
    return result,captured,digest(before)


class ContinuationPersonaTests(unittest.TestCase):
    def test_long_purpose_keeps_intrinsic_personal_tradeoff(self):
        for minimum in (32,16):
            with self.subTest(minimum=minimum):
                growth,_,_=choose(IntrinsicTradeoff(PROFILES[0]),minimum)
                care,_,_=choose(IntrinsicTradeoff(PROFILES[2]),minimum)
                self.assertEqual(growth['decision']['action_id'],'x')
                self.assertEqual(care['decision']['action_id'],'y')

    def test_renamed_actions_keep_the_personal_tradeoff(self):
        result,cap,owner=choose(IntrinsicTradeoff(PROFILES[2],('zeta','alpha')))
        self.assertEqual(result['decision']['action_id'],'alpha')
        self.assertEqual(cap['horizon_consensus']['source_owner'],owner)
        self.assertEqual(len(cap['horizon_consensus']['ballots']),1)

    def test_care_is_not_an_unconditional_command_to_sacrifice(self):
        # The original H16 owner also chooses material gain in this need-mode
        # situation. A fixed personality does not imply an unchanging choice.
        result,_,_=choose(IntrinsicTradeoff(PROFILES[2],mode='need'))
        self.assertEqual(result['decision']['action_id'],'x')
        self.assertEqual(result['decision']['next_state']['mode'],'need')


if __name__=='__main__':
    unittest.main()
