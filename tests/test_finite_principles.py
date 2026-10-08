import copy
from dataclasses import replace
import unittest
import numpy as np
from reflex.core import Policy,compile_batch,digest
from reflex.examples import context,action,effect
from reflex.decision_loop import DecisionLoop,Request
from reflex.runtime import Population
from reflex.scaling import TiledPolicy
from reflex.continuation_search import search_forecast
from reflex.purpose_plan import Goal,goal_forecast
from reflex.intertemporal import Branch
from reflex.deliberation import select


def fixture(gain=.05):
    c=context('finite',[action('finish',effect(1.)),action('ego',effect(-1.,values={'power':gain}))],
        needs={'physiology':0.,'safety':0.},values={'power':1.},mode='principle')
    c['state'].update(primary_need='belonging',mode_urgency=.1)
    return c


class FinitePrincipleTests(unittest.TestCase):
    def test_small_gain_cannot_veto_large_loss_but_large_conviction_can(self):
        p=Policy(principle_priority='finite')
        self.assertEqual(Policy().choose(fixture(),False)['action_id'],'ego')
        self.assertEqual(p.choose(fixture(),False)['action_id'],'finish')
        self.assertEqual(p.choose(fixture(1.),False)['action_id'],'ego')

    def test_scores_mode_and_fixed_preferences_are_not_rewritten(self):
        c=fixture();saved=digest(c);b=compile_batch([c])
        old=Policy().decide(b,False);new=Policy(principle_priority='finite').decide(b,False)
        np.testing.assert_array_equal(old.scores,new.scores)
        np.testing.assert_array_equal(old.mode,new.mode)
        self.assertEqual(digest(c),saved)
        self.assertTrue(new.eligible.all());self.assertFalse(old.eligible.all())

    def test_viability_is_not_relaxed_and_unknown_configuration_rejected(self):
        c=fixture(1.);c['actions'][1]['known_failure']=True
        self.assertEqual(Policy(principle_priority='finite').choose(c,False)['action_id'],'finish')
        with self.assertRaises(ValueError):Policy(principle_priority='force-win')

    def test_checkpoint_preserves_priority_and_old_checkpoints_keep_old_semantics(self):
        c=fixture();loop=DecisionLoop(c,policy=Policy(principle_priority='finite'))
        restored=DecisionLoop.from_record(c,loop.record())
        self.assertEqual(restored.policy.principle_priority,'finite')
        self.assertEqual(restored.record(),loop.record())
        with self.assertRaisesRegex(ValueError,'policy'):DecisionLoop.from_record(c,loop.record(),policy=Policy())
        old=DecisionLoop(c).record();self.assertNotIn('principle_priority',old)
        self.assertEqual(DecisionLoop.from_record(c,old).policy.principle_priority,'lexicographic')

    def test_batch_cannot_silently_mix_priority_modes(self):
        a=fixture();b=copy.deepcopy(a);b['scope']['npc']='second'
        loops=[DecisionLoop(a),DecisionLoop(b,policy=Policy(principle_priority='finite'))]
        saved=[x.record() for x in loops]
        with self.assertRaisesRegex(ValueError,'policy'):
            DecisionLoop.decide_batch([(x,Request(c,{},{})) for x,c in zip(loops,(a,b))],False)
        self.assertEqual([x.record() for x in loops],saved)

    def test_population_and_tiled_policy_use_the_same_finite_scores_and_actions(self):
        cs=[fixture(.05),fixture(1.),fixture(.3)]
        for i,c in enumerate(cs):c['scope']['npc']=str(i)
        a=Population(cs,Policy(principle_priority='finite'))
        b=Population(cs,TiledPolicy(1,principle_priority='finite'))
        for _ in range(4):
            da=a.step(False);db=b.step(False)
            np.testing.assert_array_equal(da.action,db.action)
            np.testing.assert_array_equal(da.scores,db.scores)

    def test_goal_gate_can_still_allow_explicit_personal_sacrifice(self):
        c=fixture(1.);p=Policy(principle_priority='finite')
        paths={a['id']:(Branch(1.,(a['outcomes'][0],),(1.,)),) for a in c['actions']}
        audits={k:[dict(actions=[k],terminal=True,assessment=Goal('mission',
            'success' if k=='finish' else 'failure',1. if k=='finish' else -1.).record())] for k in paths}
        f=goal_forecast(c,paths,audits,horizon=1,unit='turns',target='mission',policy=p,max_regret=2.)
        d,_=select([c],f,p);self.assertEqual(d[0]['action_id'],'ego')
        constrained=replace(f,max_regret=0.)
        d,_=select([c],constrained,p);self.assertEqual(d[0]['action_id'],'finish')

    def test_search_and_future_continuation_share_the_actual_policy(self):
        c=fixture();c['actions'][1]['outcomes'][0]['objective']=0.
        c['actions'][1]['outcomes'][0]['values']['power']=.1
        def observe(t,m):
            cc=copy.deepcopy(c);cc['tick']=t;cc['state']=m;return cc
        def advance(t,k,seed):return t+1,copy.deepcopy(next(a for a in c['actions'] if a['id']==k)['outcomes'][0])
        def evaluate(p):
            return search_forecast(c,0,observe=observe,advance=advance,terminal=lambda t:t>=2,
                assess=lambda t:Goal('mission','success',1.).record(),horizon=2,depth=0,
                target='mission',policy=p,max_regret=2.)
        for p,want in ((Policy(),'ego'),(Policy(principle_priority='finite'),'finish')):
            f=evaluate(p)
            self.assertTrue(all(n['branches'][0]['actions'][1]==want for n in f.audit['continuation_search']['proposals'].values()))
