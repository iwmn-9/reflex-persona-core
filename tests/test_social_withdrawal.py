import copy
import unittest
import numpy as np
from reflex.core import Policy, compile_batch
from reflex.social import SocialBinding, SocialEvent
from reflex.social_projection import CoarseSocialMemory
from reflex.social_withdrawal import WithdrawalSocialMemory, WithdrawalSocialPopulation
from reflex.examples import action, effect
from test_social import fixture


class WithdrawalTests(unittest.TestCase):
    def test_dislike_reduces_help_but_does_not_reward_harm(self):
        c=fixture();b=compile_batch([c]);m=WithdrawalSocialMemory(c)
        for t in range(8):m.observe(SocialEvent('partner',str(t),t,-1))
        bindings={'help':SocialBinding('partner',1),'revenge':SocialBinding('partner',-1)}
        scores,audit=m.scores(b.ids[0],b.targets[0],bindings,8)
        self.assertLess(scores[b.ids[0].index('help')],0)
        self.assertEqual(scores[b.ids[0].index('revenge')],0)
        old=CoarseSocialMemory.from_record(c,m.record());original,_=old.scores(b.ids[0],b.targets[0],bindings,8)
        self.assertGreater(original[b.ids[0].index('revenge')],0)
        self.assertEqual(m.record(),old.record());self.assertEqual(audit['revenge']['stance'],-.5)

    def test_gain_and_principles_can_still_select_claiming_without_a_grudge_bonus(self):
        c=fixture();c['values']={k:0. for k in c['values']};c['values']['power']=.9
        claim=action('claim',effect(.8,values={'power':1}));claim['target']='partner'
        c['actions']=[claim,action('alone',effect(.3))];b=compile_batch([c]);m=WithdrawalSocialMemory(c)
        for t in range(8):m.observe(SocialEvent('partner',str(t),t,-1))
        scores,_=m.scores(b.ids[0],b.targets[0],{'claim':SocialBinding('partner',-1)},8)
        self.assertEqual(scores.sum(),0)
        d=Policy(principle_priority='finite').decide(b,False,appraisal=scores[None,:])
        self.assertEqual(d.records(b)[0]['action_id'],'claim')
        # A deliberate value-driven material sacrifice is still possible.
        c['state']['primary_need']='physiology'
        c['actions'][0]['outcomes'][0]['objective']=-.2
        b=compile_batch([c]);m=WithdrawalSocialMemory(c)
        for t in range(8):m.observe(SocialEvent('partner',str(t),t,-1))
        scores,_=m.scores(b.ids[0],b.targets[0],{'claim':SocialBinding('partner',-1)},8)
        d=Policy(principle_priority='finite').decide(b,False,appraisal=scores[None,:])
        self.assertEqual(d.records(b)[0]['action_id'],'claim')

    def test_positive_relationship_and_neutral_path_preserve_existing_projection(self):
        c=fixture();b=compile_batch([c]);m=WithdrawalSocialMemory(c);bindings={'help':SocialBinding('partner',1),'revenge':SocialBinding('partner',-1)}
        for benefit in (0,1):
            for t in range(8*benefit,8*benefit+8):m.observe(SocialEvent('partner',str(t),t,benefit))
            old=CoarseSocialMemory.from_record(c,m.record())
            np.testing.assert_equal(m.scores(b.ids[0],b.targets[0],bindings,t)[0],old.scores(b.ids[0],b.targets[0],bindings,t)[0])

    def test_relations_remain_recipient_specific_and_population_uses_withdrawal(self):
        c=fixture();innocent=copy.deepcopy(c['actions'][0]);innocent.update(id='help_other',target='other');c['actions'].append(innocent)
        pop=WithdrawalSocialPopulation([c],Policy(principle_priority='finite'))
        bindings={'help':SocialBinding('partner',1),'revenge':SocialBinding('partner',-1),'help_other':SocialBinding('other',1)}
        for t in range(8):
            pop.step(False,bindings=[bindings]);pop.observe_batch([[SocialEvent('partner',str(t),t,-1)]])
        d=pop.step(False,bindings=[bindings]);self.assertEqual(pop.action_ids(d),['help_other'])
        self.assertEqual(pop.social_audit[0]['revenge']['adjustment'],0)


if __name__=='__main__':unittest.main()
