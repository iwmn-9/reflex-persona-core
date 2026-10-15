import copy
from dataclasses import replace
import unittest
from reflex.cooperation import Offer, Ballot, evaluate, negotiate
from reflex.core import Policy
from reflex.examples import action, context, effect

SCOPE=('game','episode',7,'public-revision')


class VoluntaryCooperationTests(unittest.TestCase):
    def fixture(self):
        offers=[Offer('ab',(('a','work'),('b','work'))),Offer('ac',(('a','work'),('c','work')))]
        legal={a:{'work','alone'} for a in 'abc'}
        return offers,legal

    def test_refusal_leads_to_different_partner_without_commanding_refuser(self):
        offers,legal=self.fixture()
        bs=[Ballot('a',SCOPE,'alone',('ab','ac')),Ballot('b',SCOPE,'alone',()),Ballot('c',SCOPE,'alone',('ac',))]
        chosen,audit=negotiate(offers,bs,tuple('abc'),scope=SCOPE,legal=legal)
        self.assertEqual(chosen,{'a':'work','c':'work','b':'alone'})
        self.assertEqual(audit['attempts'][0]['refused'],['b'])
        self.assertEqual(audit['agreements'][0]['offer'],'ac')

    def test_no_common_goal_or_cooperation_bonus_required(self):
        c=context('private',[action('alone',effect()),action('work',effect())],values={'power':.8})
        before=copy.deepcopy(c)
        options=[action('outside',effect(.1)),action('ab',effect(.8,values={'power':.3}))]
        ranked,a=evaluate(c,options,{'outside':'alone','ab':'work'},('outside',))
        self.assertEqual(ranked,('ab',));self.assertEqual(c,before)
        self.assertEqual(a['outside_root'],'alone')

    def test_solo_advantage_and_unfair_offer_can_be_rejected(self):
        c=context('private',[action('alone',effect()),action('work',effect())])
        options=[action('outside',effect(.8)),action('ab',effect(-.8))]
        ranked,_=evaluate(c,options,{'outside':'alone','ab':'work'},('outside',))
        self.assertFalse(ranked)

    def test_personality_can_change_consent_on_same_material_terms(self):
        c=context('private',[action('alone',effect()),action('work',effect())],mode='principle',values={'benevolence':.9})
        options=[action('outside',effect(.5)),action('ab',effect(.2,values={'benevolence':.8,'power':-.8}))]
        roots={'outside':'alone','ab':'work'}
        caring,_=evaluate(c,options,roots,('outside',))
        selfish=copy.deepcopy(c);selfish['values']['benevolence']=0.;selfish['values']['power']=.9
        other,_=evaluate(selfish,options,roots,('outside',))
        self.assertEqual(caring,('ab',));self.assertFalse(other)

    def test_unsigned_future_offer_cannot_eliminate_strict_fallback(self):
        c=context('private',[action('alone',effect()),action('work',effect())],mode='principle',values={'benevolence':.9})
        options=[action('outside',effect(.5)),action('ab',effect(.2,values={'benevolence':.8}))]
        ranked,a=evaluate(c,options,{'outside':'alone','ab':'work'},('outside',),policy=Policy())
        self.assertEqual(a['outside_root'],'alone');self.assertEqual(ranked,('ab',))

    def test_equivalent_noop_is_not_a_reason_to_form_agreement(self):
        c=context('private',[action('alone',effect()),action('work',effect())])
        ranked,_=evaluate(c,[action('outside',effect(.2)),action('ab',effect(.2))],
            {'outside':'alone','ab':'work'},('outside',))
        self.assertFalse(ranked)

    def test_conflicting_groups_never_double_book_a_person(self):
        offers,legal=self.fixture()
        bs=[Ballot('a',SCOPE,'alone',('ab','ac')),Ballot('b',SCOPE,'alone',('ab',)),Ballot('c',SCOPE,'alone',('ac',))]
        selected,a=negotiate(offers,bs,tuple('abc'),scope=SCOPE,legal=legal)
        self.assertEqual(len(a['agreements']),1);self.assertEqual(selected['c'],'alone')
        selected,b=negotiate(offers,bs,tuple('cab'),scope=SCOPE,legal=legal)
        self.assertEqual(b['agreements'][0]['offer'],'ac');self.assertEqual(selected['b'],'alone')

    def test_changed_circumstances_require_new_signatures(self):
        offers,legal=self.fixture()
        bs=[Ballot('a',SCOPE,'alone',('ab',)),Ballot('b',SCOPE,'alone',('ab',)),Ballot('c',SCOPE,'alone',())]
        for scope in [('game','episode',8,'public-revision'),('game','episode',7,'different')]:
            with self.assertRaises(ValueError):negotiate(offers,bs,tuple('abc'),scope=scope,legal=legal)
        withdrawn=[bs[0],replace(bs[1],ranked=()),bs[2]]
        selected,a=negotiate(offers,withdrawn,tuple('abc'),scope=SCOPE,legal=legal)
        self.assertFalse(a['agreements']);self.assertEqual(selected['a'],'alone')

    def test_unavailable_roles_duplicate_owners_and_forged_membership_rejected(self):
        with self.assertRaises(ValueError):Offer('bad',(('a','work'),('a','alone')))
        offers,legal=self.fixture()
        bs=[Ballot('a',SCOPE,'alone',('ab',)),Ballot('b',SCOPE,'alone',('ac',)),Ballot('c',SCOPE,'alone',())]
        with self.assertRaises(ValueError):negotiate(offers,bs,tuple('abc'),scope=SCOPE,legal=legal)
        bs[1]=replace(bs[1],ranked=('ab',));legal['b']={'alone'}
        with self.assertRaises(ValueError):negotiate(offers,bs,tuple('abc'),scope=SCOPE,legal=legal)

    def test_three_person_proposal_uses_same_protocol(self):
        offer=Offer('abc',tuple((a,'different-role-'+a) for a in 'abc'))
        bs=[Ballot(a,SCOPE,'alone',('abc',)) for a in 'abc']
        legal={a:{'alone','different-role-'+a} for a in 'abc'}
        selected,a=negotiate([offer],bs,tuple('abc'),scope=SCOPE,legal=legal)
        self.assertEqual(selected,dict(offer.roots));self.assertEqual(len(a['agreements']),1)

    def test_unfinished_method_can_complete_before_new_partner_proposals(self):
        offers,legal=self.fixture()
        bs=[Ballot('a',SCOPE,'alone',('ac','ab')),Ballot('b',SCOPE,'alone',('ab',)),Ballot('c',SCOPE,'alone',('ac',))]
        chosen,a=negotiate(offers,bs,tuple('cab'),scope=SCOPE,legal=legal,continuing=('ab',))
        self.assertEqual(chosen,{'a':'work','b':'work','c':'alone'})
        self.assertTrue(a['renewals'][0]['renewed'])

    def test_ongoing_method_cannot_override_withdrawal_or_new_legality(self):
        offers,legal=self.fixture()
        bs=[Ballot('a',SCOPE,'alone',('ac',)),Ballot('b',SCOPE,'alone',('ab',)),Ballot('c',SCOPE,'alone',('ac',))]
        chosen,a=negotiate(offers,bs,tuple('abc'),scope=SCOPE,legal=legal,continuing=('ab',))
        self.assertEqual(chosen['b'],'alone');self.assertFalse(a['renewals'][0]['renewed'])
        self.assertEqual(a['agreements'][0]['offer'],'ac')
        chosen,a=negotiate([offers[1]],[bs[0],replace(bs[1],ranked=()),bs[2]],tuple('abc'),scope=SCOPE,legal=legal,continuing=('ab',))
        self.assertEqual(a['renewals'][0]['reason'],'opportunity unavailable')


if __name__=='__main__':unittest.main()
