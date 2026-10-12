import copy,json,unittest
from reflex.core import NEEDS
from reflex.examples import context,action,effect
from reflex.public_responses import PublicResponses


def public(tick=0):
    c=context('public-peer',[action('share',effect(.1,values={'benevolence':1.})),action('take',effect(.3,values={'power':1.}))],
        needs={k:0. for k in NEEDS},mode='principle');c['scope']['npc']='peer';c['tick']=tick;return c


PROFILES=(dict(id='giving',traits=(.5,)*5,values={'benevolence':1.}),dict(id='taking',traits=(.5,)*5,values={'power':1.}))


class PublicResponseTests(unittest.TestCase):
    def test_actual_hidden_profile_fields_are_replaced_by_declared_hypotheses(self):
        c=public();bank=PublicResponses(c['scope'],PROFILES);a=bank.models(c)
        c['personality']={k:.99 for k in c['personality']};c['values']={k:float(k=='security') for k in c['values']}
        self.assertEqual(a,bank.models(c));self.assertEqual(a['giving']['share'],1.)

    def test_predictive_adoption_and_revocation_follow_public_changes(self):
        bank=PublicResponses(public()['scope'],PROFILES)
        for t in range(20):bank.observe('peer',public(t),'share',str(t))
        p,trial=bank.forecast('peer',public(20));self.assertTrue(trial['active']);self.assertGreater(p['share'],p['take'])
        for t in range(20,40):bank.observe('peer',public(t),'take',str(t))
        p,trial=bank.forecast('peer',public(40));self.assertGreater(p['take'],p['share'])

    def test_hypothetical_learning_is_detached_and_invalid_updates_are_atomic(self):
        bank=PublicResponses(public()['scope'],PROFILES);saved=bank.record();branch=copy.deepcopy(bank)
        branch.observe('peer',public(),'share','model-only');self.assertEqual(bank.record(),saved)
        with self.assertRaises(ValueError):bank.observe('peer',public(),'illegal','actual')
        self.assertEqual(bank.record(),saved)

    def test_duplicate_event_never_increases_evidence(self):
        bank=PublicResponses(public()['scope'],PROFILES);bank.observe('peer',public(),'share','same')
        saved=bank.record()
        with self.assertRaises(ValueError):bank.observe('peer',public(),'share','same')
        self.assertEqual(bank.record(),saved)

    def test_save_restore_preserves_predictions_and_rejects_other_owner(self):
        scope=public()['scope'];bank=PublicResponses(scope,PROFILES)
        for t in range(12):bank.observe('peer',public(t),'share',str(t))
        checkpoint=json.loads(json.dumps(bank.record()));restored=PublicResponses.from_record(scope,checkpoint)
        self.assertEqual(bank.record(),restored.record());self.assertEqual(bank.forecast('peer',public(12)),restored.forecast('peer',public(12)))
        scope=copy.deepcopy(scope);scope['npc']='other'
        with self.assertRaises(ValueError):PublicResponses.from_record(scope,checkpoint)


if __name__=='__main__':unittest.main()
