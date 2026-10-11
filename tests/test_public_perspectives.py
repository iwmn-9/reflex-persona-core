import copy
import unittest
from reflex.board_models import ThanksPosition
from reflex.public_perspectives import PublicPerspectives
from reflex.strong_search import PublicMemory,PERSONA
from reflex.strong_table import decide as incumbent
from reflex.thanks_policy_rollout import OwnerPolicyModel,Branch,decide
from reflex.laboratory import PROFILES
from reflex.policy_rollout import evaluate_policy
from reflex.monte_carlo import RolloutBudget


def bank(owner=0):
    return PublicPerspectives('no_thanks','series-123',owner,range(4),lambda a:PublicMemory('no_thanks',a))


class FailingObserver:
    def __init__(self,viewer):self.viewer=viewer;self.events=0
    def observe(self,*args):
        self.events+=1
        if self.viewer==2:raise RuntimeError('observer failed')
    def record(self):return dict(events=self.events)


class PublicPerspectiveTests(unittest.TestCase):
    def position(self):return ThanksPosition(((),)*4,(11,)*4,0,20,seen=(20,),remaining=2,payments=(0,)*4)

    def test_public_replay_matches_separately_constructed_observers(self):
        b=bank();separate={a:PublicMemory('no_thanks',a) for a in range(4)};s=self.position()
        for tick,action in enumerate(('PASS','PASS','TAKE')):
            actor=s.turn;event=f'public-{tick}'
            b.observe(s,actor,action,event,range(4))
            for viewer,m in separate.items():
                if viewer!=actor:m.observe(s,actor,action,event)
            s=s.play(action)
        snapshots=b.snapshot('no_thanks','series-123',0)
        self.assertEqual({a:m.record() for a,m in snapshots.items()},{a:m.record() for a,m in separate.items()})
        self.assertEqual(b.delivered,3)

    def test_unwitnessed_event_and_foreign_scope_cannot_supply_history(self):
        b=bank();before=b.receipt()
        with self.assertRaises(ValueError):b.observe(self.position(),1,'PASS','hidden',(1,2,3))
        self.assertEqual(b.receipt(),before)
        for args in (('goofspiel','series-123',0),('no_thanks','other-world',0),('no_thanks','series-123',1)):
            with self.assertRaises(ValueError):b.snapshot(*args)

    def test_failed_observer_rolls_back_every_hypothetical_view(self):
        b=PublicPerspectives('test','world',0,range(4),FailingObserver);before=b.receipt()
        with self.assertRaises(RuntimeError):b.observe(None,1,'move','one',range(4))
        self.assertEqual(b.receipt(),before)

    def test_branch_snapshot_is_detached_from_owned_public_models(self):
        b=bank();before=b.receipt();views=b.snapshot('no_thanks','series-123',0)
        views[1].observe(self.position(),0,'PASS','virtual')
        self.assertEqual(b.receipt(),before)

    def test_empty_public_history_preserves_the_existing_searched_samples(self):
        s=self.position();p=PROFILES[0];m=PublicMemory('no_thanks',0)
        _,d,_=incumbent('no_thanks',s,0,p,123,0,0,m,None,'adaptive',PERSONA,variant='certified_expiry')
        outputs=[]
        for kind in ('searched','public_history'):
            model=OwnerPolicyModel(s,0,p,m,None,d,123,0,0,'adaptive',rival_policy=kind,perspectives=bank())
            roots={a:Branch(s.play(a),a) for a in s.legal()}
            evaluate_policy(roots,model,RolloutBudget(samples=2,min_samples=2,max_nodes=10000),'empty-history')
            outputs.append(model.samples)
        self.assertEqual(outputs[0],outputs[1])

    def test_real_rollout_uses_public_history_without_changing_owner_or_bank(self):
        b=bank();s=self.position();s=s.play('PASS');b.observe(self.position(),0,'PASS','prior',range(4))
        p=PROFILES[1];m=PublicMemory('no_thanks',1);m.observe(self.position(),0,'PASS','prior')
        # This bank belongs to actor 1; rebuild from the same public event.
        b=bank(1);b.observe(self.position(),0,'PASS','prior',range(4));before=b.receipt();memory=copy.deepcopy(m.record())
        c,d,st=decide('no_thanks',s,1,p,123,0,1,m,None,'adaptive',PERSONA,
            rival_policy='public_history',perspectives=b,rollout=RolloutBudget(samples=2,min_samples=2,max_nodes=10000))
        self.assertTrue(st['policy_rollout']['used']);self.assertEqual(st['policy_rollout']['rival_history'],before)
        self.assertEqual(m.record(),memory);self.assertEqual(b.receipt(),before)
        from reflex.tabletop_trials import score
        c=copy.deepcopy(c);c['actions']=[a for a in c['actions'] if a['id'] in st['guard']['allowed']]
        self.assertEqual(score(c)[0],d)


if __name__=='__main__':unittest.main()
