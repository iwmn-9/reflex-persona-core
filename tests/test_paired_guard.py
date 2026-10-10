import copy,json,unittest
from dataclasses import replace
from reflex.core import Policy
from reflex.examples import context,action,effect
from reflex.decision_loop import DecisionLoop,Request
from reflex.judgment import Binding
from reflex.planning import vector
from reflex.paired_value import fit
from reflex.paired_guard import ValueProposal,constrain


class PairedGuardTests(unittest.TestCase):
    def fixture(self,npc='owner'):
        c=context('paired-loop',[action('A',effect(.4)),action('B',effect(.3))]);c['scope']['npc']=npc
        model=fit([([[s,0],[s,1]],[[.1],[.9]]) for s in (-1.,-.5,.5,1.)],alpha=.01)
        proposal=ValueProposal(model,[.01],{'A':[.5,0],'B':[.5,1]},'public-opportunity-v1','own-terminal-success')
        req=Request(c,{k:Binding(k,'condition',('objective',)) for k in ('A','B')},{k:() for k in ('A','B')},value_proposal=proposal)
        return c,req

    def test_actual_commit_and_feedback_keep_immediate_target(self):
        c,req=self.fixture();loop=DecisionLoop(c);before=copy.deepcopy(req.value_proposal)
        result=loop.decide(req,False)
        self.assertEqual(result['decision']['action_id'],'B');self.assertTrue(result['value_guard']['changed'])
        self.assertEqual(loop.pending['prior'][0]['objective'],.3)
        self.assertEqual(c['personality'],loop.personality);self.assertEqual(c['values'],loop.values)
        loop.observe(result['ticket'],vector(effect(-.2)))
        self.assertEqual(next(iter(loop.memory.entries.values()))['samples'][0][0],-.2)
        self.assertEqual(req.value_proposal,before);json.dumps(result)
        self.assertEqual(DecisionLoop.from_record(c,loop.record()).record(),loop.record())

    def test_no_proposal_and_failed_evidence_keep_original_decision_exact(self):
        c,req=self.fixture();baseline=DecisionLoop(c).decide(replace(req,value_proposal=None),False)
        for p in (replace(req.value_proposal,errors=[1.]),replace(req.value_proposal,features={'A':[100,0],'B':[100,1]})):
            r=DecisionLoop(c).decide(replace(req,value_proposal=p),False)
            del r['value_guard'];self.assertEqual(r,baseline)

    def test_existing_progress_exclusion_cannot_be_reopened(self):
        c,req=self.fixture();inc=Policy().choose(c,False)
        d,a=constrain(c,inc,req.value_proposal,Policy(),{'A'},req.exact)
        self.assertEqual(d,inc);self.assertFalse(a['changed']);self.assertIn('B',a['excluded'])

    def test_strongest_principle_is_not_overridden_by_learned_success(self):
        c,req=self.fixture();c['state']['mode']='principle';c['values']['security']=1.
        c['actions'][0]['outcomes'][0]['values']={'security':.8}
        c['actions'][1]['outcomes'][0]['values']={'security':-.8}
        r=DecisionLoop(c).decide(req,False)
        self.assertEqual(r['decision']['action_id'],'A');self.assertFalse(r['value_guard']['changed'])

    def test_nonviable_action_cannot_receive_a_teacher_override(self):
        for failure in (False,True):
            c,req=self.fixture();c['actions'][1]['known_failure']=failure;c['actions'][1]['legal']=failure
            with self.assertRaisesRegex(ValueError,'viable root'):DecisionLoop(c).decide(req,False)
            p=replace(req.value_proposal,features={'A':[.5,0]})
            r=DecisionLoop(c).decide(replace(req,value_proposal=p),False)
            self.assertEqual(r['decision']['action_id'],'A');self.assertFalse(r['value_guard']['changed'])

    def test_invalid_late_proposal_does_not_partially_commit_batch(self):
        c,a=self.fixture('first');d,b=self.fixture('second');la,lb=DecisionLoop(c),DecisionLoop(d)
        before=(la.record(),lb.record());bad=replace(b.value_proposal,features={'A':[.5,0],'B':[float('nan'),1]})
        with self.assertRaises(ValueError):DecisionLoop.decide_batch([(la,a),(lb,replace(b,value_proposal=bad))],False)
        self.assertEqual(before,(la.record(),lb.record()))

    def test_independent_owners_batch_equals_individual_and_reordered(self):
        pairs=[self.fixture(str(i)) for i in range(6)];loops=[DecisionLoop(c) for c,_ in pairs]
        rr=DecisionLoop.decide_batch([(loops[i],pairs[i][1]) for i in reversed(range(6))],False)
        for i,r in zip(reversed(range(6)),rr):self.assertEqual(r,DecisionLoop(pairs[i][0]).decide(pairs[i][1],False))


if __name__=='__main__':unittest.main()
