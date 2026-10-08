import copy
from dataclasses import replace
import unittest
from reflex.examples import context,action,effect
from reflex.core import digest
from reflex.deliberation import JointForecast,select
from reflex.plan_continuity import ArrivalPreference,PlanIntention


def fixture():
    c=context('intention',[action('work',effect()),action('delay',effect())])
    fc=copy.deepcopy(c);fc['actions']=[action('old',effect(.3)),action('new',effect(.8))]
    f=JointForecast((fc,),{'old':('work',),'new':('delay',)},{'old':1.,'new':1.},6,
        continuity=ArrivalPreference('old',{'old':(2,3),'new':(3,4)}))
    return c,f


class PlanContinuityTests(unittest.TestCase):
    def test_same_completion_cannot_be_postponed_in_all_branches(self):
        c,f=fixture();chosen,audit=select([c],f)
        self.assertEqual(chosen[0]['action_id'],'work');self.assertEqual(audit['continuity']['excluded'],['new'])
        self.assertEqual(select([c],replace(f,continuity=None))[0][0]['action_id'],'delay')

    def test_earlier_mixed_or_unknown_results_do_not_force_commitment(self):
        c,f=fixture()
        for arrivals in ({'old':(3,4),'new':(2,3)}, {'old':(2,4),'new':(3,3)},
                         {'old':(None,3),'new':(4,4)}, {'old':(2,3),'new':(None,4)}):
            ff=replace(f,continuity=ArrivalPreference('old',arrivals))
            self.assertEqual(select([c],ff)[0][0]['action_id'],'delay')

    def test_purpose_gain_and_current_progress_support_override_stability(self):
        c,f=fixture()
        ff=replace(f,purpose={'old':.9,'new':1.})
        self.assertEqual(select([c],ff)[0][0]['action_id'],'delay')
        self.assertEqual(select([c],f,root_allowed=[{'delay'}])[0][0]['action_id'],'delay')

    def test_current_strongest_principle_is_not_overridden(self):
        c,f=fixture();c['values']['benevolence']=1.;c['state'].update(mode='principle',primary_need='physiology',mode_urgency=.1)
        fc=copy.deepcopy(c);fc['actions']=[action('old',effect(.3)),action('new',effect(.8,values={'benevolence':1.}))]
        ff=replace(f,contexts=(fc,))
        self.assertEqual(select([c],ff)[0][0]['action_id'],'delay')

    def test_branch_alignment_and_owner_contract_are_required(self):
        c,f=fixture()
        for bad in (ArrivalPreference('missing',{'old':(1,),'new':(2,)}),ArrivalPreference('old',{'old':(1,),'new':(2,3)}),
                    ArrivalPreference('old',{'old':(True,),'new':(2,)})):
            with self.assertRaises(ValueError):select([c],replace(f,continuity=bad))
        memory=PlanIntention(c);other=copy.deepcopy(c);other['scope']['npc']='another'
        with self.assertRaises(ValueError):memory.offer(other,target='goal',unit='turns',horizon=6)

    def test_intention_uses_only_common_prefix_and_survives_checkpoint(self):
        c,f=fixture();memory=PlanIntention(c)
        f=replace(f,audit=dict(unit='turns',plans={'old':{'proposal':'prior'}},continuation_search={'proposals':{
            'prior':{'branches':[{'actions':['work','prepare','left']},{'actions':['work','prepare','right']}]}}}))
        before=digest(memory.record());self.assertEqual(memory.offer(c,target=f.target,unit='turns',horizon=6),())
        self.assertEqual(digest(memory.record()),before)
        memory.remember(c,f,'old','work');self.assertEqual(memory.remaining,('prepare',))
        restored=PlanIntention.from_record(c,memory.record());later=copy.deepcopy(c);later['tick']=1
        self.assertEqual(restored.offer(later,target=f.target,unit='turns',horizon=6),('prepare',))
        self.assertEqual(restored.offer(later,target='changed',unit='turns',horizon=6),())
        with self.assertRaises(ValueError):restored.offer(c,target=f.target,unit='turns',horizon=6)
        saved=restored.record()
        with self.assertRaises(ValueError):restored.remember(later,f,'old','delay')
        self.assertEqual(restored.record(),saved)


if __name__=='__main__':unittest.main()
