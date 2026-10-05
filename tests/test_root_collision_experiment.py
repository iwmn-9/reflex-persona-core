"""Small regression checks for ablation identity, denominators and causality."""
import copy
import unittest
from reflex.combat import Battle,battle_record
from reflex.root_collision_experiment import jobs,aggregate,paired_divergence


def row(variant):
    return dict(scenario='open/secure',profile='care',seed=392,rival='reference',variant=variant,
        won=False,lost=False,win_credit=0.,ticks=1,decisions=3,planned_ticks=0,planning_declines=1,
        final_eliminate_progress=0.,final_secure_progress=0.,movement={},total_work={},
        completion_max_persona_regret=0.,longest_observed_stall=1)


class RootCollisionExperimentTests(unittest.TestCase):
    def test_frozen_matrix_is_paired_complete_and_uses_new_seeds(self):
        work=jobs()
        self.assertEqual(len(work),64);self.assertEqual(len(set(work)),64)
        self.assertEqual({j[4] for j in work},{391,392})
        self.assertEqual({j[3] for j in work},{'growth','steady','care','ego'})
        self.assertEqual({j[1] for j in work},{'secure','eliminate','either','both'})
        for i in range(0,len(work),2):
            self.assertEqual(work[i][:-1],work[i+1][:-1])
            self.assertEqual((work[i][-1],work[i+1][-1]),('baseline','root-completion'))

    def test_zero_counter_fields_are_optional_without_dropping_games(self):
        summary,pairs=aggregate([row('baseline'),row('root-completion')])
        self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0]['friendly_collision_delta'],0)
        self.assertEqual(pairs[0]['unresolved_stop_delta'],0)
        for r in summary:
            self.assertEqual(r['games'],1);self.assertIsNone(r['failed_move_rate'])

    def test_rates_keep_actual_attempt_denominators(self):
        a,b=row('baseline'),row('root-completion')
        a['movement']=dict(move_attempts=10,failed_moves=2,friendly_collision_moves=2)
        b['movement']=dict(move_attempts=20,failed_moves=2,friendly_collision_moves=1)
        summary,pairs=aggregate([a,b]);summary=[x for x in summary if x['dimension']=='all']
        self.assertEqual(summary[0]['failed_move_rate'],.2)
        self.assertEqual(summary[1]['failed_move_rate'],.1)
        self.assertEqual(summary[1]['friendly_collision_rate'],.05)
        self.assertEqual(pairs[0]['failed_move_delta'],0)

    def test_identical_trajectories_do_not_claim_completion_improvement(self):
        a=row('baseline');before=battle_record(Battle.start('open','secure'))
        a['trace']=[dict(before=before,after=before,choices={0:'guard',1:'guard',2:'guard'})]
        b=copy.deepcopy(a)
        self.assertIsNone(paired_divergence(a,b)['tick'])

    def test_first_changed_root_requires_matching_completion_on_same_state(self):
        a=row('baseline');before=battle_record(Battle.start('open','secure'))
        a['trace']=[dict(before=before,after=before,choices={0:'move:1:1',1:'move:1:1',2:'guard'})]
        b=copy.deepcopy(a);b['trace'][0]['choices'][1]='guard'
        b['trace'][0]['planning']=dict(root_completion=dict(adopted=True,
            before=['move:1:1','move:1:1','guard'],after=['move:1:1','guard','guard']))
        self.assertTrue(paired_divergence(a,b)['completion_caused_first_difference'])
        b['trace'][0]['planning']['root_completion']['adopted']=False
        with self.assertRaises(AssertionError):paired_divergence(a,b)

    def test_nonchoice_world_divergence_is_rejected(self):
        a=row('baseline');before=battle_record(Battle.start('open','secure'))
        a['trace']=[dict(before=before,after=before,choices={0:'guard',1:'guard',2:'guard'})]
        b=copy.deepcopy(a);b['trace'][0]['after']['tick']=1
        with self.assertRaises(AssertionError):paired_divergence(a,b)


if __name__=='__main__':unittest.main()
