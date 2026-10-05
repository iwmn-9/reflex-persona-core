import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import compile_batch, digest, Policy
from reflex.laboratory import Laboratory, ACTIONS, INDEX, outcome, consequence, deficits, experiment


class LaboratoryTests(unittest.TestCase):
    def test_multiturn_is_replayable_and_profiles_are_stable(self):
        a=Laboratory(0); original=copy.deepcopy(a.roster); trace=a.run(20)
        self.assertEqual(trace,Laboratory(0).run(20)); self.assertEqual(a.roster,original)
        self.assertEqual(len(trace),80); self.assertEqual(a.illegal,0)
        self.assertTrue(any(trace[i]['after']!=trace[i]['before'] for i in range(len(trace))))

    def test_order_of_actors_does_not_change_world_or_choices(self):
        a=Laboratory(1,"contest"); b=Laboratory(1,"contest",order=[3,1,0,2])
        key=lambda x:(x['round'],x['npc'])
        self.assertEqual(sorted(a.run(20),key=key),sorted(b.run(20),key=key))

    def test_unobserved_hidden_failure_does_not_change_input_or_randomness(self):
        normal=Laboratory(0,"ordinary"); hidden=Laboratory(0,"workshop")
        self.assertEqual(normal.run(12,True),hidden.run(12,True))

    def test_workshop_cause_is_private_and_understood_failure_is_not_repeated(self):
        lab=Laboratory(0,"workshop",neutral=True); trace=lab.run(40,True)
        self.assertGreater(sum(lab.learned_at>=0),0)
        self.assertEqual(lab.failure_after_known,0)
        for i,actor in enumerate(lab.roster):
            t=lab.learned_at[i]
            if t>=0:
                rows=[r for r in trace if r['npc']==actor['id']]
                self.assertFalse(rows[t]['context']['actions'][INDEX['WORK_RISK']]['known_failure'])
                if t+1<len(rows): self.assertTrue(rows[t+1]['context']['actions'][INDEX['WORK_RISK']]['known_failure'])
                self.assertTrue(all(r['action']!='WORK_RISK' for r in rows[t+1:]))
        separate=Laboratory(); separate.known_broken[0]=True
        flags=separate.arrays(np.full(4,.5))['known_failure']
        self.assertEqual(flags[:,INDEX['WORK_RISK']].tolist(),[True,False,False,False])

    def test_no_observations_cannot_supply_confident_reading(self):
        lab=Laboratory(1); data,probability,applied,_=lab.inputs()
        self.assertFalse(applied.any()); self.assertTrue((data['read_confidence']==0).all())
        self.assertTrue((probability==.5).all())

    def test_observed_reading_changes_means_without_changing_profile(self):
        lab=Laboratory(1); original=copy.deepcopy(lab.roster); traces=lab.run(40,True)
        changed=[r for r in traces if r['action_changed_by_reading']]
        self.assertTrue(changed)
        self.assertEqual(lab.roster,original)
        for row in changed:
            self.assertGreaterEqual(row['read_confidence'],.6)
            self.assertNotEqual(row['action'],row['reflex_action'])
            self.assertTrue(row['read_threatened'] or not row['read_ahead'])
            self.assertEqual(Policy().choose(row['context'])['action_id'],row['action'])

    def test_predictions_and_actual_rule_use_same_own_state_transition(self):
        s=dict(energy=.8,stock=1.4,skill=.1,hunger=.1,social=.4,esteem=.45,growth=.55)
        before=deficits(*(s[k] for k in ('energy','stock','hunger','social','esteem','growth')))
        for key in ACTIONS:
            for success in (False,True):
                for rival in (False,True):
                    after=outcome(key,**s,success=success,rival_claim=rival)
                    need_after=deficits(*(after[k] for k in ('energy','stock','hunger','social','esteem','growth')))
                    self.assertTrue(np.allclose(consequence(key,s,success,rival)[1:6],before-need_after))

    def test_captured_context_reproduces_numeric_choice_and_contains_no_hidden_world(self):
        lab=Laboratory(0); traces=lab.run(12,True)
        for row in traces:
            c=row['context']; compile_batch([c])
            self.assertEqual(Policy().choose(c)['action_id'],row['action'])
            self.assertNotIn('break_at',json.dumps(c))
            self.assertNotIn('secret_strategy',json.dumps(c))

    def test_seed_changes_experience_without_changing_profile(self):
        a=Laboratory(0); b=Laboratory(1)
        self.assertEqual(a.roster,b.roster); self.assertNotEqual(a.run(15),b.run(15))

    def test_episode_export_never_labels_reference_as_approved_teacher(self):
        with tempfile.TemporaryDirectory() as td:
            r=experiment(td,turns=3,seeds=1)
            requests=[json.loads(line) for line in (Path(td)/'teacher_requests.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(requests); self.assertEqual(r['checks']['illegal'],0)
            self.assertTrue(all(x['quality']=='awaiting_generation' and 'preferred' not in x for x in requests))
            self.assertEqual(len({x['context_hash'] for x in requests}),len(requests))
            for x in requests: self.assertEqual(x['context_hash'],digest(x['context']))


if __name__=='__main__': unittest.main()
