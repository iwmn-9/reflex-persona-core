import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import NEEDS, Policy, compile_batch
from reflex.cross_games import GAMES, Navigation, Duel, Project, core_hashes, experiment, run, signature
from reflex.laboratory import profiles


class CrossGameTests(unittest.TestCase):
    def test_mixed_and_separate_population_trajectories_match(self):
        for variant in (0,1):
            a = run(2,variant,30,'mixed')
            b = run(2,variant,30,'separate')
            self.assertEqual(signature(a),signature(b))

    def test_population_order_cannot_change_other_games_or_npcs(self):
        self.assertEqual(signature(run(1,0,20)),signature(run(1,0,20,order=list(reversed(range(12))))))

    def test_fixed_personality_and_values_are_identical_across_games_and_turns(self):
        traces = run(1,1,20)['traces']
        for p in profiles():
            rows = [r['context'] for r in traces if r['npc'] == p['id']]
            self.assertEqual(len({r['scope']['game'] for r in rows}),3)
            self.assertTrue(all(r['personality'] == rows[0]['personality'] and r['values'] == rows[0]['values'] for r in rows))

    def test_unsupported_need_is_absent_not_a_fabricated_reward(self):
        for cls in GAMES:
            c = cls(profiles()[0],0).observe()
            b = compile_batch([c])
            for j,n in enumerate(NEEDS):
                self.assertEqual(c['needs'][n]['supported'],n in cls.supported)
                if n not in cls.supported:
                    self.assertIsNone(c['needs'][n]['deficit'])
                    self.assertEqual(b.needs[0,j],0)
                    self.assertTrue((b.effects[0,:,:,1+j] == 0).all())

    def test_snapshot_reproduces_actual_action_and_persistent_state(self):
        policy = Policy()
        for row in run(3,0,18)['traces']:
            chosen = policy.choose(row['context'])
            self.assertEqual(chosen['action_id'],row['action'])
            self.assertEqual(chosen['next_state'],row['decision']['next_state'])

    def test_predictions_match_real_physics_without_mutating_world(self):
        for cls in GAMES:
            game = cls(profiles()[0],0)
            for turn in range(4):
                c = game.observe(); old = copy.deepcopy(game.state)
                for a in c['actions']:
                    branches = game.branches(a['id'])
                    self.assertEqual(sum(p for p,_ in branches),1.)
                    for out,(p,after) in zip(a['outcomes'],branches):
                        for n in cls.supported:
                            self.assertAlmostEqual(out['needs'][n],game.needs(old)[n]-game.needs(after)[n])
                        self.assertEqual(out['p'],p)
                    self.assertEqual(game.state,old)
                key = next(a['id'] for a in c['actions'] if a['legal'] and a['id'] != 'WAIT')
                predicted = [after for _,after in game.branches(key)]
                game.advance(key)
                self.assertIn(game.state,predicted)

    def test_topology_energy_and_task_dependencies_are_enforced_by_world(self):
        nav = Navigation(profiles()[0],0)
        self.assertFalse(nav.legal('LEFT'))
        with self.assertRaises(ValueError): nav.advance('LEFT')
        nav.state['x'],nav.state['y'] = 2,1
        self.assertFalse(nav.legal('RIGHT'))
        nav.state['energy'] = .01
        self.assertFalse(nav.legal('DOWN'))
        self.assertTrue(nav.legal('REST'))
        project = Project(profiles()[0],0)
        self.assertFalse(project.legal('TASK_C'))
        with self.assertRaises(ValueError): project.advance('TASK_C')
        project.state.update(a=2,b=2)
        self.assertTrue(project.legal('TASK_C'))
        project.state['energy'] = .01
        self.assertFalse(project.legal('TASK_C'))

    def test_help_cannot_farm_value_after_real_requests_are_fulfilled(self):
        project = Project(profiles()[2],0)
        project.state['helped'] = project.requested_help
        self.assertFalse(project.legal('HELP'))
        with self.assertRaises(ValueError): project.advance('HELP')
        self.assertEqual(project.observe()['facts']['public_ally_remaining_requests'],'0')

    def test_hidden_opponent_behavior_cannot_leak_before_an_observation(self):
        a,b = Duel(profiles()[0],9,0),Duel(profiles()[0],9,1)
        self.assertEqual(a.observe(),b.observe())
        for tick in (0,11,12,20):
            a.tick = b.tick = tick
            self.assertEqual(a.observe(),b.observe())
        a.observations = ['FEINT']*6
        self.assertGreater(a.probabilities()[0],b.probabilities()[0])
        self.assertNotIn('future',a.observe()['facts'])

    def test_cyclic_duel_has_no_single_move_that_dominates_all_responses(self):
        game = Duel(profiles()[0],0)
        for key in game.moves:
            wins = [game.resolve(key,r)['opponent_hp'] < game.state['opponent_hp'] and game.resolve(key,r)['hp'] == game.state['hp'] for r in game.moves]
            self.assertEqual(sum(wins),1)

    def test_neutral_reference_reaches_exit_and_finishes_feasible_projects(self):
        for variant in (0,1):
            result = run(4,variant,30,neutral=True)
            for a in result['summary']:
                if a['game'] == Navigation.name:
                    self.assertTrue(a['done'],a)
                elif a['game'] == Project.name:
                    self.assertEqual(a['state']['c'],a['state']['required'][2],a)
                    self.assertGreaterEqual(a['state']['time_left'],0)

    def test_terminal_noops_cannot_create_additional_resources_or_progress(self):
        for cls in GAMES:
            game = cls(profiles()[0],0)
            if cls is Navigation:
                game.state.update(x=game.goal[0],y=game.goal[1])
                key = 'REST'
            elif cls is Duel:
                game.state['opponent_hp'] = 0; key = 'REST'
            else:
                game.state['time_left'] = 0; key = 'WAIT'
            old = copy.deepcopy(game.state)
            game.advance(key)
            self.assertEqual(game.state,old)

    def test_replay_and_new_seed_leave_core_unchanged(self):
        hashes = core_hashes()
        a = run(0,0,15)
        self.assertEqual(signature(a),signature(run(0,0,15)))
        self.assertNotEqual(signature(a),signature(run(1,0,15)))
        self.assertEqual(hashes,core_hashes())

    def test_export_records_integrations_without_promoting_reference_labels(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            report = experiment(tmp,seeds=1,turns=6)
            rows = [json.loads(line) for line in (Path(tmp)/'teacher_requests.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(report['core_unchanged'])
            self.assertEqual(report['mixed_separate_mismatches'],0)
            self.assertEqual({r['family'] for r in rows},{cls.name for cls in GAMES})
            self.assertTrue(all(r['quality'] == 'awaiting_generation' and 'preferred' not in r for r in rows))


if __name__ == '__main__': unittest.main()
