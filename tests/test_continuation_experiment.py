"""Sparse diagnostic counters and deterministic postprocessing recovery."""
from collections import Counter
from contextlib import redirect_stdout
import copy
from io import StringIO
import json
from pathlib import Path
import runpy
import tempfile
import unittest

from reflex.combat import Battle, battle_record, resolve
from reflex.continuation_experiment import aggregate, jobs
from reflex.laboratory import profiles
from reflex.wait_experiment import diagnose


DIAGNOSTICS = ('expired_stop_selections', 'unresolved_stop_selections',
               'unsupported_root_selections', 'unsupported_stop_selections',
               'supported_stop_selections', 'future_gain_unproven_stops')


def row(variant, partition='heldout', recovery=True, **changes):
    result = dict(partition=partition, scenario='choke/eliminate', rival='switch',
                  profile='steady', seed=281 if partition == 'heldout' else 180,
                  recovery_options=recovery, variant=variant, won=True,
                  lost=False, win_credit=1., ticks=2, focal_failed_moves=0,
                  own_proposal_collisions=0,
                  alignment=dict(continuations={}),
                  total_work=dict(calls=2, model_transitions=24,
                                  recovery_calls=int(recovery)))
    result.update(changes)
    return result


def sparse_rows():
    return [row(v, partition=p, recovery=recovery)
            for p, recovery in (('heldout', True), ('known', False), ('known', True))
            for v in ('objective', 'persona-band')]


def dense_rows(rows):
    result = copy.deepcopy(rows)
    for r in result:
        for key in DIAGNOSTICS:
            r.setdefault(key, 0)
    return result


def canonical(result):
    summary, pairs = result
    return summary, sorted(pairs, key=lambda p: tuple(p['condition']))


class ContinuationExperimentTests(unittest.TestCase):
    def test_job_matrix_keeps_heldout_pairs_and_known_recovery_variants_distinct(self):
        work = jobs()
        self.assertEqual(len(work), 68)
        self.assertEqual(len(set(work)), 68)
        heldout = [j for j in work if j[-1] == 'heldout']
        known = [j for j in work if j[-1] == 'known']
        self.assertEqual(len(heldout), 64)
        self.assertEqual(len(known), 4)
        self.assertEqual({j[4] for j in heldout}, {281, 282})
        self.assertEqual({j[4] for j in known}, {180})
        self.assertTrue(all(j[6] for j in heldout))
        pairs = Counter(j[:5] + j[6:] for j in work)
        self.assertTrue(all(n == 2 for n in pairs.values()))
        self.assertEqual({(j[5], j[6]) for j in known},
                         {(v, recovery) for v in ('objective', 'persona-band') for recovery in (False, True)})

    def test_missing_zero_diagnostics_equal_explicit_zeros_without_mutation(self):
        sparse = sparse_rows()
        saved = copy.deepcopy(sparse)
        expected = aggregate(dense_rows(sparse))
        actual = aggregate(sparse)
        self.assertEqual(actual, expected)
        self.assertEqual(sparse, saved)
        for group in actual[0]:
            for key in ('expired_stop_selections', 'unresolved_stop_selections', 'unsupported_root_selections'):
                self.assertEqual(group[key], 0)
        self.assertTrue(all(p['expired_stop_delta'] == 0 for p in actual[1]))

    def test_one_sided_sparse_counts_keep_correct_signed_deltas(self):
        rows = sparse_rows()
        for r in rows:
            if r['partition'] == 'known' and not r['recovery_options']:
                if r['variant'] == 'objective':
                    r['expired_stop_selections'] = 3
                    r['focal_failed_moves'] = 1
                else:
                    r.update(won=False, lost=True, win_credit=0., focal_failed_moves=4)
            elif r['partition'] == 'known' and r['recovery_options']:
                if r['variant'] == 'persona-band':
                    r['expired_stop_selections'] = 5
                    r['unresolved_stop_selections'] = 2
                else:
                    r.update(won=False, lost=True, win_credit=0.)
        summary, pairs = aggregate(rows)
        known = {p['condition'][-1]: p for p in pairs if p['condition'][0] == 'known'}
        self.assertEqual(known[False]['expired_stop_delta'], -3)
        self.assertEqual(known[False]['failed_move_delta'], 3)
        self.assertEqual(known[False]['outcome_delta'], -1.)
        self.assertEqual(known[True]['expired_stop_delta'], 5)
        self.assertEqual(known[True]['outcome_delta'], 1.)
        totals = {(s['partition'], s['variant']): s for s in summary}
        self.assertEqual(totals['known', 'objective']['expired_stop_selections'], 3)
        self.assertEqual(totals['known', 'persona-band']['expired_stop_selections'], 5)
        self.assertEqual(totals['known', 'persona-band']['unresolved_stop_selections'], 2)

    def test_actual_no_stop_trace_produces_supported_sparse_diagnostics(self):
        before = Battle.start('open', 'eliminate', limit=1)
        choices = {i: f'move:{2 if i < 3 else 6}:{u.y}' for i, u in enumerate(before.units)}
        after, audit = resolve(before, choices, 42)
        recorded = dict(genre='combat', seed=281,
                        trace=[dict(before=battle_record(before), after=battle_record(after), choices=choices, audit=audit)])
        diagnostics = diagnose(recorded, profiles()[1])
        self.assertNotIn('expired_stop_selections', diagnostics)
        rows = sparse_rows()
        for r in rows:
            r.update(diagnostics)
        _, pairs = aggregate(rows)
        self.assertTrue(all(p['expired_stop_delta'] == 0 for p in pairs))

    def test_completed_raw_jsonl_reaggregates_independent_of_completion_order(self):
        rows = sparse_rows()
        rows[1]['expired_stop_selections'] = 2
        rows[3]['alignment']['continuations'] = dict(self_compared=4, self_differences=1)
        expected = canonical(aggregate(dense_rows(rows)))
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / 'trajectories.jsonl'
            raw.write_text(''.join(json.dumps(dict(r, trace=[])) + '\n' for r in reversed(rows)))
            recovered = [json.loads(line) for line in raw.read_text().splitlines()]
            self.assertEqual(canonical(aggregate(recovered)), expected)
        known = [p for p in expected[1] if p['condition'][0] == 'known']
        self.assertEqual({p['condition'][-1] for p in known}, {False, True})

    def test_summary_prints_sparse_run_diagnostics_as_zero(self):
        rows = sparse_rows()
        summary, pairs = aggregate(dense_rows(rows))
        report = dict(summary=summary, pairs=pairs, runs=rows,
                      rule_checks=6, purpose_checks=18, elapsed_seconds=1.)
        script = Path(__file__).resolve().parents[1] / 'tools' / 'summarize_continuation.py'
        main = runpy.run_path(str(script))['main']
        output = StringIO()
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'evaluation.json').write_text(json.dumps(report))
            with redirect_stdout(output):
                main(folder)
        text = output.getvalue()
        for variant in ('objective', 'persona-band'):
            line = next(line for line in text.splitlines() if line.startswith('steady ' + variant + ' '))
            self.assertIn('wins 1 / 1', line)
            self.assertIn('expired 0', line)
            self.assertIn('unresolved 0', line)
        self.assertIn('Checks 6 18', text)

    def test_work_and_alignment_sparse_counters_are_summed_not_inferred(self):
        rows = dense_rows(sparse_rows())
        rows[1]['alignment']['continuations'] = dict(self_compared=3, self_differences=2)
        rows[1]['total_work'].update(persona_batches=7, persona_cache_hits=4)
        summary, pairs = aggregate(rows)
        cheap = next(s for s in summary if (s['partition'], s['variant']) == ('heldout', 'persona-band'))
        self.assertEqual(cheap['alignment'], dict(self_compared=3, self_differences=2))
        self.assertEqual(cheap['work']['persona_batches'], 7)
        self.assertEqual(cheap['work']['persona_cache_hits'], 4)
        self.assertEqual(cheap['work']['recovery_calls'], 1)
        pair = next(p for p in pairs if p['condition'][0] == 'heldout')
        self.assertEqual((pair['self_compared_before'], pair['self_mismatch_before']), (0, 0))
        self.assertEqual((pair['self_compared_after'], pair['self_mismatch_after']), (3, 2))


if __name__ == '__main__':
    unittest.main()
