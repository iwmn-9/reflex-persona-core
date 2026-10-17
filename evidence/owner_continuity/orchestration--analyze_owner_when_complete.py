"""Analyze the full frozen registration once collected; never adopts or publishes."""
import json
import subprocess
import sys
import time
from pathlib import Path

V = Path(__file__).resolve().parent
R = V.parents[1] / 'github-preview/reflex-persona-core'
original = R / 'results/owner_recovery_confirmation'
extra = R / 'results/owner_periodic_eight'
deadline = time.monotonic() + 20 * 3600
while True:
    states = [json.loads(p.read_text()) if p.is_file() else None
              for p in (original / 'workflow.json', extra / 'workflow.json')]
    if any(s is not None and s['conclusion'] != 'success' for s in states):
        raise RuntimeError('Original workflow incomplete/failed: preserve evidence and arrange explicit completion before analysis.')
    if all(s is not None for s in states):
        break
    if time.monotonic() >= deadline:
        raise TimeoutError('Waiting for original registered workflows')
    time.sleep(45)

def run(label, arguments):
    with (extra / (label + '.log')).open('w', encoding='utf8') as output:
        subprocess.run([sys.executable, *map(str, arguments)], cwd=R,
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'analysis_complete': label}), flush=True)

run('full_confirmation_summary', [V / 'summarize_owner_confirmation.py'])
run('full_all_cadences_summary', [V / 'summarize_all_cadences.py'])
plan = json.loads((R / 'results/owner_recovery_confirmation_PLAN.json').read_text())
primary = [original / f'owner-recovery-confirm-{w["game"]}-{w["seed"]}-{group}' / part
           for w in plan['worlds'] for group in plan['profile_groups']
           for part in ('candidate', 'baseline')]
primary += [extra / f'owner-periodic-eight-{w["game"]}-{w["seed"]}-{group}' / 'baseline'
            for w in plan['worlds'] for group in plan['profile_groups']]
for kind in ('phases', 'behaviour', 'increments'):
    run('primary_complete_' + kind,
        [R / 'results' / ('review_owner_' + kind + '.py'), '--studies', *primary,
         '--output', extra / ('primary_complete_' + kind + '.json')])
auction_sources = [p / 'trajectories.jsonl' for p in primary if '-auction-' in p.parent.name]
assert len(auction_sources) == 48
run('primary_auction_bellman',
    [R / 'results/review_owner_bellman_cached.py', '--sources', *auction_sources,
     '--cache', extra / 'bellman_cache', '--workers', '4',
     '--output', extra / 'primary_auction_bellman.json'])
run('primary_auction_return_accounting',
    [R / 'results/review_auction_return_accounting.py', '--studies',
     *[p for p in primary if '-auction-' in p.parent.name],
     '--output', extra / 'primary_auction_return_accounting.json'])
print('All registered comparison analysis completed. Adoption and publication still require review.', flush=True)
