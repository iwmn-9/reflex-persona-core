"""Publish compact, lossless-for-listed-metrics views of the frozen analysis.

Full per-event probabilities and per-game calibration histograms remain generated
artifacts. All partition/profile/scenario/rival and per-game loss totals remain.
"""
from pathlib import Path
import argparse
import copy
import json
import shutil


def summarize(source, target):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads((source/'evaluation.json').read_text())
    for group in evaluation['groups']:
        if group['dimension'] == 'all':
            continue
        for scores in group['metrics'].values():
            for metric in scores.values():
                metric.pop('bins', None)
                if 'calibration' in metric:
                    metric['calibration'].pop('bins', None)
    evaluation['publication_note'] = 'All aggregate losses retained; full bins retained for partition totals. Per-event probabilities and remaining bins are reproducible generated artifacts.'
    (target/'evaluation.json').write_text(json.dumps(evaluation, indent=2, allow_nan=False)+'\n')
    games = json.loads((source/'per_game.json').read_text())
    for game in games:
        for scores in game['metrics'].values():
            for metric in scores.values():
                metric.pop('bins', None)
                if 'calibration' in metric:
                    metric['calibration'].pop('bins', None)
    (target/'per_game.json').write_text(json.dumps(games, indent=2, allow_nan=False)+'\n')
    for name in ('preregister.json', 'verification.json'):
        shutil.copyfile(source/name, target/name)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='reflex_artifacts/observed_opponent')
    p.add_argument('--output', default='evidence/observed_opponent')
    args = p.parse_args()
    summarize(args.input, args.output)
