"""Public-API parity, serialized resumption, and closed-loop game equivalence.

This measures translation fidelity, NOT improved intelligence or speed. The
original long-horizon cohorts are independent and remain Python experiments.
"""
import argparse
import copy
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from reflex.core import Policy, compile_batch, character_key, TRAITS, NEEDS, VALUES
from reflex.examples import action, context, effect, families
from reflex.laboratory import PROFILES
from tools.asymmetric_games import GAMES, CASES, AsymmetricGame, board
from tools.continuing_games import ContinuingGame
from tools.verify_rust_policy_probe import random_context


def encoded(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def request(c, priority, rigidity=0., residual=None, stochastic=True, checkpoint=None):
    policy = dict(priority=dict(kind=priority), residual=[0.]*8 if residual is None else list(residual))
    if priority == 'personal':
        policy['priority']['rigidity'] = rigidity
    config = dict(scope=c['scope'], seed=c['seed'], personality=[c['personality'][k] for k in TRAITS],
                  values=[c['values'][k] for k in VALUES], objective=c['objective'], policy=policy)
    actions = copy.deepcopy(c['actions'])
    for a in actions:
        for e in a['outcomes']:
            e['needs'] = [e['needs'].get(k, 0.) for k in NEEDS]
            e['values'] = [e['values'].get(k, 0.) for k in VALUES]
            e['style'] = [e['style'].get(k, 0.) for k in TRAITS]
    observation = dict(tick=c['tick'], needs=[c['needs'][k] for k in NEEDS],
                       facts=c['facts'], actions=actions, opponent=c['opponent'])
    return dict(config=config, state=c['state'] if checkpoint is None else None,
                checkpoint=checkpoint, observation=observation, stochastic=stochastic)


class Driver:
    def __init__(self, binary):
        self.binary = binary
        self.input_hash = hashlib.sha256()
        self.count = 0
        self.last = None
        self.start()

    def start(self):
        self.process = subprocess.Popen([str(self.binary)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf8', bufsize=1)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=30)
        assert self.process.returncode == 0, self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()

    def restart(self):
        self.close()
        self.start()

    def check(self, c, priority, rigidity=0., residual=None, stochastic=True, checkpoint=None):
        b = compile_batch([c])
        policy = Policy(residual, principle_priority=priority,
            **(dict(commitments={character_key(c): rigidity}) if priority == 'personal' else {}))
        expected = policy.decide(b, stochastic)
        line = encoded(request(c, priority, rigidity, residual, stochastic, checkpoint))
        self.input_hash.update(line+b'\n')
        self.process.stdin.write(line.decode()+'\n')
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        self.last = dict(context=c, request=json.loads(line), response=response,
            expected=expected.records(b)[0], expected_scores=expected.scores[0].tolist(),
            expected_features=expected.features[0].tolist(), expected_rng=b.rng[0].tolist())
        assert 'error' not in response, response
        got = response['decision']
        record = expected.records(b)[0]
        for key in ('action_id', 'target', 'reading_used', 'request_reading', 'next_state'):
            assert got[key] == record[key], (self.count, key, got[key], record[key])
        assert got['rng'] == b.rng[0].tolist()
        assert [a['id'] for a in got['scores']] == list(b.ids[0])
        assert [a['eligible'] for a in got['scores']] == expected.eligible[0].tolist()
        scores = np.array([a['score'] for a in got['scores']])
        features = np.array([a['features'] for a in got['scores']])
        err = max(float(np.max(abs(scores-expected.scores[0]))),
                  float(np.max(abs(features-expected.features[0]))))
        assert err <= 2e-12, (self.count, err)
        self.count += 1
        return response, err


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    sources = [*sorted((R/'rust/reflex-immediate').rglob('*.rs')),
               R/'rust/reflex-immediate/tests/data/reduction.json',
               R/'rust/Cargo.toml', R/'rust/Cargo.lock', R/'rust/reflex-immediate/Cargo.toml',
               R/'tools/verify_rust_immediate.py', R/'tools/verify_rust_policy_probe.py',
               R/'reflex/core.py', R/'reflex/examples.py', R/'reflex/laboratory.py',
               R/'tools/continuing_games.py', R/'tools/asymmetric_games.py']
    manifest = {f.relative_to(R).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    registration = dict(source=manifest, world_seeds=[90401, 90402], turns=128,
        families=list(GAMES), profiles=len(PROFILES), priorities=['lexicographic', 'finite', 'personal'],
        stochastic_by_seed=[False, True], rigidity=.5,
        acceptance='All categorical/state/RNG outputs exact; numeric error <=2e-12; every game transition and final world identical; no dropped conditions.',
        scope='Immediate base chooser only. Not learning/search integration or intelligence improvement.',
        numpy=np.__version__, python=sys.version, platform=sys.platform)
    (out/'registration.json').write_bytes(encoded(registration)+b'\n')
    subprocess.run(['cargo', 'build', '--release', '--locked', '--manifest-path', str(R/'rust/Cargo.toml'),
                    '--example', 'verify'], cwd=R, check=True)
    binary = R/'rust/target/release/examples'/('verify.exe' if sys.platform == 'win32' else 'verify')
    driver = Driver(binary)
    worst = 0.
    trajectories = []
    current = dict(phase='corpus')
    trace = []
    try:
        rng = np.random.default_rng(20261015)
        cs = [copy.deepcopy(r['context']) for r in families()]
        for game in GAMES:
            for case in CASES:
                for profile in PROFILES:
                    g = AsymmetricGame(game, profile, board(game, case))
                    cs.append(g.observe(g.start()))
        cs += [random_context(rng, i) for i in range(256)]
        cs += [random_context(rng, 10000+i, count=n) for i,n in enumerate((1, 2, 32, 128, 256))]
        cs.append(context('ties', [action('a', effect(.3)), action('b', effect(.3))], needs={n:0 for n in NEEDS}))
        for i, c in enumerate(cs):
            for priority, rigidity in [('lexicographic', 0), ('finite', 0), ('personal', 0), ('personal', .5), ('personal', 1)]:
                for stochastic in (False, True):
                    residual = np.zeros(8) if i % 2 else rng.uniform(-.08, .08, 8)
                    _, err = driver.check(c, priority, rigidity, residual, stochastic)
                    worst = max(worst, err)
        for c0 in cs[:16]:
            c = copy.deepcopy(c0)
            c['scope']['episode'] += '-sequence'
            saved = None
            for tick in range(64):
                c['tick'] = tick
                row, err = driver.check(c, 'finite', stochastic=True, checkpoint=saved)
                c['state'] = row['decision']['next_state']
                saved = row['checkpoint']
                worst = max(worst, err)
        corpus_count = driver.count
        assert corpus_count == 5124
        for game in GAMES:
            for profile_index, profile in enumerate(PROFILES):
                for priority in registration['priorities']:
                    for seed, stochastic in zip(registration['world_seeds'], registration['stochastic_by_seed']):
                        g = ContinuingGame(game, profile, turns=128)
                        python_world, rust_world = g.start(), g.start()
                        python_state, saved = None, None
                        world_rng = np.random.default_rng(seed)
                        trace = []
                        current = dict(phase='continuation', game=game, profile=profile_index,
                                       priority=priority, seed=seed, stochastic=stochastic)
                        while not g.terminal(python_world):
                            c = g.context(python_world, python_state)
                            assert c == g.context(rust_world, python_state)
                            if c['tick'] == 64:
                                driver.restart()  # saved state survives process destruction, not just a clone
                            row, err = driver.check(c, priority, .5, stochastic=stochastic, checkpoint=saved)
                            worst = max(worst, err)
                            py_policy = Policy(principle_priority=priority,
                                **(dict(commitments={character_key(c): .5}) if priority == 'personal' else {}))
                            py_choice = py_policy.choose(c, stochastic)
                            # Response stream is evaluator-owned; not supplied to either chooser.
                            likely = (c['tick']//32) % 3
                            z = str(world_rng.choice(3, p=[.8 if i == likely else .1 for i in range(3)]))
                            python_world, _ = g.transition(python_world, py_choice['action_id'], z)
                            rust_world, _ = g.transition(rust_world, row['decision']['action_id'], z)
                            assert python_world == rust_world
                            python_state, saved = py_choice['next_state'], row['checkpoint']
                            trace.append(dict(tick=c['tick'], action=py_choice['action_id'], response=z,
                                world_sha256=hashlib.sha256(encoded(python_world)).hexdigest(),
                                state=python_state))
                        trajectories.append(dict(game=game, profile=profile_index, priority=priority,
                            seed=seed, stochastic=stochastic, executed=len(trace), final=python_world,
                            checkpoint=json.loads(saved), trace=trace))
                        with gzip.open(out/'runs.jsonl.gz', 'ab') as f:
                            f.write(encoded(trajectories[-1])+b'\n')
            print(json.dumps(dict(family=game, completed_runs=len(trajectories), decisions=driver.count)), flush=True)
        assert len(trajectories) == 96
        for name, expected in manifest.items():
            assert hashlib.sha256((R/name).read_bytes()).hexdigest() == expected, 'source changed during verification'
        result = dict(complete=True, corpus_decisions=corpus_count, continuous_runs=len(trajectories),
            continuous_decisions=driver.count-corpus_count, total_decisions=driver.count,
            exact_actions_states_rng=True, identical_physical_worlds=True,
            max_numeric_absolute_error=worst, input_sha256=driver.input_hash.hexdigest(),
            runs_sha256=hashlib.sha256((out/'runs.jsonl.gz').read_bytes()).hexdigest(),
            registration_sha256=hashlib.sha256((out/'registration.json').read_bytes()).hexdigest(),
            compiler=subprocess.check_output(['rustc', '--version'], text=True).strip(),
            platform=sys.platform, scope=registration['scope'], performance_measured=False)
        (out/'result.json').write_bytes(encoded(result)+b'\n')
        print(json.dumps(result, indent=2))
    except Exception as exc:
        (out/'failure.json').write_bytes(encoded(dict(location=current, case=driver.last,
            partial_trace=trace, error=repr(exc)))+b'\n')
        (out/'result.json').write_bytes(encoded(dict(complete=False,
            passed_decisions=driver.count, completed_continuous_runs=len(trajectories),
            failure='failure.json', max_numeric_absolute_error_before_failure=worst,
            registration_sha256=hashlib.sha256((out/'registration.json').read_bytes()).hexdigest()))+b'\n')
        raise
    finally:
        driver.close()


if __name__ == '__main__':
    main()
