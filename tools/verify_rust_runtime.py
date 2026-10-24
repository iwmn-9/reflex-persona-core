"""Population/host integration fidelity, not a new intelligence experiment.

Four existing rule families, four actors with separate physical worlds batched
together, three priorities, deterministic/stochastic choices. Every world and
its RNG are saved with the population; process restart at tick64. Shared-world
collision and all-or-nothing host tests live in the Rust integration suite.
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
from reflex.core import Policy, compile_batch, character_key
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
from tools.asymmetric_games import GAMES
from tools.verify_rust_immediate import encoded, request


class Driver:
    def __init__(self, binary):
        self.binary = binary
        self.digest = hashlib.sha256()
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

    def call(self, **message):
        line = encoded(message)
        self.digest.update(line + b'\n')
        self.last = dict(request=message)
        self.process.stdin.write(line.decode() + '\n')
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        self.last['response'] = response
        assert 'error' not in response, response
        self.count += 1
        return response


def compare(c, priority, stochastic, actual):
    b = compile_batch([c])
    p = Policy(principle_priority=priority,
        **(dict(commitments={character_key(c): .5}) if priority == 'personal' else {}))
    expected = p.decide(b, stochastic)
    record = expected.records(b)[0]
    for key in ('action_id', 'target', 'reading_used', 'request_reading', 'next_state'):
        assert actual[key] == record[key], (key, actual[key], record[key])
    assert actual['rng'] == b.rng[0].tolist()
    assert [r['id'] for r in actual['scores']] == list(b.ids[0])
    assert [r['eligible'] for r in actual['scores']] == expected.eligible[0].tolist()
    error = max(float(np.max(abs(np.array([r['score'] for r in actual['scores']])-expected.scores[0]))),
                float(np.max(abs(np.array([r['features'] for r in actual['scores']])-expected.features[0]))))
    assert error <= 2e-12, error
    return record, error


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    out = Path(p.parse_args().output)
    out.mkdir(parents=True, exist_ok=False)
    sources = [*sorted((R/'rust').glob('reflex-*/**/*.rs')),
        *sorted((R/'rust').glob('reflex-*/Cargo.toml')), R/'rust/Cargo.toml', R/'rust/Cargo.lock',
        R/'tools/verify_rust_runtime.py', R/'tools/verify_rust_immediate.py', R/'reflex/core.py',
        R/'reflex/laboratory.py', R/'reflex/examples.py', R/'tools/continuing_games.py', R/'tools/asymmetric_games.py']
    manifest = {f.relative_to(R).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    registration = dict(source=manifest, families=list(GAMES), profiles=len(PROFILES), turns=128,
        priorities=['lexicographic','finite','personal'], seeds=[90411,90412], stochastic=[False,True],
        workers=[1,2], reverse_inputs_every_other_tick=True, process_restart_after_tick=63,
        scope='Runtime immediate integration equivalence. Four independent worlds per batch, not a multiplayer intelligence test.',
        acceptance='All96 trajectories complete, actions/state/RNG exact, max numeric difference<=2e-12, world+RNG save/restore exact; no dropped cases.',
        platform=sys.platform, numpy=np.__version__, python=sys.version)
    (out/'registration.json').write_bytes(encoded(registration)+b'\n')
    subprocess.run(['cargo','build','--release','--locked','--manifest-path',str(R/'rust/Cargo.toml'),
                    '--example','runtime_verify'],cwd=R,check=True)
    binary = R/'rust/target/release/examples'/('runtime_verify.exe' if sys.platform=='win32' else 'runtime_verify')
    d = Driver(binary)
    count = 0
    worst = 0.
    runs = 0
    restarts = 0
    checkpoints = 0
    current = {}
    trace = []
    try:
        for game in GAMES:
            for priority in registration['priorities']:
                for seed, stochastic in zip(registration['seeds'],registration['stochastic']):
                    current = dict(game=game,priority=priority,seed=seed)
                    session = dict(game=f'runtime-{game}',episode=f'{priority}-{seed}')
                    games = [ContinuingGame(game,profile,turns=128) for profile in PROFILES]
                    worlds = [g.start() for g in games]
                    rust_worlds = copy.deepcopy(worlds)
                    states = [None]*len(games)
                    rngs = [np.random.default_rng(seed+i*100) for i in range(len(games))]
                    rust_rngs = [np.random.default_rng(seed+i*100) for i in range(len(games))]
                    configs = []
                    for i,g in enumerate(games):
                        c = g.context(worlds[i],states[i])
                        c['scope'] = dict(**session,npc=f'npc-{i}')
                        configs.append(request(c,priority,.5)['config'])
                    d.call(command='init',session=session,configs=configs,revision=0)
                    trace = []
                    for tick in range(128):
                        current['tick'] = tick
                        contexts = []
                        items = []
                        for i,g in enumerate(games):
                            assert not g.terminal(worlds[i]), ('unexpected early end',current,i)
                            assert worlds[i] == rust_worlds[i]
                            c = g.context(worlds[i],states[i])
                            c['scope'] = configs[i]['scope']
                            contexts.append(c)
                            r = request(c,priority,.5)
                            assert r['config'] == configs[i]
                            items.append(dict(npc=f'npc-{i}',observation=r['observation']))
                        if tick%2: items.reverse()
                        rows = d.call(command='prepare',revision=tick,inputs=items,stochastic=stochastic,workers=1+tick%2)['decisions']
                        receipts = []
                        decisions = []
                        for i,(g,c) in enumerate(zip(games,contexts)):
                            actual = rows[f'npc-{i}']
                            record,error = compare(c,priority,stochastic,actual)
                            count += 1
                            worst = max(worst,error)
                            likely = (tick//32)%3
                            distribution = [.8 if z==likely else .1 for z in range(3)]
                            z = str(rngs[i].choice(3,p=distribution))
                            rust_z = str(rust_rngs[i].choice(3,p=distribution))
                            assert z == rust_z
                            worlds[i],_ = g.transition(worlds[i],record['action_id'],z)
                            rust_worlds[i],_ = g.transition(rust_worlds[i],actual['action_id'],rust_z)
                            assert worlds[i] == rust_worlds[i]
                            states[i] = record['next_state']
                            receipts.append(dict(npc=f'npc-{i}',execution=dict(kind='executed',action_id=actual['action_id'],target=actual['target'])))
                            decisions.append(dict(npc=f'npc-{i}',action=record['action_id'],response=z,state=states[i],world=worlds[i]))
                        result = d.call(command='resolve',next_revision=tick+1,receipts=list(reversed(receipts)))
                        assert result['states'] == {f'npc-{i}':s for i,s in enumerate(states)}
                        world = dict(revision=tick+1,worlds=rust_worlds,rngs=[r.bit_generator.state for r in rust_rngs])
                        saved = d.call(command='checkpoint',world=encoded(world).decode())['checkpoint']
                        if tick == 63:
                            d.restart()
                            restarts += 1
                        restored = d.call(command='restore',session=session,configs=configs,checkpoint=saved)['world']
                        assert restored == encoded(world).decode()
                        checkpoints += 1
                        recovered = json.loads(restored)
                        rust_worlds = recovered['worlds']
                        for r,s in zip(rust_rngs,recovered['rngs']): r.bit_generator.state = s
                        # Roundtrip the entire bundle, not only NPC state or a digest.
                        assert d.call(command='checkpoint',world=restored)['checkpoint'] == saved
                        trace.append(dict(tick=tick,decisions=decisions,save_sha256=hashlib.sha256(saved.encode()).hexdigest()))
                    assert all(g.terminal(w) for g,w in zip(games,worlds))
                    runs += len(games)
                    with gzip.open(out/'runs.jsonl.gz','ab') as f:
                        f.write(encoded(dict(**current,trace=trace,final=worlds))+b'\n')
            print(json.dumps(dict(family=game,completed_trajectories=runs,decisions=count)),flush=True)
        assert runs == 96 and count == 12288 and checkpoints == 3072 and restarts == 24
        for name,value in manifest.items(): assert hashlib.sha256((R/name).read_bytes()).hexdigest()==value, name
        result = dict(complete=True,trajectories=runs,batch_steps=3072,decisions=count,
            checkpoints=checkpoints,process_restarts=restarts,exact_actions_states_rng=True,
            identical_physical_worlds_and_rng=True,max_numeric_absolute_error=worst,
            input_sha256=d.digest.hexdigest(),protocol_requests=d.count,
            runs_sha256=hashlib.sha256((out/'runs.jsonl.gz').read_bytes()).hexdigest(),
            registration_sha256=hashlib.sha256((out/'registration.json').read_bytes()).hexdigest(),
            compiler=subprocess.check_output(['rustc','--version'],text=True).strip(),
            scope=registration['scope'],performance_measured=False)
        (out/'result.json').write_bytes(encoded(result)+b'\n')
        print(json.dumps(result,indent=2),flush=True)
    except Exception as e:
        (out/'failure.json').write_bytes(encoded(dict(location=current,last=d.last,trace=trace,error=repr(e)))+b'\n')
        (out/'result.json').write_bytes(encoded(dict(complete=False,trajectories=runs,decisions=count,error=repr(e)))+b'\n')
        raise
    finally:
        d.close()


if __name__=='__main__': main()
