"""Compare compiled numeric Policy with a dependency-free Rust translation.

This is a portability probe, NOT an exported/validated Rust runtime interface.
Python still owns JSON validation, identifiers, hash/seed construction, adapters,
social appraisal, return units, prediction, and search.
"""
from pathlib import Path
import copy
import argparse
import hashlib
import json
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from reflex.core import Policy, compile_batch, digest, TRAITS, NEEDS, VALUES
from reflex.examples import context, action, effect, families
from reflex.laboratory import PROFILES
from tools.asymmetric_games import GAMES, CASES, AsymmetricGame, board


def random_context(rng, i, count=None):
    acts = []
    for j in range(count or int(rng.integers(1, 20))):
        probabilities = rng.dirichlet(np.ones(int(rng.integers(1, 9))))
        outcomes = [effect(float(rng.uniform(-1, 1)),
            dict(zip(NEEDS, rng.uniform(-1, 1, 5))), dict(zip(VALUES, rng.uniform(-1, 1, 10))),
            float(rng.random()), float(p), dict(zip(TRAITS, rng.uniform(-1, 1, 5)))) for p in probabilities]
        acts.append(action(f'a{j:03}', *outcomes, legal=j == 0 or bool(rng.random() > .2),
            confidence=float(rng.random()), familiarity=float(rng.random()),
            failure=j != 0 and bool(rng.random() < .1), switch_cost=float(rng.random())))
    c = context(f'port-{i}', acts, dict(zip(NEEDS, rng.random(5))),
        dict(zip(VALUES, rng.random(10))), dict(zip(TRAITS, rng.random(5))), mode=None)
    c['seed'] = int(rng.integers(0, 2**63 - 1)); c['tick'] = int(rng.integers(0, 100000))
    for name in NEEDS:
        if rng.random() < .3: c['needs'][name] = dict(supported=True, enabled=False, deficit=None)
    enabled = [n for n in NEEDS if c['needs'][n]['enabled']]
    primary = enabled[int(rng.integers(len(enabled)))] if enabled and rng.random() < .6 else None
    c['state'].update(primary_need=primary, mode=[None, 'need', 'principle'][int(rng.integers(3))],
        mode_urgency=float(rng.random()), intent_action=acts[int(rng.integers(len(acts)))]['id'], age=int(rng.integers(100)))
    c['opponent'] = dict(confidence=float(rng.choice([.2, .6, .9])), uncertainty=float(rng.random()),
        evidence_ids=['options'], threatened=bool(rng.integers(2)), maintains_advantage=bool(rng.integers(2)),
        deltas={a['id']: float(rng.uniform(-1, 1)) for a in acts})
    return c


def encode(c, priority, rigidity, residual, stochastic):
    b = compile_batch([c]); a = b.effects.shape[1]; k = b.effects.shape[2]
    policy = Policy(residual, principle_priority=priority,
        **(dict(commitments={b.characters[0]: rigidity}) if priority == 'personal' else {}))
    expected = policy.decide(b, stochastic)
    row = [a, k, int(stochastic), ['lexicographic','finite','personal'].index(priority), rigidity,
        int(digest([c['seed'], c['scope']])[:16], 16), c['tick']]
    for key in ('traits','values','needs','enabled'): row.extend(getattr(b, key)[0].astype(float))
    for key in ('primary','mode','intent','age'): row.append(int(getattr(b, key)[0]))
    for key in ('mode_urgency','read_confidence','read_uncertainty'): row.append(float(getattr(b, key)[0]))
    row.extend([int(b.threatened[0]),int(b.ahead[0])]); row.extend(residual)
    for j in range(a):
        row.extend([b.confidence[0,j],b.familiarity[0,j],int(b.known_failure[0,j]),b.switch_cost[0,j],int(b.legal[0,j]),b.reading[0,j]])
        row.extend(b.probability[0,j]); row.extend(b.effects[0,j].ravel())
    header = [expected.action[0],expected.primary[0],expected.mode[0],expected.mode_urgency[0],
        int(expected.reading_used[0]),int(expected.request_reading[0]),expected.records(b)[0]['next_state']['age']]
    values = np.array([*header,*b.rng[0],*expected.scores[0],*expected.features[0].ravel(),*expected.eligible[0].astype(int)])
    return ' '.join(map(str,row)), values, expected.records(b)[0]['next_state']


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    output_dir=Path(parser.parse_args().output); output_dir.mkdir(parents=True,exist_ok=False)
    binary = output_dir / ('policy_probe.exe' if sys.platform=='win32' else 'policy_probe')
    subprocess.run(['rustc', '-O', str(HERE/'rust_policy_probe.rs'), '-o', str(binary)], check=True)
    rng = np.random.default_rng(20261015)
    cs = [copy.deepcopy(r['context']) for r in families()]
    for game in GAMES:
        for case in CASES:
            for p in PROFILES:
                g = AsymmetricGame(game,p,board(game,case)); cs.append(g.observe(g.start()))
    cs += [random_context(rng,i) for i in range(256)]
    cs += [random_context(rng,10000+i,count=n) for i,n in enumerate((1,2,32,128,256))]
    ties = context('ties',[action('a',effect(.3)),action('b',effect(.3))],needs={n:0 for n in NEEDS})
    cs.append(ties)
    records = []
    for i,c in enumerate(cs):
        for priority,rigidity in [('lexicographic',0),('finite',0),('personal',0),('personal',.5),('personal',1)]:
            for stochastic in (False,True):
                residual = np.zeros(8) if i % 2 else rng.uniform(-.08,.08,8)
                records.append(encode(c,priority,rigidity,residual,stochastic))
    # Stateful decisions must remain consistent across many clock/intent updates.
    for j in range(16):
        c = copy.deepcopy(cs[j]); c['scope']['episode'] += '-sequence'
        for tick in range(64):
            c['tick']=tick
            record=encode(c,'finite',0,np.zeros(8),True); records.append(record)
            c['state']=record[2]
    payload=str(len(records))+'\n'+'\n'.join(r[0] for r in records)+'\n'
    result=subprocess.run([str(binary)],input=payload,text=True,capture_output=True,check=True)
    lines=result.stdout.splitlines(); assert len(lines)==len(records)
    worst=0.; mismatches=[]
    for i,(line,(_,expected,_)) in enumerate(zip(lines,records)):
        got=np.fromstring(line,sep=' '); assert got.shape==expected.shape
        # Categorical outputs and isolated random streams are exact.
        for index in range(10):
            if got[index]!=expected[index]: mismatches.append((i,index,float(got[index]),float(expected[index])))
        worst=max(worst,float(np.max(abs(got-expected))))
        if not np.allclose(got,expected,rtol=0,atol=2e-12): mismatches.append((i,'numeric'))
    output=dict(scope='compiled numeric base Policy only; no social/return-frame/search/learner/API migration',
        cases=len(records),contexts=len(cs),stateful_decisions=1024,compiler=subprocess.check_output(['rustc','--version'],text=True).strip(),
        modes=['lexicographic','finite','personal 0/.5/1'],max_numeric_absolute_error=worst,
        failures=mismatches,action_rng_state_exact=not mismatches,
        python_core_sha256=hashlib.sha256((REPO/'reflex/core.py').read_bytes()).hexdigest(),
        rust_source_sha256=hashlib.sha256((HERE/'rust_policy_probe.rs').read_bytes()).hexdigest(),
        test_input_sha256=hashlib.sha256(payload.encode()).hexdigest(),performance_measured=False,
        limits='sampled parity is not proof at every floating threshold; Windows only; input parser assumes valid fixture')
    (output_dir/'result.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output,indent=2)); assert not mismatches


if __name__=='__main__': main()
