"""Operational pause/resume between resolved decisions of one continuing world.

This stores the full learner and world RNG. It never substitutes a new world,
shortens the game's horizon, or changes the planning/learning configuration.
Use a frozen source checkout for executions which can outlive the current job.
"""
import hashlib,json,platform,sys
from pathlib import Path
import numpy as np


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
def sha(data):return hashlib.sha256(data).hexdigest()


class OwnedCheckpoint:
    def __init__(self,path,job,arm,*,max_decisions=None,every=8):
        if max_decisions is not None and (type(max_decisions) is not int or max_decisions<1):
            raise ValueError('positive decision slice required')
        if type(every) is not int or every<1:raise ValueError('positive checkpoint cadence required')
        self.path=Path(path);self.max_decisions=max_decisions;self.every=every
        root=Path(__file__).resolve().parents[1]
        names=[p.relative_to(root).as_posix() for p in (root/'reflex').glob('*.py')]
        names+=['tools/run_owned_value.py','tools/continuing_games.py','tools/owned_checkpoint.py']
        self.contract=dict(version='owned-execution-v1',job=list(job),arm=arm,
            sources={n:sha((root/n).read_bytes()) for n in sorted(names)},
            runtime=dict(python=sys.version,numpy=np.__version__,system=platform.system(),machine=platform.machine()))
        self.saved=None
        if self.path.exists():
            envelope=json.loads(self.path.read_text(encoding='utf8'));saved=envelope['payload']
            if sha(encoded(saved))!=envelope['sha256']:raise ValueError('checkpoint bytes do not match their checksum')
            if saved['contract']!=self.contract:raise ValueError('checkpoint execution/source/runtime contract changed')
            if saved['world']['tick']!=len(saved['trace']) or saved['owner']['last_tick']+1!=len(saved['trace']):
                raise ValueError('checkpoint world and owner timeline disagree')
            if not saved['trace'] or saved['trace'][-1]['after']!=saved['world']:
                raise ValueError('checkpoint must follow an actual resolved transition')
            self.saved=saved
        self.initial_tick=0 if self.saved is None else len(self.saved['trace'])

    def stopping(self,count):
        return self.max_decisions is not None and count-self.initial_tick>=self.max_decisions

    def save(self,world,owner,rng,trace,checks,elapsed,*,terminal):
        if not terminal and not self.stopping(len(trace)) and len(trace)%self.every:return
        payload=dict(contract=self.contract,world=world,owner=owner.record(),
            rng=rng.bit_generator.state,trace=trace,checks=dict(checks),elapsed=elapsed)
        envelope=dict(payload=payload,sha256=sha(encoded(payload)))
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.path.with_name(self.path.name+'.new')
        temporary.write_bytes(encoded(envelope));temporary.replace(self.path)
