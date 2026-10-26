"""Public categorical-factor coverage beyond the four game adapters."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
from reflex.segmented_prediction import SegmentedEventLearner
from reflex.core import digest,FEATURES
from reflex.examples import context,action,effect
from tools.verify_rust_immediate import request,encoded
from tools.verify_rust_runtime import Driver


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    binary=R/'rust/target/debug/examples/owned_verify.exe'
    if not binary.exists():binary=binary.with_suffix('')
    d=Driver(binary);checks=0;maximum=0.;cases=[]
    try:
        for factors in ((),('phase',),('phase','terrain'),('phase','terrain','signal','supply')):
            for names in ((),('one','two')):
                c=context('categorical', [action('attempt',effect())])
                base=request(c,'personal',.5)
                definition=dict(model_id='categorical-v1',factors=list(factors),names=list(names),window=128)
                config=dict(actor=base['config'],predictor=definition,contract='factor-reference-v1',purpose_target='public-purpose')
                d.call(op='new',config=config)
                learner=SegmentedEventLearner(owner=digest(c['scope']),model_id='categorical-v1',factors=factors,names=names)
                memory=learner.start()
                for tick in range(160):
                    legal=('a','b','c') if tick%17<12 else ('a','c','d','e')
                    features=tuple(str((tick//(j+3))%2) for j in range(len(factors)))
                    models={name:{z:(.7 if i==n else .3/(len(legal)-1)) for i,z in enumerate(legal)} for n,name in enumerate(names)}
                    inputs=dict(context=list(features),legal=list(legal),models=[[models[name][z] for z in legal] for name in names])
                    expected=learner.predict(memory,features,legal,models)
                    actual=d.call(op='predict',inputs=inputs)['probabilities']
                    error=float(np.max(abs(np.asarray(actual)-list(expected.values()))));maximum=max(maximum,error)
                    assert error<=2e-12,(factors,names,tick,error)
                    observation=copy.deepcopy(base['observation']);observation['tick']=tick
                    conditional=[dict(observation['actions'][0]['outcomes'][0],p=1/len(legal)) for _ in legal]
                    inp=dict(observation=observation,bindings={'attempt':dict(method='attempt',situation='known',estimated=[])},
                        responses=inputs,conditional={'attempt':conditional},
                        purpose=dict(level=0.,readiness=0.,maintained=False,activities={'attempt':dict(kind='uncertain',reason='known rules',condition='known',patience=3,release='',replacement=True)}),
                        threatened=True,maintains_advantage=False)
                    choice=d.call(op='prepare',input=inp,stochastic=False)['decision']['action_id']
                    ticket=d.call(op='commit',action=choice)['ticket'];response=legal[(tick//19+tick%2)%len(legal)]
                    d.call(op='resolve',ticket=ticket,feedback=dict(observed=None,response=response,needs=None,purpose=None))
                    memory=learner.observe(memory,features,legal,models,response)
                    expected_history=[dict(inputs=dict(context=list(e.context),legal=list(e.legal),models=[list(r) for r in e.models]),outcome=e.outcome) for e in memory.history]
                    state=d.call(op='state')['state'];assert state['prediction']==dict(history=expected_history,observations=memory.observations)
                    if tick%31==30:
                        world=dict(tick=tick+1);saved=d.call(op='save',world=world)['save'];d.restart()
                        d.call(op='restore',config=config,world=world,save=saved);assert d.call(op='state')['state']==state
                    checks+=1
                cases.append(dict(factors=factors,names=names,steps=160));print(json.dumps(cases[-1]),flush=True)
        summary=dict(complete=True,cases=cases,predictions=checks,max_error=maximum,
            binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),input_sha256=d.digest.hexdigest())
        (args.output/'summary.json').write_bytes(encoded(summary))
    except BaseException:
        (args.output/'failure.json').write_bytes(encoded(dict(last=d.last,checks=checks,cases=cases)));raise
    finally:d.close()


if __name__=='__main__':main()
