"""Explicit replay of saved decision-time forecasts across rounding differences.

The frozen auditor still verifies every physical event and EXACT resulting
owner hash. Only a recomputed public probability may be replaced by the saved
predecision probability, after an eight-ULP check. This does not replay a fresh
action selection or certify an approximate owner state. Acting code is frozen.
"""
import argparse,hashlib,importlib,json,math,sys
from pathlib import Path


MAX_ULPS=8


def probability_differences(saved,fresh):
    if not isinstance(saved,dict) or not isinstance(fresh,dict) or saved.keys()!=fresh.keys():
        raise ValueError('same complete recorded response space required')
    differences=[]
    for name in fresh:
        a,b=saved[name],fresh[name]
        if any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in (a,b)):
            raise ValueError('finite probability required')
        if a!=b:
            ulps=abs(a-b)/max(math.ulp(a),math.ulp(b))
            if ulps>MAX_ULPS:raise ValueError('recorded forecast exceeds numerical replay bound')
            differences.append(dict(response=name,saved=a,recomputed=b,absolute=abs(a-b),ulps=ulps))
    if any(abs(math.fsum(p.values())-1)>8*math.ulp(1.) for p in (saved,fresh)):
        raise ValueError('normalized recorded forecast required')
    return differences


def verify_receipt(root,audit,row):
    """Validate explicit alternate-audit provenance when collecting evidence."""
    root=Path(root);path=root/'recorded_forecast_replay.json'
    compact=audit.get('recorded_forecast_replay')
    if compact is None:
        assert not path.exists(),'alternate audit receipt cannot be hidden'
        return None
    receipt=json.loads(path.read_text())
    assert compact=={k:v for k,v in receipt.items() if k not in ('differences','semantics')}
    assert receipt['version']=='recorded-forecast-replay-v1' and receipt['accepted'] is True
    assert receipt['max_allowed_ulps']==MAX_ULPS
    assert receipt['checked_forecasts']==receipt['exact_owner_hashes']==row['executed']
    manifest=(root/'source_snapshot.json').read_bytes()
    assert receipt['source_manifest_sha256']==hashlib.sha256(manifest).hexdigest()
    assert receipt['frozen_auditor_sha256']==json.loads(manifest)['tools/audit_owned_value.py']
    assert receipt['trajectory_sha256']==audit['trajectory_sha256']==hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest()
    assert audit['complete'] and audit['source_freeze_verified']
    for key in ('world_rng_replay','restored_action_feedback','physical_replay','owner_parity','fixed_axes'):
        assert audit['checks'][key]==row['executed']
    maxima=[0.];absolute=[0.];seen=set();fresh_by_tick={};differences_by_tick={}
    for d in receipt['differences']:
        tick,key=d['tick'],d['response']
        assert type(tick) is int and 0<=tick<row['executed'] and (tick,key) not in seen
        seen.add((tick,key));saved=row['trace'][tick]['law']
        assert saved[key]==d['saved']
        fresh_by_tick.setdefault(tick,dict(saved))[key]=d['recomputed']
        differences_by_tick.setdefault(tick,{})[key]={k:v for k,v in d.items() if k!='tick'}
        maxima.append(d['ulps']);absolute.append(d['absolute'])
    for tick,fresh in fresh_by_tick.items():
        actual=probability_differences(row['trace'][tick]['law'],fresh)
        assert {d['response']:d for d in actual}==differences_by_tick[tick]
    assert receipt['max_observed_ulps']==max(maxima)<=MAX_ULPS
    assert receipt['max_absolute_difference']==max(absolute)
    return compact


def audit(root):
    root=Path(root).resolve();source=root/'_source';target=root/'audit.json'
    if target.exists():raise FileExistsError('existing audit must be preserved')
    raw=(root/'trajectories.jsonl').read_bytes();manifest=(root/'source_snapshot.json').read_bytes()
    row=json.loads(raw);trace=row['trace']
    if not trace or len(trace)!=row['executed']:raise ValueError('one complete recorded trajectory required')
    hashes=json.loads(manifest)
    for name,sha in hashes.items():
        path=(source/name).resolve()
        if not path.is_relative_to(source):raise ValueError('source escaped frozen directory')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('frozen source changed')
    # This tool must run in a fresh process, so a previously imported checkout
    # cannot substitute for the original acting implementation.
    sys.path.insert(0,str(source))
    core=importlib.import_module('reflex.core')
    reader=importlib.import_module('reflex.event_reader')
    frozen=importlib.import_module('tools.audit_owned_value')
    for name,module in tuple(sys.modules.items()):
        if name in ('reflex','tools') or name.startswith(('reflex.','tools.')):
            path=getattr(module,'__file__',None)
            if path and not Path(path).resolve().is_relative_to(source):raise ValueError('fresh frozen-only audit process required')
    original=reader.EventReader.predict;seen=set();differences=[]
    receipt=dict(version='recorded-forecast-replay-v1',accepted=False,max_allowed_ulps=MAX_ULPS,
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_auditor_sha256=hashes['tools/audit_owned_value.py'],
        source_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        trajectory_sha256=hashlib.sha256(raw).hexdigest(),
        semantics=__doc__)

    def recorded(self,context):
        tick=context['tick']
        if type(tick) is not int or not 0<=tick<len(trace) or trace[tick]['tick']!=tick:
            raise ValueError('recorded factual decision required')
        t=trace[tick]
        if core.digest(self.record())!=core.digest(t['checkpoint']['predictor']):
            raise ValueError('recorded factual predictor required')
        fresh=original(self,context);saved=t['law']
        changes=probability_differences(saved,fresh)
        differences.extend(dict(tick=tick,**change) for change in changes)
        seen.add(tick)
        return {key:saved[key] for key in fresh}

    reader.EventReader.predict=recorded
    try:
        result=frozen.audit(root)
        assert seen==set(range(row['executed'])),'all factual forecasts must be checked'
        assert result['complete'] and result['checks']['restored_action_feedback']==row['executed']
        assert (root/'trajectories.jsonl').read_bytes()==raw and (root/'source_snapshot.json').read_bytes()==manifest
        receipt.update(accepted=True,checked_forecasts=len(seen),exact_owner_hashes=row['executed'],
            max_observed_ulps=max((d['ulps'] for d in differences),default=0),
            max_absolute_difference=max((d['absolute'] for d in differences),default=0),differences=differences)
        result['recorded_forecast_replay']={k:v for k,v in receipt.items() if k not in ('differences','semantics')}
        target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        return result
    except Exception as error:
        receipt.update(error=repr(error),checked_forecasts=len(seen),differences=differences)
        if target.exists():target.rename(root/'audit_replay_unaccepted.json')
        raise
    finally:
        reader.EventReader.predict=original
        (root/'recorded_forecast_replay.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();audit(a.root)
