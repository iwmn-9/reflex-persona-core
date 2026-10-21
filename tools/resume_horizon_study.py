"""Resume a verified frozen study without changing its configuration or history."""
import argparse,hashlib,json,platform,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def resume(root,expected_manifest,job,*,recorded_predictions=False,deadline=None):
    assert type(recorded_predictions) is bool
    root=Path(root).resolve();manifest=root/'source_snapshot.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==expected_manifest,'wrong frozen source manifest'
    source=(root/'_source').resolve();files=json.loads(manifest.read_text())
    assert 'tools/run_horizon_study.py' in files
    for name,sha in files.items():
        target=(source/name).resolve()
        assert target.is_relative_to(source) and target.is_file(),'source escaped frozen directory'
        assert hashlib.sha256(target.read_bytes()).hexdigest()==sha,('source changed',name)
    registration=json.loads((root/'registration.json').read_text())
    assert registration['jobs']==[job] and registration['arms']==['receding_recovery']
    trajectory=root/'trajectories.jsonl';audit_path=root/'audit.json'
    audit_only=trajectory.exists()
    assert not (audit_only and audit_path.exists() and json.loads(audit_path.read_text())['complete']),'already complete; never repeat a world'
    checkpoint=root/'checkpoint.json';original=checkpoint.read_bytes()
    envelope=json.loads(original);saved=envelope['payload'];prefix=saved['trace']
    encoded=json.dumps(saved,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
    assert hashlib.sha256(encoded).hexdigest()==envelope['sha256'],'corrupt checkpoint'
    assert saved['contract']['job']==job and saved['contract']['arm']=='receding_recovery'
    assert saved['contract']['runtime']==dict(python=sys.version,numpy=np.__version__,system=platform.system(),machine=platform.machine()),'continuation runtime changed'
    assert all(files[k]==v for k,v in saved['contract']['sources'].items()),'execution source mismatch'
    args=[sys.executable,str(source/'tools/run_horizon_study.py'),'--root',str(root),
          '--job',json.dumps(job),'--stage',registration['stage'],'--frozen']
    configuration=registration.get('configuration',{})
    assert saved['contract'].get('configuration',{})==configuration
    assert set(configuration)<= {'tail_recovery_limit','model_seeds','sampling_standard_errors','cumulative_achievement','tail_goal_refinement','sampling_check','tail_replanning','tail_replanning_horizon'}
    if 'tail_recovery_limit' in configuration:
        args+=['--recovery-limit',str(configuration['tail_recovery_limit'])]
    if 'model_seeds' in configuration:args+=['--model-seeds',*map(str,configuration['model_seeds'])]
    if 'sampling_standard_errors' in configuration:args+=['--sampling-standard-errors',str(configuration['sampling_standard_errors'])]
    if 'cumulative_achievement' in configuration:
        assert configuration['cumulative_achievement'] is True
        args+=['--cumulative-achievement']
    if 'tail_goal_refinement' in configuration:
        assert configuration['tail_goal_refinement'] is True
        args+=['--tail-goal-refinement']
    if 'sampling_check' in configuration:
        assert configuration['sampling_check'] is True
        args+=['--sampling-check']
    if 'tail_replanning' in configuration:
        assert configuration['tail_replanning'] is True
        args+=['--tail-replanning']
    if 'tail_replanning_horizon' in configuration:
        horizon=configuration['tail_replanning_horizon']
        assert configuration.get('tail_replanning') is True and type(horizon) is int and 3<=horizon<=128
        args+=['--tail-replanning-horizon',str(horizon)]
    if audit_only:
        completed=json.loads(trajectory.read_text())
        assert 0<len(prefix)==saved['world']['tick']==completed['executed']==len(completed['trace'])<=job[4],'stored execution boundary mismatch'
        # The original frozen auditor proves actual termination, including
        # death before the scheduled horizon. No later decisions are invented.
        assert completed['final']==saved['world'] and completed['checks']==saved['checks']
        assert all(all(new[k]==value for k,value in old.items()) for old,new in zip(prefix,completed['trace'])),'stored trajectory differs from checkpoint'
        original_trajectory=trajectory.read_bytes()
        args=[sys.executable,str(source/'tools/audit_owned_value.py'),str(root)]
    provenance=dict(resume_at=len(prefix),source_manifest_sha256=expected_manifest,
        original_checkpoint_sha256=hashlib.sha256(original).hexdigest(),complete=False)
    if recorded_predictions:provenance['recorded_prediction_replay_permitted']=True
    if audit_only:provenance.update(audit_only=True,original_trajectory_sha256=hashlib.sha256(original_trajectory).hexdigest())
    record=root/'resume_provenance.json'
    if record.exists():provenance['previous']=json.loads(record.read_text())
    record.write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(resume_at=len(prefix),turns=job[4])),flush=True)
    if (root/'retained_execution.json').exists() and not audit_only:
        args=[sys.executable,str(Path(__file__).with_name('guard_retained_history.py')),*args[1:]]
    def invoke(command,stage):
        if deadline is None:subprocess.run(command,check=True)
        else:
            from tools.horizon_segment import bounded_command
            bounded_command(command,root,deadline,stage)
    try:
        invoke(args,'strict_audit' if audit_only else 'frozen_execution')
    except subprocess.CalledProcessError:
        # Never hide an interrupted world or change any of its decisions.
        # The opt-in alternate auditor still requires exact resulting owners.
        if not recorded_predictions or not trajectory.exists() or audit_path.exists():raise
        from tools.horizon_segment import SegmentPaused
        try:invoke([sys.executable,str(Path(__file__).with_name('audit_recorded_forecasts.py')),str(root)],'recorded_audit')
        except SegmentPaused as error:
            raise RuntimeError('audit recovery did not finish; original failure remains unresolved') from error
    if audit_only:
        assert trajectory.read_bytes()==original_trajectory and checkpoint.read_bytes()==original,'audit changed execution evidence'
    completed=json.loads((root/'trajectories.jsonl').read_text())
    assert len(completed['trace'])>=len(prefix)
    assert all(all(new[k]==value for k,value in old.items()) for old,new in zip(prefix,completed['trace'])),'prior experience changed'
    audit=json.loads((root/'audit.json').read_text());assert audit['complete'] and audit['source_freeze_verified']
    from tools.audit_recorded_forecasts import verify_receipt
    recipe=verify_receipt(root,audit,completed)
    if recipe is not None:provenance['audit_recipe']=recipe
    provenance.update(complete=True,original_prefix_unchanged=True,completed_turns=completed['executed'])
    record.write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--job',required=True);p.add_argument('--recorded-predictions',action='store_true')
    p.add_argument('--deadline',type=float);a=p.parse_args()
    from tools.horizon_segment import SegmentPaused,PAUSED
    try:resume(a.root,a.manifest,json.loads(a.job),recorded_predictions=a.recorded_predictions,deadline=a.deadline)
    except SegmentPaused:raise SystemExit(PAUSED)
