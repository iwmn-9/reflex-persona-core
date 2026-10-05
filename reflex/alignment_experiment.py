"""Offline combat wiring for sampled forecast/execution diagnostics.

Only the all-public authored combat adapter is connected. Other games must
supply their own observed-state/action contract before reusing the comparator.
"""
from collections import Counter
from .alignment import compare_execution
from .core import digest
from .combat import battle_from_record, battle_record, alive, terminal


def audit_combat_execution(run):
    if run.get('genre') != 'combat' or 'execution_contract' not in run:
        raise ValueError('execution-traced combat run required')
    team = run['seed'] % 2
    contract = run['execution_contract']
    if contract['group'] != f'team-{team}' or contract['unit'] != 'combat-tick':
        raise ValueError('combat owner and observation unit required')
    actual = {}; previous = -1; previous_after = None; ended = False
    for row in run['trace']:
        before, after = battle_from_record(row['before']), battle_from_record(row['after'])
        if ended or terminal(before):
            raise ValueError('no real transitions from or after an absorbing terminal')
        if previous_after is not None and before.tick == previous + 1 and battle_record(before) != previous_after:
            raise ValueError('continuous observed combat states required')
        if before.tick <= previous or after.tick != before.tick + 1:
            raise ValueError('ordered distinct real combat ticks required')
        choices = {str(i): k for i, k in row['choices'].items()}
        if set(choices) != {str(i) for t in (0, 1) for i in alive(before, t)}:
            raise ValueError('all subsequently revealed living-actor actions required')
        actual[before.tick] = dict(index=before.tick, before=digest(battle_record(before)),
            after=digest(battle_record(after)), self={str(i):choices[str(i)] for i in alive(before,team)},
            opponent={str(i):choices[str(i)] for i in alive(before,1-team)},terminal=terminal(after))
        previous = before.tick; previous_after = battle_record(after); ended = terminal(after)
    comparisons = []; totals = Counter()
    for row in run['trace']:
        selected = row.get('planning', {}).get('execution')
        if selected is None: continue
        if not row['planning'].get('adopted') or selected['start'] != row['before']['tick']:
            raise ValueError('same-tick final adopted execution trace required')
        if type(selected.get('horizon')) is not int or not 1 <= selected['horizon'] <= 16:
            raise ValueError('bounded selected execution horizon required')
        samples = selected['samples']
        if not isinstance(samples, list) or not 1 <= len(samples) <= 8:
            raise ValueError('bounded sampled execution paths required')
        steps = []
        for offset in range(selected['horizon']):
            step = actual.get(selected['start'] + offset)
            steps.append(step)
            if step is not None and step['terminal']: break
        observed = dict(**contract, start=selected['start'], horizon=selected['horizon'], steps=steps)
        for sample, modeled in enumerate(samples):
            forecast = {k:v for k,v in selected.items() if k != 'samples'}
            forecast['steps'] = modeled
            result = compare_execution(forecast, observed)
            comparisons.append(dict(start=selected['start'], sample=sample, **result))
            totals.update({k:v for k,v in result.items() if type(v) is int and not k.startswith('first_')})
    return dict(selected_forecasts=sum('execution' in row.get('planning', {}) for row in run['trace']),
                sampled_paths=len(comparisons), totals=dict(totals), comparisons=comparisons)


def experiment(output, progress=None):
    """Reproduce the known seed-180 case with tracing off/on, not a strength trial."""
    import copy
    import json
    from dataclasses import replace
    from pathlib import Path
    from .combat_planning import TacticalControl
    from .laboratory import profiles
    from .validation_experiment import combat, replay, replay_progress
    from .purpose_experiment import source_hashes
    from .cross_games import core_hashes
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sources=source_hashes();core=core_hashes();profile=next(p for p in profiles() if p['id']=='steady')
    fixed=copy.deepcopy(profile);results=[];rule_checks=0;progress_checks=0
    with (output/'trajectories.jsonl').open('w',encoding='utf-8') as stream:
        for variant in ('purposeful','options'):
            control=TacticalControl(recovery_options=variant=='options');runs=[]
            for enabled in (False,True):
                run=combat(profile,180,'choke','eliminate','baseline',learn=True,survival_security=True,
                    rival='switch',planner=replace(control,trace_execution=enabled),progress_watch=True)
                rule_checks+=replay(run);progress_checks+=replay_progress(run)
                stream.write(json.dumps(dict(variant=variant,tracing=enabled,**run),ensure_ascii=False)+'\n');stream.flush()
                runs.append(run)
                if progress:progress(f'{variant}: trace={enabled}, {run["ticks"]} ticks replayed',flush=True)
            old,new=runs
            for a,b in zip(old['trace'],new['trace']):
                for key in ('before','after','choices','audit','purpose'):assert a[key]==b[key],key
                assert {k:v for k,v in b.get('planning',{}).items() if k!='execution'}==a.get('planning',{})
            stable=('won','lost','actions','learned_uses','regime_resets','decisions','planning_nodes','ticks','score','health')
            assert all(old[k]==new[k] for k in stable)
            audit=audit_combat_execution(new)
            assert audit==audit_combat_execution(json.loads(json.dumps(new)))
            assert all(c['steps'][0]['self']=='same' for c in audit['comparisons'])
            at19=next(row for row in new['trace'] if row['before']['tick']==19)
            selected=at19['planning'].get('execution')
            details=dict(tick=19,actual_choices={str(i):k for i,k in at19['choices'].items()},
                selected_path=at19['planning']['selected_path'],
                recovery=at19['purpose'][0].get('recovery'),
                samples=[] if selected is None else [dict(sample=i,root_own=s[0]['self'],root_opponent=s[0]['opponent']) for i,s in enumerate(selected['samples'])],
                comparisons=[c for c in audit['comparisons'] if c['start']==19])
            results.append(dict(variant=variant,won=new['won'],ticks=new['ticks'],trace_on_off_identical=True,
                audit={k:v for k,v in audit.items() if k!='comparisons'},tick19=details,
                same_state_self_mismatch_examples=[dict(start=c['start'],sample=c['sample'],offset=c['first_self_difference'])
                    for c in audit['comparisons'] if c['first_self_difference'] is not None][:3]))
    assert profile==fixed and sources==source_hashes() and core==core_hashes()
    report=dict(format='forecast-execution-alignment-v1',condition='choke/eliminate/steady/seed180/switch',
        games=4,source_hashes=sources,core_hashes=core,rule_transitions_replayed=rule_checks,
        purpose_updates_replayed=progress_checks,runs=results,defaults_changed=False,learning_changed=False,
        limitations=['One known regression condition, not held-out strength evidence.',
            'Counts are sampled transitions; multiple samples share each actual observation.',
            'Same public pre-state permits descriptive action comparison, not causal blame or learned trust.',
            'Diverged states, unknown actions and missing future are not policy mismatches.',
            'Terminal purpose padding is never credited as executed actions.',
            'Only the fully public combat adapter is connected; no hidden-state inference or cross-game guarantee.'])
    (output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report
