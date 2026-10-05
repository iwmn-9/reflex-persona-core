"""Read-only independent accounting/replay of the frozen congestion run."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.combat import alive,battle_from_record,resolve
from reflex.core import digest
from reflex.continuation_experiment import sources
from reflex.congestion_experiment import jobs,DESIGN,paired_divergence
from reflex.validation_experiment import replay,replay_progress


def verify(output):
    output=Path(output);prereg=json.loads((output/'preregister.json').read_text())
    assert prereg['source_hashes']==sources()
    assert prereg['design_sha256']==hashlib.sha256(DESIGN.read_bytes()).hexdigest()
    raw=[json.loads(line) for line in (output/'trajectories.jsonl').read_text().splitlines()]
    index={(r['partition'],*r['scenario'].split('/'),r['rival'],r['profile'],r['seed'],r['variant']):r for r in raw}
    assert set(index)==set(jobs()) and len(raw)==len(jobs())
    checks=updates=route_checks=0;interventions=[]
    for r in raw:
        checks+=replay(r);updates+=replay_progress(r);measured=Counter();team=r['seed']%2
        for row in r['trace']:
            before=battle_from_record(row['before']);after=battle_from_record(row['after'])
            own=alive(before,team);enemy=alive(before,1-team);choices={int(i):k for i,k in row['choices'].items()}
            assert set(map(int,row['selected_routes']))==set(own);route_checks+=len(own)
            for i in own:
                if not choices[i].startswith('move:'):continue
                measured['move_attempts']+=1
                if before.units[i].pos!=after.units[i].pos:continue
                measured['failed_moves']+=1
                friendly=any(j!=i and choices[j]==choices[i] for j in own)
                opponent=any(choices[j]==choices[i] for j in enemy)
                measured['both_collision_moves' if friendly and opponent else 'friendly_collision_moves' if friendly else 'opponent_collision_moves' if opponent else 'unclassified_failed_moves']+=1
            c=row.get('planning',{}).get('root_completion',{})
            if c.get('adopted'):
                # Posthoc local mechanism replay using REVEALED enemy choices.
                # This is not a prediction or a sample for empirical learning.
                old={**choices,**dict(zip(own,c['before']))}
                counterfactual,_=resolve(before,old,int(digest(['combat-world',r['seed']])[:16],16))
                failed=lambda selected,state:sum(selected[i].startswith('move:') and before.units[i].pos==state.units[i].pos for i in own)
                interventions.append(dict(condition=[r[k] for k in ('partition','scenario','profile','seed')],tick=before.tick,
                    before=c['before'],after=c['after'],local_failed_moves_without=failed(old,counterfactual),
                    local_failed_moves_with=failed(choices,after),max_persona_regret=max(c['persona_regret']),
                    combinations=c['adapter']['combinations'],interpretation='posthoc same-state rule replay with revealed opponent intent; never a policy input'))
        for k in ('move_attempts','failed_moves','friendly_collision_moves','opponent_collision_moves','both_collision_moves','unclassified_failed_moves'):
            assert measured[k]==r['movement'].get(k,0),(k,r['scenario'])
        assert measured['failed_moves']==sum(measured[k] for k in ('friendly_collision_moves','opponent_collision_moves','both_collision_moves','unclassified_failed_moves'))
        d=r['diagnostics']
        assert d['occupancy_prediction_trials']==measured['move_attempts']
        assert d['occupancy_actual_contests']==measured['opponent_collision_moves']+measured['both_collision_moves']
        assert d['chosen_route_moves']==d['chosen_route_improving_moves']+d['chosen_route_nonimproving_moves']==measured['move_attempts']
        assert not d['adopted_plan_friendly_conflict_ticks']
    divergences=[paired_divergence(index[jobs()[i]],index[jobs()[i+1]]) for i in range(0,len(jobs()),2)]
    result=dict(format='controlled-congestion-verification-v1',source_hashes=sources(),games=len(raw),
        independent_rule_checks=checks,independent_progress_checks=updates,route_owner_checks=route_checks,
        paired_games=len(divergences),identical_pairs=sum(d.get('same_trajectory',False) for d in divergences),
        changed_pairs=sum(d['tick'] is not None for d in divergences),interventions=interventions,
        artifact_hash_scope='Generated raw files in the selected output directory, not compact evidence exports',
        artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.json*')) if p.name!='verification.json'},
        defaults_changed=False,candidate_changed=False,
        limitation='Local intervention replay is posthoc mechanism evidence; not heldout outcome effect or opponent prediction calibration.')
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='reflex_artifacts/controlled_congestion')
    r=verify(p.parse_args().output);print({k:r[k] for k in ('games','independent_rule_checks','independent_progress_checks','identical_pairs','changed_pairs')})
