"""Independent grid replay of completed public-board rollout studies.

The frozen producer checks complete own Policy receipts at every real turn.
This auditor independently reconstructs drops/lines/terminal credit and the
registered sampling balance; it does not claim independent policy optimality.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np


def winner(grid):
    for row in range(6):
        for col in range(7):
            side=grid[row][col]
            if side is None:continue
            for dr,dc in ((1,0),(0,1),(1,1),(1,-1)):
                if all(0<=row+i*dr<6 and 0<=col+i*dc<7 and grid[row+i*dr][col+i*dc]==side for i in range(4)):
                    return side
    return None


def encoded(grid,turn):
    discs=[0,0];heights=[]
    for col in range(7):
        occupied=[r for r in range(6) if grid[r][col] is not None]
        assert occupied==list(range(len(occupied)))
        heights.append(len(occupied))
        for row in occupied:discs[grid[row][col]]|=1<<(7*col+row)
    return dict(discs=discs,heights=heights,turn=turn)


def audit(root):
    root=Path(root);plan=json.loads((root/'preregister.json').read_text(encoding='utf-8'))
    result=json.loads((root/'evaluation.json').read_text(encoding='utf-8'));assert result['plan']==plan
    for name,sha in plan['sources'].items():
        assert hashlib.sha256((root/'_source'/name).read_bytes()).hexdigest()==sha,name
    snapshot=json.loads((root/'source_snapshot.json').read_text(encoding='utf-8'))
    for name,sha in snapshot.items():
        assert hashlib.sha256((root/'_source'/name).read_bytes()).hexdigest()==sha,name
    counts=Counter();rows=[];keys=set();conditions=plan.get('conditions',['baseline','candidate'])
    for line in (root/'trajectories.jsonl').open(encoding='utf-8'):
        row=json.loads(line);rows.append(row);seed=row['seed']
        label=row.get('condition','candidate' if row['candidate'] else 'baseline')
        key=(seed,row['profile'],row.get('rival','minimax2'),label);assert key not in keys;keys.add(key)
        assert seed in plan['seeds'] and row['profile'] in plan['profiles'] and label in conditions and row['seat']==seed%2
        grid=[[None]*7 for _ in range(6)];turn=0;rng=np.random.default_rng(seed+331071)
        for _ in range(6):
            legal=[c for c in range(7) if grid[5][c] is None];col=legal[int(rng.integers(len(legal)))]
            height=next(r for r in range(6) if grid[r][col] is None);grid[height][col]=turn;turn=1-turn
        assert winner(grid) is None
        used=0
        for tick,frame in enumerate(row['trace']):
            assert frame['tick']==tick and frame['actor']==turn and frame['before']==encoded(grid,turn)
            assert winner(grid) is None
            name=frame['action'];assert name.startswith('DROP:');col=int(name[5:])
            assert 0<=col<7 and grid[5][col] is None
            height=next(r for r in range(6) if grid[r][col] is None);grid[height][col]=turn;turn=1-turn
            assert frame['after']==encoded(grid,turn);counts['public_actions']+=1
            if frame['actor']==row['seat']:
                counts['producer_checked_complete_owner_receipts']+=1
            st=frame['stats'];roll=None if st is None else st.get('policy_rollout')
            if roll:
                used+=1;assert roll['used'] and roll['completed_samples']==plan['samples']
                assert all(a['samples']==plan['samples'] for a in roll['actions'].values())
                assert roll['discarded_terminal_samples']==0
                assert roll['action']==name and name in roll['guard']['allowed']
                counts['root_interventions']+=1;counts['changed_roots']+=roll['changed']
                counts['hypothetical_owner_searches']+=roll['owner_searches']
                counts['paired_hypothetical_terminal_branches']+=len(roll['actions'])*plan['samples']
        owner_turns=sum(f['actor']==row['seat'] for f in row['trace'])
        expected_interventions=owner_turns if label=='continuous' else int(label!='baseline' and owner_turns>0)
        assert used==expected_interventions,key
        if label!='baseline' and not owner_turns:counts['candidate_without_owner_turn']+=1
        w=winner(grid);assert w==row['winner']
        assert w is not None or all(grid[5][c] is not None for c in range(7))
        assert row['credit']==(.5 if w is None else float(w==row['seat']))
        counts['primary_games']+=1
    expected=len(plan['seeds'])*len(plan['profiles'])*len(plan.get('rivals',['minimax2']))*len(conditions)
    assert counts['primary_games']==expected
    metadata=[{k:v for k,v in r.items() if k!='trace'} for r in rows]
    assert metadata==result['runs']
    actual_sha=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest()
    assert actual_sha==result['trajectory_sha256']
    proof=dict(counts=dict(counts),source_files=len(snapshot),trajectory_sha256=actual_sha,
        evaluation_sha256=hashlib.sha256((root/'evaluation.json').read_bytes()).hexdigest(),
        scope='grid rules and credit independently reconstructed; own full-record checks executed by hash-frozen producer; not an optimality or human-level proof')
    (root/'audit.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof,indent=2));return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();audit(a.root)
