"""Exposed terminal rule states, distinct from naturally encountered matches.

128 structurally legal ledger states x four fixed profiles x two controllers.
Both roots settle without another owner's choice. The independent arithmetic
below certifies all predicted endpoints, not a sampled optimistic future.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from reflex.board_models import ThanksPosition
from reflex.laboratory import PROFILES
from reflex.strong_search import PublicMemory, SearchBudget
from reflex.strong_table import decide


def points(cards,chips):
    hand=set(cards);return sum(c for c in hand if c-1 not in hand)-chips


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    plan=dict(version='certified-expiry-terminal-exposure-v1',states=128,generation_seed=90315,
        profiles=[p['id'] for p in PROFILES],budget=asdict(SearchBudget(8,16,16,1)),
        construction='24 distinct seen cards; 23 allocated to four players, one current; 44 chips, next actor 0, owner >=1',
        primary='known terminal winner-credit loss vs best legal root must stay within .12',
        secondary='actual own-score changes, profile preservation, complete terminal prediction equality',
        limitations=['synthetic legal rule states; prior reachability/encounter frequency not established',
                     'same known game, no human-level proof'],
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((Path(__file__).resolve().parents[1]/'reflex').glob('*.py'))})
    path=root/'preregister.json';assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    rng=np.random.default_rng(plan['generation_seed']);rows=[];counts={p['id']:dict(improved=0,same=0,worse=0,credit_improved=0,credit_worse=0) for p in PROFILES}
    for case in range(128):
        seen=tuple(sorted(map(int,rng.permutation(np.arange(3,36))[:24])));card=int(rng.choice(seen))
        allocation=rng.integers(0,4,size=23);held=[[],[],[],[]]
        for actor,c in zip(allocation,[x for x in seen if x!=card]):held[int(actor)].append(c)
        viewer=case%4;next_actor=(viewer+1)%4;chips=[0]*4;chips[viewer]=int(rng.integers(1,23))
        other=[a for a in range(4) if a not in (viewer,next_actor)];chips[other[0]]=int(rng.integers(0,45-chips[viewer]));chips[other[1]]=44-sum(chips)
        s=ThanksPosition(tuple(tuple(h) for h in held),tuple(chips),viewer,card,0,seen,0,(0,0,0,0))
        terminals=[]
        for move in ('TAKE','PASS'):
            cards=[list(h) for h in held];end_chips=chips.copy()
            if move=='TAKE':cards[viewer].append(card)
            else:end_chips[viewer]-=1;end_chips[next_actor]+=1;cards[next_actor].append(card)
            terminals.append([points(c,h) for c,h in zip(cards,end_chips)])
        shares=[]
        for terminal in terminals:
            top=min(terminal);shares.append(float(terminal[viewer]==top)/terminal.count(top))
        decisions=[]
        for p in PROFILES:
            pair=[]
            for variant in ('progress','certified_expiry'):
                context,d,stats=decide('no_thanks',s,viewer,p,9100+case,0,0,PublicMemory('no_thanks',viewer),None,'adaptive',SearchBudget(**plan['budget']),variant=variant)
                assert context['personality']==dict(zip(('openness','conscientiousness','extraversion','agreeableness','neuroticism'),p['traits']))
                for i,move in enumerate(('TAKE','PASS')):
                    assert stats['actions'][move]['mean_score']==terminals[i][viewer]
                    assert stats['actions'][move]['win_share']==shares[i]
                chosen=('TAKE','PASS').index(d['action_id']);assert max(shares)-shares[chosen]<=.12+1e-12
                pair.append(dict(variant=variant,action=d['action_id'],score=terminals[chosen][viewer],credit=shares[chosen]))
            old,new=pair;c=counts[p['id']];delta=old['score']-new['score']
            c['improved' if delta>0 else 'worse' if delta<0 else 'same']+=1
            c['credit_improved']+=new['credit']>old['credit'];c['credit_worse']+=new['credit']<old['credit']
            decisions.append(dict(profile=p['id'],decisions=pair))
        rows.append(dict(case=case,public_state=asdict(s),known_terminal_scores=terminals,known_terminal_credits=shares,decisions=decisions))
    result=dict(plan=plan,rows=rows,counts=counts,decisions=1024,terminal_action_predictions_checked=2048,
        fixed_profiles_preserved=True,all_goal_regrets_within_declared_tolerance=True)
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(counts,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
