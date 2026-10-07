"""Replay private real episodes through exact rules, independently of Probe.actual.

Usage: python tools/audit_payback_cycle.py PATH_TO_CLOSED_LOOP_PAYBACK
Only aggregate evidence is published; full model/actual traces stay local.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.purpose_experiment import source_hashes


def audit(folder):
    folder=Path(folder)
    result=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    prereg=json.loads((folder/'preregister.json').read_text(encoding='utf-8'))
    assert result['source_hashes']==prereg['source_hashes']==source_hashes()
    assert result['preregister_digest']==digest(prereg)
    rows=[json.loads(s) for s in (folder/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==len(result['runs'])==result['episodes']
    transitions=0;revised=0;deaths=0;secondary={}
    for r,short in zip(rows,result['runs']):
        assert {k:v for k,v in r.items() if k!='trace'}==short
        spec=r['spec'];genre=spec['genre'];trace=r['trace'];total=0.
        assert trace and len({t['persona_hash'] for t in trace})==1
        if genre=='combat':assert r['actual_seed'] not in prereg['model_combat_seeds']
        for index,t in enumerate(trace):
            if index:assert t['before']==trace[index-1]['after']
            if genre=='resources':
                from reflex.resource_world import world_from_record,world_record,step,terminal
                w=world_from_record(t['before']);assert not terminal(w)
                after=step(w,t['root'])
                while not terminal(after) and after.turn!=0:after=step(after,'wait')
                if spec.get('famine') and after.round==2:after=replace(after,food_yield=0);revised+=1
                worth=lambda s:(sum(s.empires[0].stock)+s.empires[0].science+s.empires[0].culture)/100
                record=world_record(after)
            elif genre=='auction':
                from reflex.contests import public_from_record,settle
                w=public_from_record(t['before']);assert w.round<len(spec['prizes'])
                rival=5 if spec.get('rival_shift') and w.round>=1 else max((3,)+tuple(w.last[1:]))
                bids=(int(t['root'].split(':')[1]),)+tuple(min(rival,b,8) for b in w.budgets[1:])
                after,_=settle(w,bids)
                if w.remaining:after=replace(after,prize=w.remaining[0],remaining=w.remaining[1:])
                if spec.get('revised_prize') and after.round==1:after=replace(after,prize=3);revised+=1
                worth=lambda s:s.scores[0]/30
                record=asdict(after)
            else:
                from reflex.combat import battle_from_record,battle_record,resolve,terminal,legal,potential
                w=battle_from_record(t['before']);assert not terminal(w) and w.units[0].hp>0
                keys=legal(w,3);shots=[k for k in keys if k.startswith('shoot:')]
                enemy='guard' if w.tick==0 else shots[0] if shots else 'reload' if 'reload' in keys else 'guard'
                after,_=resolve(w,{0:t['root'],3:enemy},r['actual_seed'])
                worth=lambda s:potential(s,0,'eliminate')/2
                record=battle_record(after)
                deaths+=int(w.units[0].hp>0 and after.units[0].hp==0)
            assert digest(record)==digest(t['after'])
            gain=worth(after)-worth(w)
            assert abs(t['flow']['objective']-gain)<1e-12
            total+=gain;transitions+=1
            for modeled in t['continuations']:
                assert modeled['seed'] in (prereg['model_combat_seeds'] if genre=='combat' else [0])
                assert modeled['actions'][0]==t['root']
                assert len(modeled['actions'])<=prereg['horizon']
        assert abs(total-r['score'])<1e-12
        assert r['actions']==[t['root'] for t in trace]
        if genre=='combat':assert terminal(after) or after.units[0].hp==0
        elif genre=='resources':assert terminal(after)
        else:assert after.round==len(spec['prizes'])
        key=genre+'.'+r['variant']
        item=secondary.setdefault(key,dict(episodes=0,wins=0,surviving=0,shortages=0,goal_progress_sum=0.,move_reversals=0))
        item['episodes']+=1
        if genre=='combat':
            from reflex.combat import victory
            item['wins']+=int(0 in victory(after));item['surviving']+=int(after.units[0].hp>0)
            # A diagnostic A->B->A return, NOT proof the moves were useless.
            for i in range(len(trace)-2):
                a=trace[i]['after']['units'][0];b=trace[i+2]['after']['units'][0]
                item['move_reversals']+=int((a['x'],a['y'])==(b['x'],b['y']) and r['actions'][i+1].startswith('move:') and r['actions'][i+2].startswith('move:'))
        elif genre=='resources':
            from reflex.resource_world import winners,goal_progress
            item['wins']+=int(0 in winners(after));item['shortages']+=after.empires[0].shortages
            item['goal_progress_sum']+=goal_progress(after.empires[0],after.routes)
    for row in result['summary']:
        selected=[r for r in rows if r['variant']=='persona' and r['spec']['genre']==row['genre']]
        better=worse=equal=changed=0
        for r in selected:
            base=next(b for b in rows if b['spec']==r['spec'] and b['profile']==r['profile'] and b['actual_seed']==r['actual_seed'] and b['variant']==row['reference'])
            assert base['trace'][0]['before']==r['trace'][0]['before']
            d=r['score']-base['score'];better+=d>1e-9;worse+=d< -1e-9;equal+=abs(d)<=1e-9
            changed+=r['actions']!=base['actions']
        assert [len(selected),int(better),int(worse),int(equal),int(changed)]==[row[k] for k in ('paired_episodes','better','worse','equal','changed')]
    return dict(episodes=len(rows),real_transitions_replayed=transitions,observed_revision_transitions=revised,
        owner_death_transitions=deaths,frozen_source_verified=True,summary_recomputed=True,secondary=secondary,
        trajectories_sha256=hashlib.sha256((folder/'trajectories.jsonl').read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder');args=parser.parse_args()
    print(json.dumps(audit(args.folder),ensure_ascii=False,indent=2))
