"""Forward actual-owner execution versus backward finite forecast, public only.

Synthetic terminal states are not population frequencies. Opponent probability
models are held fixed as a family, while their virtual public evidence updates.
The forward executor uses the ordinary real controller at EVERY owner return.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from reflex.board_models import ThanksPosition
from reflex.laboratory import PROFILES
from reflex.strong_search import PERSONA
from reflex.strong_table import decide
from reflex.supported_memory import SupportedPublicMemory


def points(cards,chips):
    hand=sorted(cards);return sum(c for i,c in enumerate(hand) if i==0 or hand[i-1]!=c-1)-chips


def evaluate(s,viewer,p,seed,memory,state,root,variant,root_decision,*,mode='adaptive'):
    m=copy.deepcopy(memory);st=copy.deepcopy(root_decision['next_state']);st['intent_action']=root
    st['age']=min(state['age']+1,1000000) if state and state['intent_action']==root else 0
    current=s;probability=1.;leaves=[];clock=0;returns=0
    while True:
        actor=current.turn
        if clock==0:take=float(root=='TAKE')
        elif actor==viewer:
            _,d,_=decide('no_thanks',current,viewer,p,seed,0,clock,m,st,mode,PERSONA,variant=variant)
            take=float(d['action_id']=='TAKE');st=d['next_state'];returns+=1
        else:take=1. if len(current.legal())==1 else m.predict(current,actor,adaptive=mode=='adaptive')['TAKE']
        if take:
            cards=[list(c) for c in current.cards];chips=list(current.chips)
            cards[actor].append(current.card);chips[actor]+=current.pot
            terminal=tuple(points(c,h) for c,h in zip(cards,chips));leaves.append((probability*take,terminal))
        probability*=1-take
        if probability==0:break
        assert 'PASS' in current.legal()
        if actor!=viewer and mode=='adaptive':m.observe(current,actor,'PASS',f'encounter-0-tick-{clock}-actor-{actor}')
        current=current.play('PASS');clock+=1
        assert clock<=44
    assert abs(sum(q for q,_ in leaves)-1)<1e-10
    credit=sum(q*(v[viewer]==min(v))/v.count(min(v)) for q,v in leaves)
    score=sum(q*v[viewer] for q,v in leaves)
    progress=sum(q*(.5+np.arctan((min(v[a] for a in range(4) if a!=viewer)-v[viewer])/35)/np.pi) for q,v in leaves)
    return dict(credit=credit,score=score,progress=float(progress),owner_returns=returns,leaf_count=len(leaves))


def run(root,states=64,mode='adaptive'):
    if mode not in ('adaptive','planned'):raise ValueError('supported controller mode required')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    plan=dict(version='policy-consistent-settlement-exposure-v2',states=states,seed=72419,mode=mode,
        variants=['certified_expiry','settlement'],profiles=[p['id'] for p in PROFILES],budget=asdict(PERSONA),
        construction='23 held plus one current card; total chip/pot ledger 44; each owner and next seat >=1, no immediate forced terminal certificate',
        primary='forecast must equal separate forward execution of the SAME future owner controller on every public branch',
        secondary='legacy continuation mismatch; model credit regret, raw score and fixed identity',
        limitations=['synthetic legal ledger states, reachability/frequency unproven','exactness conditional on imperfect public opponent models','no humans measured'],
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((Path(__file__).resolve().parents[1]/'reflex').glob('*.py'))})
    path=root/'preregister.json';assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    rng=np.random.default_rng(plan['seed']);rows=[]
    for case in range(states):
        seen=tuple(sorted(map(int,rng.permutation(np.arange(3,36))[:24])));card=int(rng.choice(seen))
        held=[[],[],[],[]]
        for c,a in zip([c for c in seen if c!=card],rng.integers(4,size=23)):held[int(a)].append(c)
        viewer=case%4;chips=list(map(int,rng.multinomial(40,[.25]*4)));chips[viewer]+=1;chips[(viewer+1)%4]+=1
        pot=44-sum(chips)
        s=ThanksPosition(tuple(tuple(h) for h in held),tuple(chips),viewer,card,pot,seen,0,(0,)*4)
        for p in PROFILES:
            memory=SupportedPublicMemory('no_thanks',viewer);records=[]
            for variant in plan['variants']:
                before=memory.record();c,d,stats=decide('no_thanks',s,viewer,p,7300+case,0,0,memory,None,mode,PERSONA,variant=variant)
                assert before==memory.record();assert c['personality']==dict(zip(('openness','conscientiousness','extraversion','agreeableness','neuroticism'),p['traits']))
                actual={a:evaluate(s,viewer,p,7300+case,memory,None,a,variant,d,mode=mode) for a in s.legal()}
                predictions=stats['actions'];error={a:{k:float(predictions[a][field]-actual[a][k]) for k,field in (('credit','win_share'),('score','mean_score'),('progress','goal_progress'))} for a in s.legal()}
                if variant=='settlement':assert all(abs(v)<1e-10 for e in error.values() for v in e.values()),error
                chosen=d['action_id'];regret=max(v['credit'] for v in actual.values())-actual[chosen]['credit']
                if variant=='settlement':assert regret<=.12+1e-10
                records.append(dict(variant=variant,action=chosen,predicted=predictions,forward_actual_owner=actual,error=error,model_credit_regret=regret))
            rows.append(dict(case=case,profile=p['id'],public_state=asdict(s),controllers=records))
        print('states',case+1,'conditions',len(rows),flush=True)
    summary={}
    for variant in plan['variants']:
        selected=[c for r in rows for c in r['controllers'] if c['variant']==variant]
        errors=[v for c in selected for e in c['error'].values() for k,v in e.items() if k=='credit']
        summary[variant]=dict(conditions=len(selected),root_forecasts=len(errors),max_absolute_credit_error=max(map(abs,errors)),
            mean_absolute_credit_error=float(np.mean(np.abs(errors))),mean_model_credit_regret=float(np.mean([r['model_credit_regret'] for r in selected])),
            regret_above_declared_tolerance=sum(r['model_credit_regret']>.12+1e-10 for r in selected),
            future_owner_returns=sum(v['owner_returns'] for c in selected for v in c['forward_actual_owner'].values()))
    result=dict(plan=plan,rows=rows,summary=summary)
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--states',type=int,default=64)
    p.add_argument('--mode',choices=('adaptive','planned'),default='adaptive');a=p.parse_args();run(a.root,a.states,a.mode)
