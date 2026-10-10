"""Independent public rule/ledger replay and paired learning-series analysis."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.board_models import ThanksPosition
from reflex.core import TRAITS, VALUES
from reflex.laboratory import PROFILES
from reflex.strong_search import PublicMemory
from reflex.tabletop_trials import score


def normalized(v):return json.loads(json.dumps(v))


def points(cards,chips):
    return sum(c for c in cards if c-1 not in cards)-chips


def audit(root,e):
    plan=e['plan'];counts=Counter();fingerprints={};worlds={}
    for name,sha in plan['sources'].items():
        assert hashlib.sha256((root/'_source'/name).read_bytes()).hexdigest()==sha,name
    assert hashlib.sha256((root/'_source/tools/run_policy_rollout.py').read_bytes()).hexdigest()==plan['script_sha256']
    index={(r['seed'],r['condition'],r['encounter']):r for r in e['matches']}
    arms=plan.get('arms',[dict(label=str(b),band=b,future_seed='resampled') for b in plan['bands']])
    labels=['baseline',*[a['label'] for a in arms]];by_label={a['label']:a for a in arms}
    assert len(index)==len(e['matches'])==len(plan['seeds'])*len(labels)*plan['encounters']
    profiles={p['id']:p for p in PROFILES}
    for seed in plan['seeds']:
        for label in labels:
            path=root/f'{seed}-{label}.jsonl';fingerprints[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            memories=[PublicMemory('no_thanks',a) for a in range(4)]
            current=None;previous=None;reveals=[];ticks=0;exposed=False
            def finish():
                if current is None:return
                r=index[seed,label,current]
                assert previous==r['final'] and previous['card'] is None and previous['remaining']==0
                assert sum(map(len,previous['cards']))==24 and len(reveals)==24
                scores=[points(h,c) for h,c in zip(previous['cards'],previous['chips'])]
                assert scores==r['scores'];winners=[a for a,x in enumerate(scores) if x==min(scores)]
                assert r['credits']==[1/len(winners) if a in winners else 0. for a in range(4)]
                assert r['beliefs']==normalized([m.record() for m in memories]) and ticks==r['steps']
                k=(seed,current)
                if k in worlds:assert worlds[k]==reveals
                else:worlds[k]=reveals[:]
                if label!='baseline':assert exposed==(r['intervention'] is not None)
                counts['games']+=1
            for line in path.open(encoding='utf-8'):
                t=json.loads(line);enc=t['encounter']
                if current!=enc:
                    finish();current=enc;previous=None;reveals=[];ticks=0;exposed=False
                r=index[seed,label,enc];b=t['before'];after=t['after'];actor=b['turn'];move=t['moves'][str(actor)]
                assert t['tick']==ticks;ticks+=1;counts['public_actions']+=1
                assert set(t['moves'])==set(t['decisions'])=={str(actor)}
                assert t['benchmark']==r['benchmark']
                if previous is None:
                    assert b['cards']==[[]]*4 and b['chips']==[11]*4 and b['pot']==0 and b['remaining']==23 and b['turn']==enc%4
                    reveals.append(b['card'])
                elif previous['card'] is None:
                    assert b['card'] not in previous['seen'] and b['seen']==sorted(previous['seen']+[b['card']])
                    assert b['remaining']==previous['remaining']-1
                    for key in ('chips','cards','turn','pot','payments'):assert b[key]==previous[key]
                    reveals.append(b['card'])
                else:assert b==previous
                chips=b['chips'][:];cards=[h[:] for h in b['cards']];paid=b['payments'][:]
                assert sum(chips)+b['pot']==44 and len(b['seen'])==24-b['remaining']
                if move=='PASS':
                    assert chips[actor]>0;chips[actor]-=1;paid[actor]+=1
                    assert after['turn']==(actor+1)%4 and after['pot']==b['pot']+1 and after['card']==b['card']
                else:
                    assert move=='TAKE';chips[actor]+=b['pot'];cards[actor]=sorted(cards[actor]+[b['card']])
                    assert after['turn']==actor and after['pot']==0 and after['card'] is None
                assert after['cards']==cards and after['chips']==chips and after['payments']==paid
                assert after['seen']==b['seen'] and after['remaining']==b['remaining'] and sum(chips)+after['pot']==44
                observed=ThanksPosition(tuple(map(tuple,b['cards'])),tuple(b['chips']),actor,b['card'],b['pot'],tuple(b['seen']),b['remaining'],tuple(b['payments']))
                actual_learning=[]
                for viewer in range(4):
                    if viewer!=actor:
                        rec=memories[viewer].observe(observed,actor,move,f'encounter-{enc}-tick-{t["tick"]}-actor-{actor}')
                        actual_learning.append(dict(observer=viewer,actor=actor,**rec))
                assert normalized(actual_learning)==t['learning'];counts['public_learning_records']+=len(actual_learning)
                d=t['decisions'][str(actor)];assert d['action']==move;st=d['search']
                if actor==r['benchmark']:
                    assert d['method']=='objective-search' and st['sample_count']==plan['strong']['validate']
                    names=list(st['actions']);best=max(range(len(names)),key=lambda i:(st['actions'][names[i]]['win_share'],-st['actions'][names[i]]['mean_score'],-i))
                    assert names[best]==move;counts['unchanged_cpu_decisions']+=1
                else:
                    p=profiles[r['profiles'][str(actor)]];c=d['context']
                    assert c['personality']==dict(zip(TRAITS,p['traits'])) and c['values']=={k:float(p['values'].get(k,0)) for k in VALUES}
                    assert c['scope']['npc']==f'player-{actor}' and c['seed']==seed and c['tick']==t['tick'] and c['opponent'] is None
                    assert c['facts']['seen']==str(tuple(b['seen'])) and c['facts']['public_counter_ledger']==str(tuple(b['chips']))
                    assert all(v['values']['benevolence']==v['values']['universalism']==0 for a in c['actions'] for v in a['outcomes'])
                    rc=normalized(c);rc['actions']=[a for a in rc['actions'] if a['id'] in st['guard']['allowed']]
                    replay,_=score(rc);assert replay['action_id']==move and replay['next_state']==d['next_state']
                    # Initial discovery stored its hash before explanation facts.
                    # Current producer fixes this; disclose rather than rewrite it.
                    if 'arms' in plan:assert replay=={k:v for k,v in d.items() if k in replay}
                    counts['fixed_persona_policy_replays']+=1
                    roll=st.get('policy_rollout')
                    eligible=label!='baseline' and not exposed and actor==r['target'] and b['remaining']<=by_label[label]['band'] and len(observed.legal())>1
                    if eligible:
                        assert roll is not None;exposed=True
                        v=r['intervention'];assert v['tick']==t['tick'] and v['remaining']==b['remaining'] and v['public_state']==b
                        assert v['stats']==roll and v['action']==move
                    else:assert roll is None
                    if roll:
                        assert roll['used'] and roll['completed_samples']==plan['samples']
                        assert all(q['samples']==plan['samples'] for q in roll['actions'].values())
                        assert roll['owner_budget']==plan['persona'] and roll['discarded_terminal_samples']==0
                        counts['isolated_real_interventions']+=1;counts['changed_roots']+=roll['changed']
                        counts['hypothetical_owner_searches']+=roll['owner_searches']
                        counts['paired_hypothetical_terminal_branches']+=2*roll['completed_samples']
                previous=after
            finish()
    return dict(counts=dict(counts),trajectory_sha256=fingerprints,source_files=len(plan['sources']),
        scope='rules independently reconstructed; public learning and Policy replays are integration checks, not independent inference-quality proofs')


def analyze(root):
    root=Path(root);e=json.loads((root/'evaluation.json').read_text(encoding='utf-8'));plan=e['plan']
    assert plan==json.loads((root/'preregister.json').read_text(encoding='utf-8'))
    proof=audit(root,e);index={(r['seed'],r['condition'],r['encounter']):r for r in e['matches']}
    summary=[];contrasts=[]
    arms=plan.get('arms',[dict(label=str(b),band=b,future_seed='resampled') for b in plan['bands']])
    for arm in arms:
        label=arm['label'];band=arm['band'];rows=[r for r in e['matches'] if r['condition']==label]
        summary.append(dict(**arm,interventions=sum(r['intervention'] is not None for r in rows),
            changed=sum(r['intervention'] is not None and r['intervention']['stats']['changed'] for r in rows),
            credit=float(np.mean([r['credits'][r['target']] for r in rows])),
            score=float(np.mean([r['scores'][r['target']] for r in rows])),
            owner_searches=sum(r['intervention']['stats']['owner_searches'] for r in rows if r['intervention'])))
        for profile in ('each_target','growth','steady','care','ego'):
            series=[];score_series=[];better=same=worse=0
            for seed in plan['seeds']:
                delta=[];sd=[]
                for enc in range(plan['encounters']):
                    a=index[seed,'baseline',enc];b=index[seed,label,enc];target=a['target']
                    assert a['profiles']==b['profiles'] and a['benchmark']==b['benchmark'] and a['target']==b['target']
                    if profile!='each_target' and a['target_profile']!=profile:continue
                    v=b['credits'][target]-a['credits'][target];delta.append(v);sd.append(a['scores'][target]-b['scores'][target])
                    better+=v>0;same+=v==0;worse+=v<0
                if delta:series.append(float(np.mean(delta)));score_series.append(float(np.mean(sd)))
            if not series:continue
            values=np.array(series);rng=np.random.default_rng(910071)
            boot=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
            contrasts.append(dict(**arm,reference='baseline',profile=profile,series=len(values),mean_credit_difference=float(values.mean()),
                paired_series_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),mean_own_score_improvement=float(np.mean(score_series)),
                better=int(better),same=int(same),worse=int(worse),series_differences=series))
    result=dict(audit=proof,summary=summary,contrasts=contrasts,
        evaluation_sha256=hashlib.sha256((root/'evaluation.json').read_bytes()).hexdigest(),
        limitations=['learning series, not turns or simulated branches, are independent units','bootstrap intervals lack multiplicity correction',
                     'post-intervention public learning and later games may differ','known game, known controller families, no optimal-CPU/human-level proof'])
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(audit=proof['counts'],summary=summary,contrasts=[{k:v for k,v in r.items() if k!='series_differences'} for r in contrasts]),indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();analyze(a.root)
