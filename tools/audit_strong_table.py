"""Independent game/ledger reconstruction, plus public learning and policy replay.

Rule checks do not call the game resolvers. Learning replay reuses the learner,
and is an integration check, not an independent proof of inference quality.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import subprocess
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest, TRAITS, VALUES
from reflex.laboratory import PROFILES
from reflex.goofspiel import Position
from reflex.board_models import ThanksPosition
from reflex.strong_search import PublicMemory, SearchBudget
from reflex.strong_table import play, summarize
from reflex.supported_memory import SupportedPublicMemory,ValidatedPublicMemory,GuardedPublicMemory
from reflex.tabletop_trials import score


def normalized(x):return json.loads(json.dumps(x,ensure_ascii=False))


def card_score(cards,chips):
    total=0;previous=None
    for c in sorted(cards):
        if previous is None or previous+1!=c:total+=c
        previous=c
    return total-chips


def audit(root,replay=False,source_ref=None,replay_games=None):
    global PublicMemory,SupportedPublicMemory,ValidatedPublicMemory,GuardedPublicMemory,SearchBudget,play,summarize
    root=Path(root);evaluation=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    plan=json.loads((root/'preregister.json').read_text(encoding='utf-8'));assert plan==evaluation['plan']
    base=Path(__file__).resolve().parents[1]
    frozen={}
    for name,sha in plan['sources'].items():
        normalized_name=name.replace('\\','/');data=(base/normalized_name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=sha:
            assert source_ref is not None,'use --source-ref for the registered implementation: '+name
            data=subprocess.check_output(['git','show',f'{source_ref}:{normalized_name}'],cwd=base)
            assert normalized_name in ('reflex/opponent_beliefs.py','reflex/goal_guard.py','reflex/goal_progress.py','reflex/strong_search.py','reflex/prediction_support.py','reflex/supported_memory.py','reflex/loop_predictor.py','reflex/finite_continuation.py','reflex/settlement_solver.py','reflex/strong_table.py'),'unexpected runtime dependency change'
            frozen[normalized_name]=data
        assert hashlib.sha256(data).hexdigest()==sha,name
    # The experiment modules and feedback learner may later fix connections. Replay
    # their actual immutable implementation, with every shared dependency still
    # checked against the registration. No changes to the working tree.
    if frozen:
        registered_names={name.replace('\\','/') for name in plan['sources']}
        for name in ('reflex/opponent_beliefs.py','reflex/goal_guard.py','reflex/goal_progress.py','reflex/strong_search.py','reflex/prediction_support.py','reflex/supported_memory.py','reflex/loop_predictor.py','reflex/finite_continuation.py','reflex/settlement_solver.py','reflex/strong_table.py'):
            if name not in registered_names:continue
            data=frozen.get(name,(base/name).read_bytes());module_name=name[:-3].replace('/','.')
            module=types.ModuleType(module_name);module.__package__='reflex';module.__file__=str(base/name)
            sys.modules[module_name]=module;exec(compile(data,str(base/name),'exec'),module.__dict__)
        PublicMemory=sys.modules['reflex.strong_search'].PublicMemory;SearchBudget=sys.modules['reflex.strong_search'].SearchBudget
        if 'reflex/supported_memory.py' in registered_names:
            memory_module=sys.modules['reflex.supported_memory'];SupportedPublicMemory=memory_module.SupportedPublicMemory
            if hasattr(memory_module,'ValidatedPublicMemory'):ValidatedPublicMemory=memory_module.ValidatedPublicMemory
            if hasattr(memory_module,'GuardedPublicMemory'):GuardedPublicMemory=memory_module.GuardedPublicMemory
        play=sys.modules['reflex.strong_table'].play;summarize=sys.modules['reflex.strong_table'].summarize
    trace=root/'trajectories.jsonl';assert hashlib.sha256(trace.read_bytes()).hexdigest()==evaluation['trajectory_sha256']
    def new_memories(game,bench):
        factory={'global':PublicMemory,'supported':SupportedPublicMemory,'validated':ValidatedPublicMemory,'guarded':GuardedPublicMemory}[plan.get('memory_kind','global')]
        return [(factory if a!=bench else PublicMemory)(game,a) for a in range(4)]
    indexed={(r['game'],r['seed'],r['mode'],r['encounter']):r for r in evaluation['matches']}
    assert len(indexed)==len(plan['games'])*len(plan['seeds'])*len(plan['modes'])*plan['encounters']
    counts=Counter();seen=set();previous=None;current=None;memories=None;running=Counter();failures=[0]*4
    contexts={};losses=defaultdict(list);observed_decks={};deck=[];row_hashes={};hasher=None
    behavior=defaultdict(Counter)
    profile_by_id={p['id']:p for p in PROFILES}
    def finish():
        nonlocal previous,current,deck
        if current is None:return
        r=indexed[current];game=current[0];assert running['steps']==r['steps']
        assert failures==r['failures'];assert running['changes']==sum(r['reading_changes']);assert running['guards']==sum(r['guards'])
        if game=='goofspiel':
            assert previous['round']==13;sc=previous['scores'];assert all(not h for h in previous['hands'])
        else:
            assert previous['card'] is None and previous['remaining']==0
            assert sum(map(len,previous['cards']))==24
            sc=[card_score(h,c) for h,c in zip(previous['cards'],previous['chips'])]
            world_key=(game,r['seed'],r['encounter'])
            if world_key in observed_decks:assert observed_decks[world_key]==deck
            else:observed_decks[world_key]=list(deck)
        assert sc==r['scores'];top=(max if game=='goofspiel' else min)(sc);w=[i for i,x in enumerate(sc) if x==top]
        assert previous==r['final']
        assert r['credits']==[1/len(w) if a in w else 0. for a in range(4)]
        assert r['beliefs']==normalized([m.record() for m in memories])
        row_hashes[current]=hasher.hexdigest();seen.add(current)
    with trace.open(encoding='utf-8') as f:
        for line in f:
            t=json.loads(line);key=(t['game'],t['seed'],t['mode'],t['encounter'])
            if key!=current:
                old=current;finish();current=key;previous=None;running=Counter();failures=[0]*4;deck=[]
                assert key not in seen
                if old is None or key[:3]!=old[:3]:memories=new_memories(key[0],indexed[key]['benchmark'])
                hasher=hashlib.sha256()
            hasher.update((digest(t)+'\n').encode());r=indexed[key];b=t['before'];a=t['after'];game=key[0]
            assert t['benchmark']==r['benchmark'];assert t['tick']==running['steps'];running['steps']+=1;counts['steps']+=1
            if previous is not None:
                if game=='no_thanks' and previous['card'] is None:
                    assert b['card'] not in previous['seen'] and 3<=b['card']<=35;deck.append(b['card'])
                    assert b['seen']==sorted(previous['seen']+[b['card']]);assert b['remaining']==previous['remaining']-1
                    assert all(b[k]==previous[k] for k in ('chips','cards','turn','pot','payments'))
                else:assert b==previous
            elif game=='no_thanks':
                assert b['chips']==[11]*4 and b['pot']==0 and b['cards']==[[]]*4 and b['remaining']==23
                deck.append(b['card']);assert b['turn']==r['encounter']%4
            else:
                assert b['scores']==[0]*4 and b['hands']==[list(range(1,14))]*4
                assert sorted(b['prizes'])==list(range(1,14))
                world_key=(game,r['seed'],r['encounter'])
                if world_key in observed_decks:assert observed_decks[world_key]==b['prizes']
                else:observed_decks[world_key]=b['prizes']
            if game=='goofspiel':
                assert set(t['moves'])=={'0','1','2','3'}
                bids=[int(t['moves'][str(i)].split(':')[1]) for i in range(4)];high=max(bids)
                winners=[i for i,x in enumerate(bids) if x==high];sc=b['scores'].copy()
                if len(winners)==1:sc[winners[0]]+=b['prizes'][b['round']]
                assert a['scores']==sc;assert a['round']==b['round']+1
                assert a['discarded']==b['discarded']+(b['prizes'][b['round']] if len(winners)>1 else 0)
                assert sum(a['scores'])+a['discarded']==sum(b['prizes'][:a['round']])
                for i,x in enumerate(bids):
                    assert x in b['hands'][i];assert a['hands'][i]==[v for v in b['hands'][i] if v!=x]
                    failures[i]+=int(x==high and len(winners)>1)
                observed=Position(tuple(tuple(h) for h in b['hands']),tuple(b['scores']),tuple(b['prizes']),b['round'],b['discarded'])
            else:
                actor=b['turn'];assert set(t['moves'])=={str(actor)};move=t['moves'][str(actor)]
                chips=b['chips'].copy();cards=[h.copy() for h in b['cards']];payments=b['payments'].copy()
                assert sum(chips)+b['pot']==44;assert len(b['seen'])==24-b['remaining']
                if move=='PASS':
                    assert chips[actor]>0;chips[actor]-=1;payments[actor]+=1
                    assert a['turn']==(actor+1)%4 and a['pot']==b['pot']+1 and a['card']==b['card']
                else:
                    assert move=='TAKE';chips[actor]+=b['pot'];cards[actor]=sorted(cards[actor]+[b['card']])
                    assert a['turn']==actor and a['pot']==0 and a['card'] is None
                failures[actor]+=int(b['chips'][actor]==0)
                assert a['chips']==chips and a['cards']==cards and a['payments']==payments
                assert a['seen']==b['seen'] and a['remaining']==b['remaining'];assert sum(chips)+a['pot']==44
                counts['chip_ledger_steps']+=1
                observed=ThanksPosition(tuple(tuple(h) for h in b['cards']),tuple(b['chips']),actor,b['card'],b['pot'],tuple(b['seen']),b['remaining'],tuple(b['payments']))
            assert set(t['decisions'])==set(t['moves'])
            for actor_text,d in t['decisions'].items():
                actor=int(actor_text);chosen=t['moves'][actor_text];assert d['action']==chosen;counts['decisions']+=1
                st=d.get('search')
                if st and 'actions' in st:
                    prefix=f'{game}_{d["method"]}'
                    counts[prefix+'_forecasts']+=1
                    if len(st['actions'])>1:
                        counts[prefix+'_multiple_roots']+=1;values=[q['win_share'] for q in st['actions'].values()]
                        counts[prefix+'_all_roots_zero_share']+=max(values)==0
                        counts[prefix+'_all_roots_equal_share']+=max(values)==min(values)
                if actor==r['benchmark']:
                    assert d['method']=='objective-search';assert st['sample_count']==plan['strong']['validate']
                    names=list(st['actions']);sign=1 if game=='goofspiel' else -1
                    expected=max(range(len(names)),key=lambda i:(st['actions'][names[i]]['win_share'],sign*st['actions'][names[i]]['mean_score'],-i))
                    assert chosen==names[expected];counts['benchmark_decisions']+=1;continue
                c=d['context'];p=profile_by_id[r['profiles'][actor_text]]
                bucket=behavior[(game,r['mode'],p['id'])];bucket['decisions']+=1
                if game=='goofspiel' and len(b['hands'][actor])>1:
                    bid=int(chosen.split(':')[1]);bucket['rank_sum']+=b['hands'][actor].index(bid)/(len(b['hands'][actor])-1)
                    bucket['nonforced_bids']+=1
                elif game=='no_thanks':
                    bucket['takes']+=chosen=='TAKE';bucket['passes']+=chosen=='PASS'
                    added=card_score(b['cards'][actor]+[b['card']],0)-card_score(b['cards'][actor],0)-b['pot']
                    bucket['voluntary_costly_takes']+=chosen=='TAKE' and b['chips'][actor]>0 and added>1
                assert c['personality']==dict(zip(TRAITS,p['traits']))
                assert c['values']=={k:float(p['values'].get(k,0)) for k in VALUES}
                assert c['scope']['npc']==f'player-{actor}' and c['seed']==r['seed'] and c['tick']==t['tick']
                assert c['opponent'] is None
                assert all(row['values'].get('benevolence',0)==row['values'].get('universalism',0)==0 for act in c['actions'] for row in act['outcomes'])
                if game=='goofspiel':
                    assert c['facts']['scores']==str(b['scores'])
                    for j,h in enumerate(b['hands']):assert c['facts'][f'public_hand_{j}']==str(h)
                else:
                    assert c['facts']['public_counter_ledger']==str(tuple(b['chips']));assert c['facts']['seen']==str(tuple(b['seen']))
                    expires=(b['remaining']==0 and (plan.get('variant')=='horizon_progress' or
                        plan.get('variant') in ('certified_expiry','settlement') and b['chips'][(actor+1)%4]==0))
                    if expires:
                        assert 'expired_proxies' in c['facts']
                        assert c['needs']['safety']['enabled'] is False and c['needs']['safety']['deficit'] is None
                        assert c['state']['primary_need']!='safety'
                        for act in c['actions']:
                            for outcome in act['outcomes']:
                                assert outcome['needs']['safety']==outcome['values']['security']==outcome['style']['neuroticism']==0
                        counts['no_thanks_real_expiry_choices']+=1
                        if plan.get('variant') in ('certified_expiry','settlement'):assert 'expiry_certificate' in c['facts']
                    else:assert 'expired_proxies' not in c['facts']
                filtered=normalized(c)
                if d['method']!='reflex':
                    guard=st['guard'];assert chosen in guard['allowed'];running['guards']+=guard['guard_changed']
                    finite=game=='no_thanks' and plan.get('variant')=='settlement' and b['remaining']==0
                    assert st['sample_count']==(0 if finite else plan['persona']['validate'])
                    if finite:
                        assert st['finite_chain_nodes']<=45 and all(q['standard_error']==0 for q in st['actions'].values())
                        counts['finite_continuation_choices']+=1
                    filtered['actions']=[act for act in filtered['actions'] if act['id'] in guard['allowed']]
                    signal=guard.get('signal','terminal_success')
                    if signal=='goal_progress_on_constant_success':
                        assert all(q['win_share']==0 for q in st['actions'].values())
                        assert guard['primary_constant']==0
                        counts[f'{game}_progress_fallbacks']+=1
                        means={name:q['goal_progress'] for name,q in st['actions'].items()}
                    else:
                        assert signal=='terminal_success'
                        means={name:q['win_share'] for name,q in st['actions'].items()}
                    best=max(means.values())
                    for name,bound in guard['bounds'].items():
                        assert math.isclose(best-means[name],bound['estimated_regret'],abs_tol=1e-12)
                        assert math.isclose(bound['lower_bound'],bound['estimated_regret']-2*bound['sampling_error'],abs_tol=1e-12)
                        assert (bound['lower_bound']<=.12+(1e-12 if finite else 0))==(name in guard['allowed'])
                    if d['method']=='adaptive':running['changes']+=st['frozen_action']!=chosen
                expected=score(filtered)[0];assert expected['action_id']==chosen and expected['next_state']==d['next_state']
                counts['personality_selections_replayed']+=1
                # Diversity counterfactuals share the SAME actual public context.
                if r['mode']=='adaptive':
                    choices=[]
                    for alternate in PROFILES:
                        counter=normalized(filtered);counter['personality']=dict(zip(TRAITS,alternate['traits']))
                        counter['values']={k:float(alternate['values'].get(k,0)) for k in VALUES}
                        choices.append(score(counter)[0]['action_id'])
                    counts[f'{game}_same_state_persona_probes']+=1
                    counts[f'{game}_different_persona_choices']+=len(set(choices))>1
            expected_learning=[]
            actors=range(4) if game=='goofspiel' else (b['turn'],)
            for observer in range(4):
                if r['mode']!='adaptive' and observer!=r['benchmark']:continue
                for actor in actors:
                    if actor==observer:continue
                    rec=memories[observer].observe(observed,actor,t['moves'][str(actor)],f'encounter-{r["encounter"]}-tick-{t["tick"]}-actor-{actor}')
                    expected_learning.append(dict(observer=observer,actor=actor,**rec))
                    if not rec['forced']:
                        who='benchmark' if observer==r['benchmark'] else ('npc-reads-benchmark' if actor==r['benchmark'] else 'npc-reads-npc')
                        losses[(game,r['mode'],who)].append((rec['mixture_log_loss'],rec['uniform_log_loss']))
            # Game implementation emits Goof observations by observer/actor,
            # and No Thanks by observer. These orders are identical here.
            assert normalized(expected_learning)==t['learning'];counts['learning_records_replayed']+=len(expected_learning)
            previous=dict(a)
            if game=='goofspiel':previous['prizes']=b['prizes']
    finish();assert seen==set(indexed)
    assert normalized(summarize(evaluation['matches'],plan['encounters']))==evaluation['summary']
    replayed=0
    if replay:
        seed=plan['seeds'][0];bench=seed%4;others=[p for i,p in enumerate(PROFILES) if i!=(seed//4)%4]
        roster={a:p for a,p in zip([a for a in range(4) if a!=bench],others)}
        for game in plan['games']:
            if replay_games is not None and game not in replay_games:continue
            for mode in plan['modes']:
                memories=new_memories(game,bench)
                for encounter in range(plan['encounters']):
                    h=hashlib.sha256()
                    def emit(row):h.update((digest(normalized(row))+'\n').encode())
                    actual,_=play(game,seed,encounter,bench,roster,mode,memories,
                        strong=SearchBudget(**plan['strong']),persona=SearchBudget(**plan['persona']),emit=emit,
                        **({'variant':plan['variant']} if 'variant' in plan else {}))
                    key=(game,seed,mode,encounter);assert normalized(actual)==indexed[key]
                    assert h.hexdigest()==row_hashes[key];replayed+=1
    proof=dict(matches=len(indexed),**counts,all_finished=True,source_hashes_checked=len(plan['sources']),
        source_reference=source_ref,immutable_modules_replayed=sorted(frozen),
        trajectory_sha256=evaluation['trajectory_sha256'],exact_search_replayed_matches=replayed,
        replay_scope=dict(series=plan['seeds'][0],games=(list(replay_games) if replay_games is not None else plan['games']),modes=plan['modes'],encounters=plan['encounters']) if replay else None,
        learning_replay='same algorithm from past public acts; clock/isolation integration check',
        fixed_personality_verified=True,one_objective_cpu_verified=True,matched_worlds_verified=True,
        predictive_losses=[dict(game=k[0],mode=k[1],observer_kind=k[2],observations=len(v),
            mean_log_loss=sum(x for x,y in v)/len(v),uniform_log_loss=sum(y for x,y in v)/len(v)) for k,v in losses.items()],
        behavior=[dict(game=k[0],mode=k[1],profile=k[2],**dict(v)) for k,v in behavior.items()],
        enjoyment_measured=False,optimal_intelligence_proved=False)
    (root/'audit.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--replay',action='store_true')
    p.add_argument('--source-ref');args=p.parse_args()
    print(json.dumps(audit(args.root,args.replay,args.source_ref),ensure_ascii=False,indent=2))
