"""Reconstruct existing-game rules and public ledgers without policy/resolver calls."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.goofspiel import Position
from reflex.board_models import ThanksPosition
from reflex.goofspiel_beliefs import PublicBidBeliefs
from reflex.opponent_beliefs import HypothesisTracker
from reflex.tabletop_trials import thanks_hypotheses, play


def card_score(cards,chips):
    total=0
    for c in cards:
        if c-1 not in cards:total+=c
    return total-chips


def check(root,replay=False):
    root=Path(root);result=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    registration=json.loads((root/'preregister.json').read_text(encoding='utf-8'))
    assert registration==result['plan']
    source=Path(__file__).resolve().parents[1]/'reflex'
    for name,sha in registration['source_hashes'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,name
    tracefile=root/'trajectories.jsonl'
    assert hashlib.sha256(tracefile.read_bytes()).hexdigest()==result['trace_sha256']
    runs=[];decisions=0;steps=0;focal=0;ledgers=0;public_bid_observations=0;prediction_checks=0
    replayed=0;flat={}
    with tracefile.open(encoding='utf-8') as f:
        for line in f:
            world=json.loads(line);r=world['run'];game=r['game'];previous=None;counts=Counter()
            if replay and r['seed']==registration['seeds'][0] and r['profile_index']==0:
                generated,trace,predictions=play(game,r['seed'],r['profile_index'],r['mode'])
                assert json.loads(json.dumps(dict(run=generated,trace=trace,predictions=predictions),ensure_ascii=False))==world
                replayed+=1
            ledger=[11]*4;seen=[];cards=[[] for _ in range(4)]
            public=PublicBidBeliefs(4,r['seat']) if game=='goofspiel' else tuple(HypothesisTracker(('uniform','tactical','reserve','accept')) for _ in range(4))
            expected_predictions=[]
            for t in world['trace']:
                b=t['before'];a=t['after'];actor=b.get('turn');steps+=1
                if previous is not None:
                    if game=='no_thanks' and previous['card'] is None:
                        assert b['card'] not in previous['seen'] and 3<=b['card']<=35
                        assert b['seen']==sorted(previous['seen']+[b['card']])
                        assert b['remaining']==previous['remaining']-1
                        assert all(b[k]==previous[k] for k in ('cards','chips','turn','pot','payments'))
                    else:assert b==previous
                if game=='goofspiel':
                    bids=t['actual'];assert len(bids)==4
                    assert b['round']==t['tick'] and a['round']==b['round']+1
                    assert b['prizes']==list(range(1,14))[::1 if r['seed']%2==0 else -1]
                    assert a['prizes']==b['prizes']
                    prize=b['prizes'][b['round']];high=max(bids);w=[i for i,x in enumerate(bids) if x==high]
                    scores=b['scores'].copy()
                    if len(w)==1:scores[w[0]]+=prize
                    assert a['scores']==scores
                    assert a['discarded']==b['discarded']+(prize if len(w)>1 else 0)
                    assert sum(a['scores'])+a['discarded']==sum(a['prizes'][:a['round']])
                    for i,x in enumerate(bids):
                        assert x in b['hands'][i]
                        assert a['hands'][i]==[v for v in b['hands'][i] if v!=x]
                    counts['tied_prizes']+=len(w)>1
                    assert len(t['decisions'])==4
                else:
                    assert b['chips']==ledger and b['cards']==cards
                    assert sum(b['chips'])+b['pot']==44 and len(b['seen'])==24-b['remaining']
                    chosen=t['actual'];assert chosen in ('TAKE','PASS')
                    assert len(t['decisions'])==1 and t['decisions'][0]['actor']==actor
                    payments=b['payments'].copy()
                    if chosen=='PASS':
                        assert ledger[actor]>0;ledger[actor]-=1;payments[actor]+=1
                        assert a['turn']==(actor+1)%4 and a['pot']==b['pot']+1 and a['card']==b['card']
                    else:
                        ledger[actor]+=b['pot'];cards[actor]=sorted(cards[actor]+[b['card']])
                        assert a['turn']==actor and a['pot']==0 and a['card'] is None
                    assert a['chips']==ledger and a['cards']==cards and a['payments']==payments
                    assert a['seen']==b['seen'] and a['remaining']==b['remaining']
                    assert sum(a['chips'])+a['pot']==44;ledgers+=1
                    if actor==r['seat']:
                        counts['forced_take']+=b['chips'][actor]==0
                        counts['passes']+=chosen=='PASS'
                        added=card_score(b['cards'][actor]+[b['card']],0)-card_score(b['cards'][actor],0)-b['pot']
                        counts['voluntary_costly_take']+=chosen=='TAKE' and b['chips'][actor]>0 and added>1
                for d in t['decisions']:
                    c=d['context'];decision=d['decision'];stats=d['stats'];i=d['actor'];decisions+=1;counts['decisions']+=1
                    assert c['scope']['npc']==f'player-{i}' and c['tick']==t['tick']
                    assert hashlib.sha256(json.dumps(c,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==decision['context_hash']
                    assert d['profile']==r['roster'][i]
                    if game=='goofspiel':
                        assert c['facts']['scores']==str(b['scores'])
                        for j,h in enumerate(b['hands']):assert c['facts'][f'public_hand_{j}']==str(h)
                        assert decision['action_id']==f'BID:{t["actual"][i]}'
                        public_bid_observations+=1
                    else:
                        assert c['facts']['public_counter_ledger']==str(tuple(b['chips']))
                        assert c['facts']['own_chips']==str(b['chips'][i])
                        assert decision['action_id']==t['actual']
                    assert all(row['values'].get('benevolence',0)==row['values'].get('universalism',0)==0 for act in c['actions'] for row in act['outcomes'])
                    assert c['opponent'] is None # no separately injected sealed-action deltas
                    if i==r['seat']:
                        if r['mode']=='learned':
                            confidence=public.snapshot().strongest_confidence if game=='goofspiel' else max(t.snapshot().confidence for j,t in enumerate(public) if j!=i)
                            assert stats['confidence']==confidence
                        focal+=1;counts['focal_decisions']+=1
                        counts['persona_difference_states']+=len(set(stats['same_state_persona_choices'].values()))>1
                        counts['changed_from_reflex']+=decision['action_id']!=stats['reflex_action']
                        counts['mc_used']+=stats.get('used',False)
                        counts['model_evaluated']+=stats.get('model_evaluated',False)
                        counts['reading_changed']+=bool(stats.get('reading_applied') and decision['action_id']!=stats['baseline_action'])
                        counts['additional_nodes']+=stats.get('additional_nodes',0)
                        if 'baseline' in stats:
                            summary=flat.setdefault((game,r['mode']),Counter())
                            summary['focal_mc_decisions']+=1
                            pred=(stats.get('learned',{}) if stats.get('reading_applied') else stats['baseline']).get('actions',{})
                            values=[p['win_share'] for p in pred.values()]
                            summary['all_roots_zero_win_share']+=bool(values and max(values)==0)
                            summary['all_roots_equal_win_share']+=bool(values and max(values)==min(values))
                if game=='goofspiel':
                    observed=Position(tuple(tuple(x) for x in b['hands']),tuple(b['scores']),tuple(b['prizes']),b['round'],b['discarded'])
                    diag=public.reveal(observed,tuple(t['actual']))
                    expected_predictions.extend(dict(tick=t['tick'],actor=k,**v) for k,v in diag.items())
                elif actor!=r['seat']:
                    observed=ThanksPosition(tuple(tuple(x) for x in b['cards']),tuple(b['chips']),b['turn'],b['card'],b['pot'],tuple(b['seen']),b['remaining'],tuple(b['payments']))
                    episode=f'tabletop-{game}-{r["seed"]}-{r["profile_index"]}'
                    diag=public[actor].observe(thanks_hypotheses(observed),t['actual'],f'{episode}:{t["tick"]}')
                    expected_predictions.append(dict(tick=t['tick'],actor=f'player-{actor}',**diag))
                previous=a
            assert json.loads(json.dumps(expected_predictions))==world['predictions']
            prediction_checks+=len(expected_predictions)
            assert dict(counts)==r['counts']
            if game=='goofspiel':assert previous['round']==13;score=previous['scores']
            else:
                assert previous['card'] is None and previous['remaining']==0
                assert sum(map(len,cards))==24
                score=[card_score(cs,ch) for cs,ch in zip(cards,ledger)]
            assert score==r['scores'] and score[r['seat']]==r['focal_score']
            best=max(score) if game=='goofspiel' else min(score);winners=[i for i,x in enumerate(score) if x==best]
            assert winners==r['winner_seats'] and r['win_share']==(1/len(winners) if r['seat'] in winners else 0)
            assert len(world['trace'])==r['turns'];runs.append(r)
    assert runs==result['runs'] and decisions==result['actual_decisions']
    for group in result['summary']['groups']:
        for mode,summary in group['modes'].items():
            rs=[r for r in runs if (r['game'],r['profile'],r['mode'])==(group['game'],group['profile'],mode)]
            assert len(rs)==summary['matches'] and sum(r['win_share'] for r in rs)==summary['winner_credit']
            assert math.isclose(sum(r['focal_score'] for r in rs)/len(rs),summary['mean_score'],rel_tol=0,abs_tol=1e-12)
    proof=dict(matches=len(runs),independent_rule_steps=steps,actual_decisions=decisions,focal_decisions=focal,
        public_bid_contexts=public_bid_observations,public_chip_ledger_steps=ledgers,all_finished=True,
        source_identity=True,trace_sha256=result['trace_sha256'],mean_score_absolute_tolerance=1e-12,
        past_public_learning_records_replayed=prediction_checks,
        unseen_real_deck_not_in_context=True,social_competition_not_mislabelled=True,
        independent_policy_replay=False,enjoyment_measured=False,default_promoted=False)
    proof.update(exact_policy_replayed_matches=replayed,
        policy_replay_scope='first registered seed / growth profile / both games / all three modes' if replay else None,
        flat_terminal_predictions=[dict(game=k[0],mode=k[1],**v) for k,v in flat.items()])
    (root/'audit.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--replay',action='store_true');args=p.parse_args()
    print(json.dumps(check(args.root,args.replay),indent=2))
