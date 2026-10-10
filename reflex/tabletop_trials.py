"""Existing-rule four-player trials, with a single controlled focal NPC.

All four seats use the common finite-priority personality Policy. The focal
seat adds terminal MC and then optional public-action hypotheses. Future own
actions are uniform assumptions, not a simulation of the actual future NPC.
There are no relationships/allies in these games: rival gains are not kindness.
"""
from collections import Counter
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import Policy, compile_batch, digest, TRAITS, VALUES
from .examples import action
from .laboratory import PROFILES
from .goofspiel import Position, observe as bid_observe, referee
from .goofspiel_beliefs import PublicBidBeliefs
from .board_models import ThanksAdapter, ThanksPosition, card_points
from .rollout_boards import BoardRolloutModel, thanks_tactical
from .monte_carlo import RolloutBudget, evaluate_actions
from .opponent_beliefs import HypothesisTracker

MODES=('reflex','mc','learned')
BUDGET=RolloutBudget(samples=16,min_samples=8,max_nodes=50000,max_steps=512,rollout_policy='random')
PLAN=dict(version='existing-tabletop-v1',seeds=list(range(5008,5016)),development_seed=5000,
    games=['goofspiel','no_thanks'],profiles=[p['id'] for p in PROFILES],modes=list(MODES),
    players=4,controlled_seat='seed modulo 4',rivals='three fixed-persona finite-policy reflex NPCs',
    goal='terminal fractional winner credit; raw scores reported separately',budget=asdict(BUDGET),
    goofspiel='13 cards, ascending on even seeds / descending on odd; unique high wins; highest tie discards',
    no_thanks='3..35, nine unseen removals, 24 drawn cards, 11 initial chips; taker starts next card',
    needs='existing adapter resource/esteem/safety proxies; not all five needs supported',
    social='not connected: competitive outcomes alone are not benevolent/hostile social events',
    continuation='uniform own future and uniform/unobserved or frozen learned rival mixture; real NPCs redecide',
    learning='finite existing HypothesisTracker; actual revealed actions only; reset each match',
    adoption='diagnostic comparison only; no default policy promotion or result-dependent tuning')


def competitive(c):
    """Do not label a competitive rival's gain/denial as help or betrayal."""
    c=copy.deepcopy(c)
    for a in c['actions']:
        for row in a['outcomes']:
            for key in ('benevolence','universalism'):row['values'][key]=0.
    c['facts']['social_semantics']='no allies or help agreement; competitive gains do not identify kindness'
    return c


def score(c):
    b=compile_batch([c]);r=Policy(principle_priority='finite').decide(b,False)
    return r.records(b)[0],{a:float(r.scores[0,j]) for j,a in enumerate(b.ids[0])}


def persona_probe(c):
    choices={}
    for p in PROFILES:
        row=copy.deepcopy(c);row['personality']=dict(zip(TRAITS,p['traits']))
        row['values']={k:float(p['values'].get(k,0)) for k in VALUES}
        choices[p['id']]=score(row)[0]['action_id']
    return choices


def thanks_hypotheses(s):
    names=s.legal();uniform={a:1/len(names) for a in names}
    added=card_points(s.cards[s.turn]+(s.card,))-card_points(s.cards[s.turn])
    chosen=dict(tactical=thanks_tactical(s,None),
        reserve='TAKE' if s.chips[s.turn]<=2 or added<=s.pot else 'PASS',
        accept='TAKE' if added<=s.pot+4 else 'PASS')
    return dict(uniform=uniform,**{k:{a:float(a==(v if v in names else 'TAKE')) for a in names} for k,v in chosen.items()})


class ThanksForecastModel(BoardRolloutModel):
    def __init__(self,*args,snapshots=None,**kwargs):
        super().__init__(*args,**kwargs);self.snapshots=snapshots

    def choose(self,s,rng,policy):
        names=s.legal();u=float(rng.random())
        if self.snapshots is not None and s.turn!=self.viewer:
            probs=self.snapshots[s.turn].predict(thanks_hypotheses(s))
        else:probs={a:1/len(names) for a in names}
        cumulative=0.
        for name in names:
            cumulative+=probs[name]
            if u<cumulative:return name
        return names[-1]


def thanks_observe(s,p,seed,tick,episode,memory,budget=None,snapshots=None):
    adapter=ThanksAdapter(sample_seed=seed+991)
    roots={a:s.play(a) for a in s.legal()}
    stats=dict(used=False,completed_samples=0,additional_nodes=0,actions={})
    packed={a:[dict(adapter.consequence(s,child,s.turn),p=1.)] for a,child in roots.items()}
    if budget:
        packed_mc,stats=evaluate_actions(roots,ThanksForecastModel(adapter,s,p,seed,tick,episode,memory,snapshots=snapshots),
            budget,[seed,'no_thanks',episode,f'player-{s.turn}',tick])
        if stats['used']:packed=packed_mc
    acts=[action(a,*packed[a]) for a in s.legal()]
    for a in acts:a['target']=adapter.target(s,a['id'])
    c=competitive(adapter.context(s,p,seed,tick,episode,acts,memory))
    c['facts']['continuation']='own uniform; rivals uniform or frozen public-action mixture; unseen cards sampled'
    return c,stats


def choose(game,s,viewer,p,seed,tick,episode,memory,mode,beliefs):
    def observation(budget=None,learn=False):
        if game=='goofspiel':
            c,st=bid_observe(s,viewer,p,'win_share',seed,tick,episode,memory,budget,
                beliefs=beliefs.snapshot() if learn else None,common_random=True)
            return competitive(c),st
        return thanks_observe(s,p,seed,tick,episode,memory,budget,
            tuple(t.snapshot() for t in beliefs) if learn else None)
    immediate,im_stats=observation();reflex,_=score(immediate)
    if mode=='reflex':return immediate,reflex,dict(reflex_action=reflex['action_id'],used=False,reason='reflex',additional_nodes=0)
    c,st=observation(BUDGET);d,own_scores=score(c)
    stats=dict(reflex_action=reflex['action_id'],used=st['used'],baseline=st,
        additional_nodes=st['additional_nodes'],model_evaluated=False,reading_applied=False,
        reason='uniform_future',baseline_action=d['action_id'])
    if mode=='mc':return c,d,stats
    confidence=(beliefs.snapshot().strongest_confidence if game=='goofspiel' else
        max(t.snapshot().confidence for i,t in enumerate(beliefs) if i!=viewer))
    stats['confidence']=confidence
    if confidence<.2:stats['reason']='insufficient_public_evidence';return c,d,stats
    if game=='goofspiel' and s.scores[viewer]>max(x for i,x in enumerate(s.scores) if i!=viewer)+sum(s.prizes[s.round:]):
        stats['reason']='mathematically_locked_lead';return c,d,stats
    learned,ls=observation(BUDGET,True);ld,l_scores=score(learned)
    stats.update(model_evaluated=True,additional_nodes=stats['additional_nodes']+ls['additional_nodes'],learned=ls)
    old=ls['actions'].get(d['action_id'],{});new=ls['actions'].get(ld['action_id'],{})
    gain=l_scores[ld['action_id']]-l_scores[d['action_id']]
    required=.025+.65*((old.get('return_se') or 0)+(new.get('return_se') or 0))
    interval=st['actions'].get(d['action_id'],{}).get('win_rate_interval95')
    maintained=game=='goofspiel' and s.scores[viewer]>max(x for i,x in enumerate(s.scores) if i!=viewer) and interval and interval[0]>.5
    applied=bool(ls['used'] and not maintained and (ld['action_id']==d['action_id'] or gain>required))
    stats.update(reading_applied=applied,learned_action=ld['action_id'],persona_gain=gain,required_gain=required,
        reason='maintained_advantage' if maintained else 'reading_accepted' if applied else 'insufficient_gain_or_samples')
    return (learned,ld,stats) if applied else (c,d,stats)


def snapshot(s):return json.loads(json.dumps(asdict(s)))


def play(game,seed,profile_index,mode):
    if game not in PLAN['games'] or mode not in MODES or profile_index not in range(4):raise ValueError('registered game/profile/mode required')
    seat=seed%4;roster=[PROFILES[(profile_index+i-seat)%4] for i in range(4)]
    memories=[None]*4;episode=f'tabletop-{game}-{seed}-{profile_index}'
    if game=='goofspiel':
        state=Position.start(4,13,'ascending' if seed%2==0 else 'descending');ledger=None;deck=None
        beliefs=[PublicBidBeliefs(4,i) for i in range(4)]
    else:
        deck=tuple(int(x) for x in np.random.default_rng(seed+1729).permutation(np.arange(3,36))[:24]);draw=1
        state=ThanksPosition.start(4,deck[0],seat);ledger=[11]*4
        beliefs=tuple(HypothesisTracker(('uniform','tactical','reserve','accept')) for _ in range(4))
    trace=[];predictions=[];counts=Counter()
    for tick in range(2000):
        done=state.terminal if game=='goofspiel' else state.card is None and state.remaining==0
        if done:break
        if game=='no_thanks' and state.card is None:state=state.draw(deck[draw]);draw+=1
        before=snapshot(state);decisions=[]
        actors=range(4) if game=='goofspiel' else [state.turn]
        for actor in actors:
            view=state if game=='goofspiel' else replace(state,chips=tuple(ledger))
            assert game=='goofspiel' or view.chips==state.chips
            c,d,st=choose(game,view,actor,roster[actor],seed,tick,episode,memories[actor],mode if actor==seat else 'reflex',
                beliefs[actor] if game=='goofspiel' else beliefs)
            assert d['action_id'] in ([f'BID:{x}' for x in state.hands[actor]] if game=='goofspiel' else state.legal())
            assert c['personality']==dict(zip(TRAITS,roster[actor]['traits']))
            assert all(c['values'][k]==roster[actor]['values'].get(k,0) for k in VALUES)
            memories[actor]=d['next_state'];counts['decisions']+=1
            if actor==seat:
                st['same_state_persona_choices']=persona_probe(c)
                counts['persona_difference_states']+=len(set(st['same_state_persona_choices'].values()))>1
                counts['focal_decisions']+=1;counts['changed_from_reflex']+=d['action_id']!=st['reflex_action']
                counts['mc_used']+=st.get('used',False);counts['model_evaluated']+=st.get('model_evaluated',False)
                counts['reading_changed']+=bool(st.get('reading_applied') and d['action_id']!=st['baseline_action'])
                counts['additional_nodes']+=st.get('additional_nodes',0)
            decisions.append(dict(actor=actor,profile=roster[actor]['id'],context=c,decision=d,stats=st))
        if game=='goofspiel':
            bids=tuple(int(x['decision']['action_id'].split(':')[1]) for x in decisions)
            after=state.play(bids);referee(state,bids,after)
            diag=beliefs[seat].reveal(state,bids)
            for i,b in enumerate(beliefs):
                if i!=seat:b.reveal(state,bids)
            predictions.extend(dict(tick=tick,actor=k,**v) for k,v in diag.items())
            counts['tied_prizes']+=bids.count(max(bids))>1
            actual=list(bids)
        else:
            actor=state.turn;actual=decisions[0]['decision']['action_id']
            if actor!=seat:
                diag=beliefs[actor].observe(thanks_hypotheses(replace(state,chips=tuple(ledger))),actual,f'{episode}:{tick}')
                predictions.append(dict(tick=tick,actor=f'player-{actor}',**diag))
            if actor==seat:
                counts['forced_take']+='PASS' not in state.legal()
                counts['passes']+=actual=='PASS'
                added=card_points(state.cards[actor]+(state.card,))-card_points(state.cards[actor])-state.pot
                counts['voluntary_costly_take']+=actual=='TAKE' and 'PASS' in state.legal() and added>1
            after=state.play(actual)
            ledger[actor]+=state.pot if actual=='TAKE' else -1
            assert tuple(ledger)==after.chips and sum(after.chips)+after.pot==44
            from .rule_baseline_experiment import check
            from .rule_baseline import ThanksRules
            check(ThanksRules(4),state,actual,after)
        trace.append(dict(tick=tick,before=before,decisions=decisions,actual=actual,after=snapshot(after)))
        state=after
    else:raise AssertionError('game did not finish')
    scores=list(state.scores if game=='goofspiel' else state.scores());best=max(scores) if game=='goofspiel' else min(scores)
    winners=[i for i,x in enumerate(scores) if x==best]
    return dict(game=game,seed=seed,profile=PROFILES[profile_index]['id'],profile_index=profile_index,
        mode=mode,seat=seat,roster=[p['id'] for p in roster],scores=scores,
        winner_seats=winners,win_share=1/len(winners) if seat in winners else 0.,
        focal_score=scores[seat],turns=len(trace),finished=True,counts=dict(counts)),trace,predictions


SOURCES=('tabletop_trials.py','core.py','goofspiel.py','goofspiel_beliefs.py','opponent_beliefs.py',
    'board_models.py','rollout_boards.py','monte_carlo.py','planning.py','laboratory.py',
    'examples.py','rule_baseline.py','rule_baseline_experiment.py')


def summarize(runs,predictions):
    groups=[]
    for game in PLAN['games']:
        for profile in PLAN['profiles']:
            row=dict(game=game,profile=profile,modes={})
            for mode in MODES:
                rs=[r for r in runs if (r['game'],r['profile'],r['mode'])==(game,profile,mode)]
                row['modes'][mode]=dict(matches=len(rs),winner_credit=sum(r['win_share'] for r in rs),
                    mean_score=float(np.mean([r['focal_score'] for r in rs])),counts=dict(sum((Counter(r['counts']) for r in rs),Counter())))
            groups.append(row)
    paired={}
    lookup={(r['game'],r['seed'],r['profile'],r['mode']):r for r in runs}
    for game in PLAN['games']:
        paired[game]={}
        for mode,baseline in (('mc','reflex'),('learned','mc')):
            counts=Counter()
            for r in runs:
                if r['game']!=game or r['mode']!=mode:continue
                old=lookup[game,r['seed'],r['profile'],baseline]
                delta=r['win_share']-old['win_share']
                counts['win_better' if delta>1e-9 else 'win_worse' if delta< -1e-9 else 'win_same']+=1
                score_delta=(r['focal_score']-old['focal_score'])*(1 if game=='goofspiel' else -1)
                counts['score_better' if score_delta>0 else 'score_worse' if score_delta<0 else 'score_same']+=1
            paired[game][mode+'_against_'+baseline]=dict(counts)
    prediction=[]
    for game in PLAN['games']:
        for mode in MODES:
            rows=[x for x in predictions if x['game']==game and x['mode']==mode and not x['forced']]
            prediction.append(dict(game=game,mode=mode,observations=len(rows),
                mean_log_loss=float(np.mean([x['log_loss'] for x in rows])),
                uniform_log_loss=float(np.mean([x['uniform_log_loss'] for x in rows]))))
    return dict(groups=groups,paired=paired,prediction=prediction)


def experiment(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if (root/'preregister.json').exists():raise FileExistsError('fresh output required')
    registration=dict(PLAN,source_hashes={p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in SOURCES})
    (root/'preregister.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    runs=[];predictions=[]
    with (root/'trajectories.jsonl').open('w',encoding='utf-8') as f:
        for game in PLAN['games']:
            for seed in PLAN['seeds']:
                for pi in range(4):
                    for mode in MODES:
                        r,t,p=play(game,seed,pi,mode);runs.append(r)
                        predictions.extend(dict(game=game,mode=mode,seed=seed,profile=PROFILES[pi]['id'],**x) for x in p)
                        f.write(json.dumps(dict(run=r,trace=t,predictions=p),ensure_ascii=False)+'\n')
                print(f'completed {game}/{seed}',flush=True)
    result=dict(plan=registration,runs=runs,summary=summarize(runs,predictions),
        matches=len(runs),actual_decisions=sum(r['counts']['decisions'] for r in runs),
        trace_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest(),
        all_finished=all(r['finished'] for r in runs),enjoyment_measured=False,training=False,default_promoted=False)
    (root/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    r=experiment(args.output);print(json.dumps({k:v for k,v in r.items() if k not in ('plan','runs')},indent=2))
