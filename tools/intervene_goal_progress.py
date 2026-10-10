"""Late-game alternative-root teachers with the ACTUAL future controllers.

World state belongs to the evaluator. Controllers still see only public state
and their own memory. These are simulation teachers, not observed experience,
and sampled retrospective best actions are not an omniscient NPC policy.
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
from reflex.goofspiel import Position
from reflex.laboratory import PROFILES
from reflex.strong_search import PublicMemory, STRONG, PERSONA, random_stream
from reflex.strong_table import play, PlaybackStart
from reflex.strong_table import decide
from reflex.goal_progress import relative_progress
from tools.probe_outcome_value import features, predict


def position(game,b):
    if game=='goofspiel':return Position(tuple(tuple(h) for h in b['hands']),tuple(b['scores']),tuple(b['prizes']),b['round'],b['discarded'])
    return ThanksPosition(tuple(tuple(h) for h in b['cards']),tuple(b['chips']),b['turn'],b['card'],b['pot'],tuple(b['seen']),b['remaining'],tuple(b['payments']))


def run(root,output,models):
    root=Path(root);output=Path(output);models=Path(models);output.mkdir(parents=True,exist_ok=True)
    registration=dict(version='actual-controller-root-interventions-v1',seeds=list(range(6700,6716)),encounters=[0,4],
        source_variant='baseline',goof_case='round 10, lowest-seat NPC, every legal remaining root',
        thanks_case='first nonforced NPC turn with remaining <= 3, every legal root',
        worlds='Goof actual deterministic continuation; No Thanks actual future plus two independent public-unseen samples',
        controller='same real adaptive decision, public learning and per-owner state after every revealed step',
        required_reference='actual-root actual-world branch must exactly match stored final scores, credits and all beliefs',
        labels='simulation teacher winner credit, NOT public real experience and NOT exact expected optimal values',
        comparison='MC objective root, factual ridge root, actual personality root and same-state progress root vs sampled retrospective best',
        strong=asdict(STRONG),persona=asdict(PERSONA),
        sources={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((Path(__file__).resolve().parents[1]/'reflex').glob('*.py'))},
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    registered=output/'preregister.json';assert not registered.exists()
    registered.write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    fitted={}
    selection=json.loads((root/'intervention_analysis_preregister.json').read_text(encoding='utf-8'))
    for game in ('goofspiel','no_thanks'):
        assert hashlib.sha256((models/f'{game}-frozen.npz').read_bytes()).hexdigest()==selection['factual_models'][game]
        with np.load(models/f'{game}-frozen.npz',allow_pickle=False) as f:fitted[game]={k:f[k] for k in f.files}
    cases=[];missing=[];reference_checked=0;branches=0
    for shard in range(4):
        folder=root/f'shard-{shard}'/'baseline';evaluation=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
        finals={(r['game'],r['seed'],r['encounter']):r for r in evaluation['matches']}
        series=None;episode=None;memories=None;states=None;selected=set()
        with (folder/'trajectories.jsonl').open(encoding='utf-8') as f:
            for line in f:
                t=json.loads(line);game=t['game'];seed=t['seed'];enc=t['encounter'];key=(game,seed,enc)
                if (game,seed)!=series:
                    series=(game,seed);memories=[PublicMemory(game,a) for a in range(4)]
                if key!=episode:episode=key;states=[None]*4
                b=t['before'];s=position(game,b);bench=t['benchmark']
                eligible=(enc in registration['encounters'] and key not in selected and
                    (s.round==10 if game=='goofspiel' else s.turn!=bench and len(s.legal())>1 and s.remaining<=3))
                if eligible:
                    selected.add(key);actor=min(a for a in range(4) if a!=bench) if game=='goofspiel' else s.turn
                    original=finals[key];d=t['decisions'][str(actor)];names=tuple(d['search']['actions'])
                    others=[p for i,p in enumerate(PROFILES) if i!=(seed//4)%4]
                    roster={a:p for a,p in zip([a for a in range(4) if a!=bench],others)}
                    _,progress_decision,_=decide(game,s,actor,roster[actor],seed,enc,t['tick'],memories[actor],states[actor],
                                               'adaptive',PERSONA,variant='progress')
                    if game=='goofspiel':decks=[()]
                    else:
                        world=random_stream(seed,game,enc,0,0,'world');deck=list(map(int,world.permutation(np.arange(3,36))[:24]))
                        real=tuple(c for c in deck if c not in s.seen);assert len(real)==s.remaining
                        unseen=np.array([c for c in range(3,36) if c not in s.seen])
                        decks=[real]+[tuple(map(int,random_stream(seed,game,enc,t['tick'],actor,f'teacher-future-{i}').permutation(unseen)[:s.remaining])) for i in (1,2)]
                    rewards=[];scores=[]
                    for move in names:
                        root_rewards=[];root_scores=[]
                        for world,deck in enumerate(decks):
                            start=PlaybackStart(s,tuple(deck),tuple(copy.deepcopy(states)),t['tick'])
                            r,_=play(game,seed,enc,bench,roster,'adaptive',copy.deepcopy(memories),
                                strong=STRONG,persona=PERSONA,start=start,forced_root=(actor,move))
                            branches+=1;root_rewards.append(r['credits'][actor]);root_scores.append(r['scores'])
                            final=r['final']
                            if game=='goofspiel':
                                assert final['round']==13 and sum(final['scores'])+final['discarded']==sum(final['prizes'])
                                assert all(not h for h in final['hands'])
                            else:
                                assert final['card'] is None and final['remaining']==0 and final['pot']==0
                                assert sum(final['chips'])==44 and sum(map(len,final['cards']))==24
                            if world==0 and move==d['action']:
                                assert r['scores']==original['scores'] and r['credits']==original['credits']
                                assert json.loads(json.dumps(r['final']))==original['final']
                                assert json.loads(json.dumps(r['beliefs']))==original['beliefs'];reference_checked+=1
                        rewards.append(root_rewards);scores.append(root_scores)
                    reward=np.array(rewards);mean=reward.mean(1);stats=d['search']['actions'];direction=1 if game=='goofspiel' else -1
                    mc=max(range(len(names)),key=lambda i:(stats[names[i]]['win_share'],direction*stats[names[i]]['mean_score'],-i))
                    q=predict(fitted[game],np.array([features(game,b,actor,a,d['context']) for a in names]));ridge=int(q.argmax())
                    actual=names.index(d['action']);progress_root=names.index(progress_decision['action_id']);best=float(mean.max())
                    advance=relative_progress(np.asarray(scores),actor,direction=direction,scale=max(s.prizes) if game=='goofspiel' else 35)
                    advance_mean=advance.mean(1);best_advance=float(advance_mean.max())
                    cases.append(dict(game=game,seed=seed,encounter=enc,tick=t['tick'],actor=actor,profile=roster[actor]['id'],
                        public_before=b,actions=list(names),teacher_credit=rewards,teacher_scores=scores,teacher_progress=advance.tolist(),
                        owner_state=copy.deepcopy(states[actor]),owner_personality=d['context']['personality'],owner_values=d['context']['values'],owner_needs=d['context']['needs'],
                        mc_prediction=[stats[a]['win_share'] for a in names],ridge_prediction=q.tolist(),
                        mc_root=names[mc],ridge_root=names[ridge],actual_root=names[actual],progress_root=names[progress_root],
                        regret=dict(mc=best-float(mean[mc]),ridge=best-float(mean[ridge]),personality=best-float(mean[actual]),progress=best-float(mean[progress_root])),
                        progress_regret={name:best_advance-float(advance_mean[i]) for name,i in (('mc',mc),('ridge',ridge),('personality',actual),('progress',progress_root))},
                        retrospective_best_credit=best,labels_are_simulated=True))
                    print(game,seed,enc,'cases',len(cases),'branches',branches,flush=True)
                for a,d in t['decisions'].items():
                    if int(a)!=bench:states[int(a)]=d['next_state']
                actors=range(4) if game=='goofspiel' else (s.turn,)
                for observer in range(4):
                    for actor in actors:
                        if observer!=actor:memories[observer].observe(s,actor,t['moves'][str(actor)],f'encounter-{enc}-tick-{t["tick"]}-actor-{actor}')
        for key in finals:
            if key[2] in registration['encounters'] and key not in selected:missing.append(key)
    summary=[]
    for game in ('goofspiel','no_thanks'):
        rows=[r for r in cases if r['game']==game]
        informative=[r for r in rows if np.ptp(np.mean(r['teacher_credit'],axis=1))>0]
        summary.append(dict(game=game,cases=len(rows),informative_cases=len(informative),
            mean_sampled_regret={k:float(np.mean([r['regret'][k] for r in rows])) for k in ('mc','ridge','personality','progress')},
            mean_sampled_progress_regret={k:float(np.mean([r['progress_regret'][k] for r in rows])) for k in ('mc','ridge','personality','progress')},
            informative_mean_sampled_regret={k:(float(np.mean([r['regret'][k] for r in informative])) if informative else None) for k in ('mc','ridge','personality','progress')}))
    assert reference_checked==len(cases)
    result=dict(registration=registration,cases=cases,summary=summary,missing=missing,branches=branches,
                exact_actual_root_reference_checks=reference_checked,adopted_as_controller=False,
                limitations=['small late-game probe, same known rule families and controller profiles',
                    'one/three world labels do not establish exact expected action quality',
                    'future sampled-best selection is retrospective; no oracle values enter NPC decisions'])
    (output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--models',required=True)
    a=p.parse_args();run(a.root,a.output,a.models)
