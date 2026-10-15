"""Diagnostic terminal consequences of alternative roots on one actual owner.

Reconstructs an unplanned stored run exactly up to the nominated tick. After
that, paired counterfactual branches share random uniforms and the same owner;
only their first action differs. Continuation uses the ordinary reflex owner,
including learning/needs/progress. Known experimental truth is used ONLY by
this offline evaluator, never fed to the controller. This does not establish
personality regret, an optimal policy, or a usable online value estimator.
"""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest,compile_batch
from reflex.laboratory import PROFILES
from reflex.actor_model import ActorModel
from reflex.model_observer import resolve_feedback
from tools.run_prediction_recovery import RecoveryAdapter,truth,sample


def diagnose(root,game,profile,regime,tick,roots,samples):
    rows=[json.loads(x) for x in (root/'trajectories.jsonl').read_text().splitlines()]
    r=next(x for x in rows if (x['game'],x['profile'],x['regime'],x['arm'])==(game,profile,regime,'recovery'))
    assert r['plan_every']==0 and 0<=tick<r['turns'] and samples>1
    p=next(p for p in PROFILES if p['id']==profile)
    a=RecoveryAdapter(game,p,r['seed']%2);s=a.start();actor=a.actor_for(a.raw(s),'recovery')
    uniforms=np.random.default_rng(r['seed']).random(r['turns'])
    fb=lambda b,e,k,c:a.feedback(b,e,k,c,'channels','censored')
    for t in range(tick):
        if a.terminal(s):s=a.start(s['round']+1,t)
        c=a.raw(s,actor.state);result=actor.decide(a.request(c),False)
        assert result['decision']['action_id']==r['trace'][t]['action']
        end,_,_=a.transition(s,result['decision']['action_id'],sample(truth(a.symbols(),t,r['turns'],regime),uniforms[t]))
        resolve_feedback(actor,result['ticket'],fb(s,end,result['decision']['action_id'],c))
        assert digest(actor.record())==r['trace'][t]['owner_digest']
        s=end
    if a.terminal(s):s=a.start(s['round']+1,tick)
    c=a.raw(s,actor.state);req=a.request(c);before=digest(actor.record())
    selected=copy.deepcopy(actor).decide(req,False)['decision']['action_id']
    assert selected==r['trace'][tick]['action'] and set(roots)<=set(a.keys(s)) and selected in roots
    stage=actor._prepare(req);known=copy.deepcopy(stage['effective'])
    for action in known['actions']:
        action['outcomes']=[dict(a.outcome(c,action['id'],z),p=probability)
            for z,probability in truth(a.symbols(),tick,r['turns'],regime).items()]
    batch=compile_batch([known]);scored=actor.policy.decide(batch,False)
    immediate={k:float(v) for k,v in zip(batch.ids[0],scored.scores[0])}
    # Seed bank is independent of the played trajectory; paired within each
    # root comparison. It is not presented as unseen-game generalization.
    draws=np.random.default_rng(52017).random((samples,16));branches=[];decisions=0
    for index,draw in enumerate(draws):
        outcomes={}
        for first in roots:
            model=ActorModel(actor,root_request=req,request_for=a.request,feedback=fb)
            w=copy.deepcopy(s);actions=[];responses=[]
            for depth,u in enumerate(draw):
                if a.terminal(w):break
                cc=a.raw(w,model.actor.state)
                prepared,request,_,_=model.prepare(w,cc,prepared=depth==0)
                choice=model.preview();key=first if depth==0 else choice['action_id']
                model.commit_choice(key)
                end,_,response=a.transition(w,key,sample(truth(a.symbols(),w['tick'],r['turns'],regime),u))
                model.advance(w,end,key,prepared,request)
                actions.append(key);responses.append(response);w=end;decisions+=1
            assert a.terminal(w),'diagnostic must reach the declared endpoint'
            outcomes[first]=dict(score=a.goal(w),actions=actions,responses=responses)
        branches.append(dict(sample=index,outcomes=outcomes))
    assert digest(actor.record())==before
    means={k:float(np.mean([b['outcomes'][k]['score'] for b in branches])) for k in roots}
    differences={}
    for key in roots:
        d=np.array([b['outcomes'][key]['score']-b['outcomes'][selected]['score'] for b in branches])
        differences[key]=dict(mean=float(d.mean()),standard_error=float(d.std(ddof=1)/np.sqrt(samples)),
                              better=int(np.sum(d>1e-12)),equal=int(np.sum(np.abs(d)<=1e-12)),worse=int(np.sum(d< -1e-12)))
    result=dict(game=game,profile=profile,regime=regime,tick=tick,public=s,owner_digest=before,
        selected=selected,prediction=actor.predictor.predict(c),roots=list(roots),samples=samples,
        progress_allowed=sorted(stage['progress_allowed']),same_person_true_law_immediate_scores=immediate,
        exact_prefix_replayed=tick,model_decisions=decisions,mean_terminal_score=means,
        paired_difference_from_selected=differences,branches=branches,semantics=__doc__)
    path=root/f'horizon_{game}_{profile}_{regime}_{tick}.json'
    if path.exists():assert json.loads(path.read_text())['branches']==branches,'counterfactual replay drift'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('branches','semantics','public')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--game',required=True)
    p.add_argument('--profile',required=True);p.add_argument('--regime',required=True);p.add_argument('--tick',type=int,required=True)
    p.add_argument('--roots',nargs='+',required=True);p.add_argument('--samples',type=int,default=128);a=p.parse_args()
    diagnose(Path(a.root),a.game,a.profile,a.regime,a.tick,tuple(a.roots),a.samples)
