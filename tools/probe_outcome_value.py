"""Exploratory factual value prediction from public state and OWN chosen action.

This does not select actions. Labels are observed terminal winner credits. The
fit is factual, with behaviour-policy/support bias, not a counterfactual oracle.
Game features stay in this adapter/tool; the fitted numerical learner is shared.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def features(game,b,actor,move,context):
    traits=[context['personality'][k] for k in ('openness','conscientiousness','extraversion','agreeableness','neuroticism')]
    values=[context['values'][k] for k in ('achievement','security','power')]
    rivals=[i for i in range(4) if i!=actor]
    if game=='goofspiel':
        h=b['hands'][actor];bid=int(move.split(':')[1]);prize=b['prizes'][b['round']]
        othermax=[max(b['hands'][i]) for i in rivals]
        chance=float(np.prod([sum(c<bid for c in b['hands'][i])/len(b['hands'][i]) for i in rivals]))
        future=b['prizes'][b['round']:]
        x=[len(h)/13,prize/13,sum(future)/91,bid/13,h.index(bid)/max(1,len(h)-1),
           max(h)/13,np.mean(h)/13,b['scores'][actor]/91,max(b['scores'][i] for i in rivals)/91,
           np.mean([b['scores'][i] for i in rivals])/91,max(othermax)/13,min(othermax)/13,np.mean(othermax)/13,
           chance,prize*chance/13,sum(p<=prize for p in future)/len(future)]
    else:
        def points(cards):return sum(c for c in cards if c-1 not in cards)
        scores=[points(h)-chips for h,chips in zip(b['cards'],b['chips'])]
        card=b['card'];pot=b['pot'];stock=b['chips'][actor];own=b['cards'][actor]
        added=[points(h+[card])-points(h) for h in b['cards']];take=move=='TAKE';nxt=(actor+1)%4
        x=[b['remaining']/24,card/35,pot/44,stock/44,points(own)/300,scores[actor]/300,
           min(scores[i] for i in rivals)/300,np.mean([scores[i] for i in rivals])/300,
           (scores[actor]-min(scores[i] for i in rivals))/100,len(own)/24,sum(own)/420,
           added[actor]/35,((pot-added[actor]) if take else -1)/35,
           ((stock+pot) if take else stock-1)/44,min(added[i] for i in rivals)/35,added[nxt]/35,
           max(b['chips'][i] for i in rivals)/44,min(b['chips'][i] for i in rivals)/44,
           np.mean([b['chips'][i] for i in rivals])/44,float(take),float(b['remaining']==0)]
    return np.array(x+traits+values,dtype=float)


def basis(x,mean,scale):
    z=np.clip((x-mean)/scale,-4,4)
    # Quadratic features are shared across games; no action table or game branch
    # inside the fit. Regularize every non-intercept coefficient equally.
    i,j=np.triu_indices(z.shape[1]);return np.column_stack((np.ones(len(z)),z,z[:,i]*z[:,j]))


def fit(x,y,weights,alpha=10.):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);weights=np.asarray(weights,dtype=float)
    if x.ndim!=2 or not len(x) or y.shape!=(len(x),) or weights.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(weights).all():
        raise ValueError('matched finite factual training arrays required')
    if np.any((y<0)|(y>1)) or np.any(weights<=0) or not np.isfinite(alpha) or alpha<=0:
        raise ValueError('bounded outcomes, positive weights and regularization required')
    mean=np.average(x,axis=0,weights=weights)
    scale=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)),.05);b=basis(x,mean,scale)
    matrix=(b.T*weights)@b;rhs=b.T@(weights*y)
    matrix+=alpha*np.diag(np.r_[0.,np.ones(b.shape[1]-1)])
    coef=np.linalg.solve(matrix,rhs)
    return dict(mean=mean,scale=scale,coef=coef)


def predict(model,x):return np.clip(basis(x,model['mean'],model['scale'])@model['coef'],.02,.98)


def extract(root):
    evaluation=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    outcomes={(r['game'],r['seed'],r['encounter']):r['credits'] for r in evaluation['matches'] if r['mode']=='adaptive'}
    rows={g:[] for g in ('goofspiel','no_thanks')}
    with (root/'trajectories.jsonl').open(encoding='utf-8') as f:
        for line in f:
            t=json.loads(line)
            if t['mode']!='adaptive':continue
            game=t['game'];key=(game,t['seed'],t['encounter'])
            for a,d in t['decisions'].items():
                a=int(a)
                if a==t['benchmark']:continue
                rows[game].append((features(game,t['before'],a,d['action'],d['context']),
                    outcomes[key][a],d['search']['actions'][d['action']]['win_share'],t['seed'],t['encounter'],a))
    result={}
    for game,data in rows.items():
        x=np.array([r[0] for r in data]);y=np.array([r[1] for r in data]);mc=np.array([r[2] for r in data])
        key=np.array([r[3:] for r in data],int);unique,inv,count=np.unique(key,axis=0,return_inverse=True,return_counts=True)
        result[game]=dict(x=x,y=y,mc=mc,key=key,weights=1/count[inv])
    return result


def metrics(y,p,w):
    return dict(observations=len(y),weighted_brier=float(np.average((p-y)**2,weights=w)),
        weighted_log_loss=float(np.average(-(y*np.log(np.clip(p,.02,.98))+(1-y)*np.log(np.clip(1-p,.02,.98))),weights=w)),
        weighted_prediction=float(np.average(p,weights=w)),weighted_actual=float(np.average(y,weights=w)))


def run(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    trace_sha=hashlib.sha256((source/'trajectories.jsonl').read_bytes()).hexdigest()
    registration=dict(version='factual-public-value-probe-v2',source_trace_sha256=trace_sha,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        development_train_seeds=list(range(6100,6108)),development_test_seeds=list(range(6108,6116)),
        mode='adaptive',actors='NPC only; own traits allowed, no rival identities or hidden future',
        learner='quadratic ridge, fixed alpha 10, standardized public features, output clipped [.02,.98]',
        weighting='each NPC episode has total weight one, so long matches do not get more training importance',
        target='observed terminal winner credit under the actual behaviour policy',
        prospective='after model freeze, baseline series 6700..6715 only, not used for fitting',
        caveats=['historical development split was previously inspected for other diagnostics',
                 'factual chosen-action forecasts cannot establish counterfactual action quality',
                 'this tool never changes runtime policy'])
    registered=output/'preregister.json';assert not registered.exists()
    registered.write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    data=extract(source);results=[]
    for game,d in data.items():
        train=d['key'][:,0]<6108;test=~train
        model=fit(d['x'][train],d['y'][train],d['weights'][train])
        p=predict(model,d['x'][test]);w=d['weights'][test];y=d['y'][test]
        results.append(dict(game=game,development_fit_rows=int(train.sum()),
            holdout_ridge=metrics(y,p,w),holdout_mc=metrics(y,d['mc'][test],w),
            holdout_constant=metrics(y,np.full(len(y),np.average(d['y'][train],weights=d['weights'][train])),w)))
        final=fit(d['x'],d['y'],d['weights'])
        np.savez_compressed(output/f'{game}-frozen.npz',**final)
        results[-1]['frozen_model_sha256']=hashlib.sha256((output/f'{game}-frozen.npz').read_bytes()).hexdigest()
        print(game,json.dumps(results[-1]),flush=True)
    payload=dict(registration=registration,results=results,adopted_as_controller=False)
    (output/'development.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source,a.output)
