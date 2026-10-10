"""Learn action differences from matched, fully evaluated alternatives.

Public features and terminal purpose targets belong to the adapter. No game,
opponent identity, or reward definition is inferred here. Constant case effects
are removed before fitting; factual chosen-action labels cannot train this API.
"""
import numpy as np


def _basis(x,mean,scale):
    z=np.clip((np.asarray(x,float)-mean)/scale,-4,4)
    i,j=np.triu_indices(z.shape[1])
    return np.column_stack((z,z[:,i]*z[:,j]))


def fit(cases,*,alpha=10.):
    """Each case: features[actions, dimensions], targets[actions, purposes]."""
    if not cases or not np.isfinite(alpha) or alpha<=0:raise ValueError('matched cases and positive regularization required')
    xs=[];ys=[]
    for x,y in cases:
        x=np.asarray(x,float);y=np.asarray(y,float)
        if x.ndim!=2 or y.ndim!=2 or len(x)<2 or len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((y<0)|(y>1)):
            raise ValueError('two or more matched alternatives with bounded purpose targets required')
        if xs and (x.shape[1]!=xs[0].shape[1] or y.shape[1]!=ys[0].shape[1]):raise ValueError('stable feature and purpose meanings required')
        xs.append(x);ys.append(y)
    x=np.concatenate(xs);weights=np.concatenate([np.full(len(a),1/len(a)) for a in xs])
    mean=np.average(x,axis=0,weights=weights);scale=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=weights)),.05)
    design=[];targets=[]
    for x,y in zip(xs,ys):
        b=_basis(x,mean,scale);design.append(b-b.mean(0));targets.append(y-y.mean(0))
    b=np.concatenate(design);y=np.concatenate(targets)
    coef=np.linalg.solve((b.T*weights)@b+alpha*np.eye(b.shape[1]),b.T@(weights[:,None]*y))
    return dict(version='matched-action-differences-v1',mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),
                cases=len(cases),purposes=y.shape[1],alpha=float(alpha),constant_case_effects_removed=True)


def predict(model,features):
    x=np.asarray(features,float);mean=np.asarray(model['mean'],float);scale=np.asarray(model['scale'],float);coef=np.asarray(model['coef'],float)
    if x.ndim!=2 or len(x)<2 or x.shape[1]!=len(mean) or not np.isfinite(x).all():raise ValueError('matched finite candidate features required')
    b=_basis(x,mean,scale);values=(b-b.mean(0))@coef
    supported=bool(np.all(np.abs((x-mean)/scale)<=4))
    return values,supported


def calibrate(model,cases,*,quantile=.9):
    """Held-out absolute errors of action differences, not reward confidence."""
    if not cases or not 0<quantile<1:raise ValueError('held-out matched cases and interior quantile required')
    errors=[]
    for x,y in cases:
        p,_=predict(model,x);y=np.asarray(y,float)
        if y.shape!=p.shape or not np.isfinite(y).all() or np.any((y<0)|(y>1)):raise ValueError('bounded matched calibration targets required')
        i,j=np.triu_indices(len(y),1);errors.extend(np.abs((p[i]-p[j])-(y[i]-y[j])))
    return np.quantile(np.asarray(errors),quantile,axis=0,method='higher').tolist()
