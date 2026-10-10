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


def fit(cases,*,alpha=10.,case_weights=None):
    """Each case: features[actions, dimensions], targets[actions, purposes]."""
    if not cases or not np.isfinite(alpha) or alpha<=0:raise ValueError('matched cases and positive regularization required')
    xs=[];ys=[]
    for x,y in cases:
        x=np.asarray(x,float);y=np.asarray(y,float)
        if x.ndim!=2 or y.ndim!=2 or len(x)<2 or len(x)!=len(y) or not 1<=x.shape[1]<=64 or not 1<=y.shape[1]<=16 or not np.isfinite(x).all() or np.any(np.abs(x)>1e6) or not np.isfinite(y).all() or np.any((y<0)|(y>1)):
            raise ValueError('two or more matched alternatives with bounded purpose targets required')
        if xs and (x.shape[1]!=xs[0].shape[1] or y.shape[1]!=ys[0].shape[1]):raise ValueError('stable feature and purpose meanings required')
        xs.append(x);ys.append(y)
    case_weights=np.ones(len(cases)) if case_weights is None else np.asarray(case_weights,float)
    if case_weights.shape!=(len(cases),) or not np.isfinite(case_weights).all() or np.any(case_weights<=0):raise ValueError('one positive finite case weight required')
    x=np.concatenate(xs);weights=np.concatenate([np.full(len(a),w/len(a)) for a,w in zip(xs,case_weights)])
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
    if model.get('version')!='matched-action-differences-v1' or mean.ndim!=1 or not 1<=len(mean)<=64 or scale.shape!=mean.shape or coef.ndim!=2 or coef.shape[0]!=len(mean)+len(mean)*(len(mean)+1)//2 or not 1<=coef.shape[1]<=16 or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0) or not np.isfinite(coef).all():raise ValueError('valid bounded fitted model required')
    if x.ndim!=2 or len(x)<2 or x.shape[1]!=len(mean) or not np.isfinite(x).all() or np.any(np.abs(x)>1e6):raise ValueError('matched finite candidate features required')
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


def arbitrate(model,features,incumbent,errors,*,max_regret=.12):
    """Retain incumbent unless supported learned PURPOSE gain clears error.

    Errors are empirical held-out diagnostics, not true-game confidence bounds.
    Purpose 0 is primary. Secondary heads never fabricate a primary zero event.
    """
    values,supported=predict(model,features);errors=np.asarray(errors,float)
    if type(incumbent) is not int or not 0<=incumbent<len(values) or errors.shape!=(values.shape[1],) or not np.isfinite(errors).all() or np.any(errors<0) or not np.isfinite(max_regret) or not 0<=max_regret<=1:
        raise ValueError('valid incumbent, empirical errors and bounded purpose tolerance required')
    best=int(values[:,0].argmax());gain=float(values[best,0]-values[incumbent,0]);changed=bool(supported and gain>max_regret+errors[0])
    return (best if changed else incumbent),dict(supported=supported,incumbent=incumbent,candidate=best,
        learned_gain=gain,calibration_error=float(errors[0]),tolerance=max_regret,changed=changed,
        uncertainty='empirical paired teacher residual; not true-game guarantee')
