"""Small bounded ranker calibration; seed smoke and reviewed training kept separate."""
import json
from pathlib import Path
import numpy as np
from .core import Policy, compile_batch, digest, VERSION
from .teachers import write_json


def fit(examples, output, seed_smoke=False, steps=200):
    if not examples: raise ValueError("no training examples")
    allowed="pending_user_alpha_review" if seed_smoke else "human_alpha_accepted"
    for row in examples:
        if row["quality"]!=allowed: raise ValueError("unreviewed data cannot enter reviewed training")
        if seed_smoke and row["source"]!="authored_design_seed": raise ValueError("seed smoke accepts authored seeds only")
    train=[r for r in examples if r.get("split")=="train"]
    if not train: raise ValueError("explicit family-level train split required")
    # Enforce entire counterfactual families in one partition, not NPC-level leakage.
    assignment={}
    for row in examples:
        if row.get("split") not in ("train","validation","test"): raise ValueError("explicit split required")
        old=assignment.setdefault(row["family"],row["split"])
        if old!=row["split"]: raise ValueError("family leakage across partitions")
    batch=compile_batch([r["context"] for r in train]); base=Policy().decide(batch,False)
    labels=np.zeros_like(base.scores)
    for i,row in enumerate(train):
        for key in row["preferred"]:
            if key not in batch.ids[i]: raise ValueError("invalid training action")
            j=batch.ids[i].index(key)
            if not base.eligible[i,j]: raise ValueError("label conflicts with legal/failure/principle gate")
            labels[i,j]=1
        if labels[i].sum()==0: raise ValueError("empty training targets")
        labels[i]/=labels[i].sum()
    theta=np.zeros(8); losses=[]; temperature=.15
    for step in range(steps):
        delta=np.einsum('nad,d->na',base.features,theta)
        logits=(base.scores+np.clip(delta,-.08,.08))/temperature
        logits=np.where(base.eligible,logits,-np.inf)
        exp=np.exp(logits-np.max(logits,axis=1,keepdims=True)); probability=exp/exp.sum(1,keepdims=True)
        loss=-float(np.sum(labels*np.log(np.maximum(probability,1e-15)))/len(train))
        active=(abs(delta)<.08)[:,:,None]
        grad=np.einsum('na,nad->d',probability-labels,base.features*active)/(len(train)*temperature)+.05*theta
        theta=np.clip(theta-.025*grad,-.08,.08)
        if step in (0,steps-1): losses.append(loss)
    artifact=dict(version=VERSION,architecture="8 coefficient bounded ranker residual; no language generation",
        artifact_type="pipeline_smoke" if seed_smoke else "reviewed_ranker",approved_for_default=False,
        coefficients=theta.tolist(),training_hash=digest(examples),examples=len(examples),train_rows=len(train),
        steps=steps,initial_loss=losses[0],last_loss=losses[-1],evaluation=evaluate(examples,Policy(theta)))
    write_json(output,artifact)
    return artifact


def load(path, allow_smoke=False):
    artifact=json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact["version"]!=VERSION: raise ValueError("model contract mismatch")
    if artifact["artifact_type"]=="pipeline_smoke" and not allow_smoke: raise ValueError("seed smoke is not a reviewed model")
    return Policy(artifact["coefficients"])


def evaluate(examples, policy=None):
    policy=policy or Policy(); result={}
    for split in ("train","validation","test"):
        rows=[r for r in examples if r.get("split")==split]
        if not rows: continue
        batch=compile_batch([r["context"] for r in rows]); decisions=policy.decide(batch,False).records(batch)
        good=sum(d["action_id"] in row["preferred"] for row,d in zip(rows,decisions))
        result[split]=dict(rows=len(rows),families=len({r["family"] for r in rows}),correct=good,accuracy=good/len(rows))
    return result
