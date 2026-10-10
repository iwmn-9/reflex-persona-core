"""Freeze matched-action learner, then score unseen alternative execution labels."""
import argparse,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from reflex.paired_value import fit,predict,calibrate,arbitrate
from tools.teach_actual_alternatives import numeric_features


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def train(data,root):
    data=Path(data);root=Path(root);root.mkdir(parents=True,exist_ok=True)
    dataset=load(data);remaining=dataset['plan'].get('remaining',0)
    plan=dict(train_seeds=list(range(7800,7808)),calibration_seeds=list(range(7808,7816)),test_seeds=list(range(8200,8216)),
        alpha=10.,error_quantile=.9,max_regret=.12,
        kind='matched action ranking, not factual selected-action prediction',
        scope=('No Thanks current_card_only' if not remaining else 'No Thanks first nonforced turn per NPC with 1..'+str(remaining)+' remaining')+'; public feature encoder version in teach_actual_alternatives',
        weighting='each NPC episode has total fit weight one; all comparisons within a matched case',
        primary='arbitrated root minus incumbent true simulated winner regret on held-out series',
        secondary=['greedy learned ranking','progress ranking','per-profile and coverage'],
        selector='primary learned gain must exceed .12 plus held-out empirical pair error; no actual test labels enter choice',
        limitations=['empirical error quantile is not statistical coverage guarantee','changed future policy must be separately validated'],
        input_sha256=hashlib.sha256(data.read_bytes()).hexdigest())
    if (root/'preregister.json').exists():
        assert load(root/'preregister.json')==plan
        save(root/'feature_encoding_repair.json',dict(reason='raw public needs are supported/enabled/deficit packets, not scalar deficits',
            normalization='expand each packet to supported, enabled and deficit; None deficit encodes zero with the explicit support flags',
            target_or_rollout_changes=0,fit_hyperparameters_changed=False,test_case_labels_inspected=False))
    else:save(root/'preregister.json',plan)
    rows=dataset['cases'];train=[r for r in rows if r['seed'] in plan['train_seeds']];cal=[r for r in rows if r['seed'] in plan['calibration_seeds']]
    counts=Counter((r['seed'],r['encounter'],r['actor']) for r in train)
    cases=lambda rs:[(numeric_features(r['features']),r['targets']) for r in rs]
    model=fit(cases(train),alpha=plan['alpha'],case_weights=[1/counts[r['seed'],r['encounter'],r['actor']] for r in train])
    errors=calibrate(model,cases(cal),quantile=plan['error_quantile'])
    fitted=dict(model=model,errors=errors,scope='current_card_only' if not remaining else 'last_six_cards',feature_encoder='teach_actual_alternatives.public_features',max_regret=.12)
    save(root/'model.json',fitted)
    save(root/'freeze.json',dict(training_cases=len(train),calibration_cases=len(cal),errors=errors,
        model_sha256=hashlib.sha256((root/'model.json').read_bytes()).hexdigest(),test_labels_accessed=False))
    print('frozen',len(train),len(cal),'errors',errors,flush=True)


def evaluate(data,root):
    data=Path(data);root=Path(root);plan=load(root/'preregister.json');freeze=load(root/'freeze.json')
    assert hashlib.sha256((root/'model.json').read_bytes()).hexdigest()==freeze['model_sha256']
    fitted=load(root/'model.json');rows=load(data)['cases'];assert sorted({r['seed'] for r in rows})==plan['test_seeds']
    outcomes=[]
    for r in rows:
        features=numeric_features(r['features']);p,supported=predict(fitted['model'],features);truth=np.asarray(r['targets']);inc=r['actions'].index(r['source_action'])
        chosen,gate=arbitrate(fitted['model'],features,inc,fitted['errors'],max_regret=fitted['max_regret'])
        greedy=int(p[:,0].argmax());progress=int(p[:,1].argmax())
        outcomes.append(dict(seed=r['seed'],encounter=r['encounter'],actor=r['actor'],profile=r['profile'],tick=r['tick'],
            selected=r['actions'][chosen],incumbent=r['source_action'],gate=gate,
            credit_regret={k:float(truth[:,0].max()-truth[a,0]) for k,a in (('incumbent',inc),('arbitrated',chosen),('greedy',greedy))},
            progress_regret={k:float(truth[:,1].max()-truth[a,1]) for k,a in (('incumbent',inc),('greedy_progress',progress))}))
    groups=defaultdict(list)
    for r in outcomes:groups[r['seed'],r['encounter'],r['actor']].append(r)
    series=defaultdict(list)
    for (seed,_,_),rs in groups.items():series[seed].append({k:float(np.mean([r['credit_regret'][k] for r in rs])) for k in ('incumbent','arbitrated','greedy')})
    means={seed:{k:float(np.mean([r[k] for r in rs])) for k in ('incumbent','arbitrated','greedy')} for seed,rs in series.items()}
    progress_series=defaultdict(list)
    for (seed,_,_),rs in groups.items():progress_series[seed].append({k:float(np.mean([r['progress_regret'][k] for r in rs])) for k in ('incumbent','greedy_progress')})
    progress_means={seed:{k:float(np.mean([r[k] for r in rs])) for k in ('incumbent','greedy_progress')} for seed,rs in progress_series.items()}
    improvements=np.array([r['incumbent']-r['arbitrated'] for r in means.values()]);rng=np.random.default_rng(82997)
    interval=np.quantile(improvements[rng.integers(len(means),size=(10000,len(means)))].mean(1),[.025,.975]).tolist()
    result=dict(plan=plan,model_sha256=freeze['model_sha256'],test_teacher_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),cases=outcomes,
        summary=dict(cases=len(rows),independent_series=len(means),supported=sum(r['gate']['supported'] for r in outcomes),
            changed=sum(r['gate']['changed'] for r in outcomes),mean_credit_regret={k:float(np.mean([r[k] for r in means.values()])) for k in ('incumbent','arbitrated','greedy')},
            arbitrated_credit_regret_improvement=float(improvements.mean()),paired_series_bootstrap_95=interval,
            better=sum(r['credit_regret']['arbitrated']<r['credit_regret']['incumbent']-1e-12 for r in outcomes),
            worse=sum(r['credit_regret']['arbitrated']>r['credit_regret']['incumbent']+1e-12 for r in outcomes),
            mean_progress_regret={k:float(np.mean([r[k] for r in progress_means.values()])) for k in ('incumbent','greedy_progress')}),
        limitations=['late-game matched teacher labels, not full-game strength','same known game and policies',
                     'model frozen before held-out label inspection; controller future still incumbent in this diagnostic'])
    save(root/'evaluation.json',result);print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('train','evaluate'));p.add_argument('--data',required=True);p.add_argument('--root',required=True);a=p.parse_args()
    (train if a.mode=='train' else evaluate)(a.data,a.root)
