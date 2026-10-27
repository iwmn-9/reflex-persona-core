"""Replay historical actual learning; compare only fully audited research pairs."""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.model_observer import resolve_feedback
from tools.owned_checkpoint import encoded, sha
from tools.run_adverse_recovery import game_for


def historical_diagnosis(plan):
    path=ROOT/'evidence/long_horizon/continuation_selected_confirmation/runs.jsonl.gz'
    wanted={s['id'] for s in plan['specs']};rows=[]
    with gzip.open(path,'rt') as f:
        for line in f:
            data=json.loads(line)
            if data['spec']['id'] in wanted:rows.append(data)
    output=[]
    for data in rows:
        spec=data['spec'];g=game_for(spec);w=g.start();a=g.actor(g.context(w),learning='segmented')
        checks=[]
        for t in data['result']['trace']:
            assert w==t['before'] and digest(a.record())==t['before_owner']
            c=g.context(w,a.state);p=a.predictor.predict(c)
            checks.append(dict(tick=t['tick'],prediction=p,truth=t['truth'],response=t['response'],
                expected_brier_excess=sum((p[k]-v)**2 for k,v in t['truth'].items()),
                observed_log_loss=-math.log(p[t['response']]),
                dominant_correct=max(p,key=p.get)==max(t['truth'],key=t['truth'].get)))
            r=DecisionLoop.decide_batch([(a,g.request(c))],False,_counterfactual_roots=(t['action'],))[0]
            after,flow=g.transition(w,t['action'],t['response'])
            resolve_feedback(a,r['ticket'],g.feedback(w,after,t['action'],c,flow))
            assert after==t['after'] and digest(a.record())==t['owner']
            w=after
        windows=[]
        for begin in range(0,128,8):
            subset=checks[begin:begin+8]
            if subset:
                windows.append(dict(start=begin,end=begin+len(subset),
                    expected_brier_excess=sum(r['expected_brier_excess'] for r in subset)/len(subset),
                    dominant_correct=sum(r['dominant_correct'] for r in subset)))
        output.append(dict(id=spec['id'],arm='stable-H32' if data['retry'] else 'original-H16',
            score=data['result']['score'],replayed=len(checks),windows=windows,predictions=checks))
    paired=[]
    for name in sorted(wanted):
        arms=[r for r in output if r['id']==name];assert len(arms)==2
        paired.append(dict(id=name,same_predictions=arms[0]['predictions']==arms[1]['predictions']))
    return dict(source_sha256=sha(path.read_bytes()),cases=output,pairs=paired,
        interpretation='Truth is evaluator-only. These proper-score diagnostics are not causal intelligence or personality decompositions.')


def summarize(root, plan):
    root=Path(root);pairs=[];progress=[];runtimes=set()
    for spec in plan['specs']:
        arms={}
        for arm in plan['arms']:
            d=root/f"adverse-{spec['id']}-{arm}"
            if not (d/'status.json').exists():
                progress.append(dict(id=spec['id'],arm=arm,state='missing',actual_steps=0));continue
            status=json.loads((d/'status.json').read_bytes())
            progress.append(dict(id=spec['id'],arm=arm,**{k:status[k] for k in ('state','actual_steps','complete')}))
            if not status['complete']:continue
            assert status['state']=='complete'
            checkpoint=json.loads((d/'checkpoint.json').read_bytes());p=checkpoint['payload']
            assert sha(encoded(p))==checkpoint['sha256']
            assert status['checkpoint_sha256']==sha((d/'checkpoint.json').read_bytes())
            r=json.loads((d/'result.json').read_bytes())
            assert r['complete'] and r['trace']==p['trace'] and r['configuration']['spec']==spec
            assert r['configuration']['arm']==arm and r['final']==p['world']
            assert r['replayed_actual_steps']==len(r['trace'])==status['actual_steps']
            assert r['checks']==dict(fixed_axes=len(r['trace']),owner_parity=len(r['trace']),restored=len(r['trace']))
            g=game_for(spec);w=g.start()
            for t in r['trace']:
                assert w==t['before'];w,_=g.transition(w,t['action'],t['response']);assert w==t['after']
            assert g.terminal(w) and r['score']==g.goal(w).value
            assert r['conditional_increment']==sum(t['conditional_increment'] for t in r['trace'])
            arms[arm]=r;runtimes.add(encoded(r['runtime']))
        if len(arms)!=len(plan['arms']):continue
        a,b=arms[plan.get('reference_arm','stable')],arms[plan.get('candidate_arm','stratified')]
        assert a['runtime']==b['runtime']
        assert a['configuration']['registration']==b['configuration']['registration']
        common=min(len(a['trace']),len(b['trace']))
        assert [t['response'] for t in a['trace'][:common]]==[t['response'] for t in b['trace'][:common]]
        pairs.append(dict(id=spec['id'],game=spec['game'],profile=spec['profile'],
            score_delta=b['score']-a['score'],conditional_delta=b['conditional_increment']-a['conditional_increment'],
            changed_actions=sum(x['action']!=y['action'] for x,y in zip(a['trace'],b['trace'])),
            scores={k:v['score'] for k,v in arms.items()},steps={k:len(v['trace']) for k,v in arms.items()}))
    groups=defaultdict(list)
    for p in pairs:groups[p['game']].append(p)
    families={k:{v:sum(r[v] for r in rows)/len(rows) for v in ('score_delta','conditional_delta')} for k,rows in groups.items()}
    complete=len(pairs)==len(plan['specs'])
    checks=dict(complete=complete,all_families=set(families)=={'auction','combat','resources','gambling'},
        no_family_loss=all(v>=-1e-12 for r in families.values() for v in r.values()),
        concordant_adverse_improvement=any(r['id'] in plan['concordant_adverse'] and r['score_delta']>1e-12 and r['conditional_delta']>1e-12 for r in pairs))
    return dict(complete=complete,pairs_complete=len(pairs),pairs_registered=len(plan['specs']),
        progress=progress,pairs=pairs,families=families,gate_checks=checks,development_gate_passed=all(checks.values()),
        adoption=False,independent_confirmation='not started',runtimes=[json.loads(v) for v in sorted(runtimes)])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--root')
    p.add_argument('--historical',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_bytes())
    result=historical_diagnosis(plan) if a.historical else summarize(a.root,plan)
    Path(a.output).write_bytes(encoded(result))
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','predictions','progress','pairs')},ensure_ascii=True))
