"""Alternate design: one-step choice plus wider reference-purpose rollouts.

The two arms end their purpose evaluation at at most 18 owner decisions from
the root. Preference flows cover two versus one decisions; this is a whole
design comparison, not a pure causal estimate of sample count alone.
"""
import argparse,copy,json,sys,time
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.frontier_value import PolicyTail
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from tools.closed_loop_models import controller
from tools.run_foundation_audit import prepare,GAMES
from tools.run_contingent_coverage import cases
from tools.run_frontier_value import owned,verify,pool_run,evaluate_episode,crps


def episode(job):
    case,profile,seed,arm=job
    if arm=='proxy':return evaluate_episode((case,profile,seed,arm,None))
    start=time.perf_counter();b,w,prefix=prepare(case,profile,seed);actor=owned(b,w)
    ctrl=controller(b,horizon=1,nodes=4096,outcomes=4096)
    ctrl.frontier=PolicyTail(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,assess=lambda w:b.goal(w).record(),
        request_for=ctrl.request_for,feedback=ctrl.feedback,model_id=ctrl.model_id,target=b.target,seeds=tuple(range(701,709)),steps=17,record=b.record)
    trace=[]
    while not b.terminal(w):
        c=b.observe(b.project(w),actor.state);req=request_for(c,case['game']);cap={};before=b.record(w);saved=digest(actor.record())
        d=DecisionLoop.decide_batch([(actor,req)],False,ctrl.planner(actor,req,b.project(w),captured=cap))[0];key=d['decision']['action_id']
        prediction=None if 'node' not in cap else [(p,g['value']) for p,_,_,g,_ in cap['node']['returns']]
        actor.abandon(d['ticket']);w,flow=b.actual(w,key,seed)
        trace.append(dict(before=before,root=key,after=b.record(w),flow=flow,goal=b.goal(w).record(),prediction=prediction,
            tree=cap['audit'],owner_before=saved,owner_after=digest(actor.record()),decision=d['decision']))
        if len(trace)>1000:raise AssertionError('unsettled rollout episode')
    verify(b,prefix+trace,seed,w);goal=b.goal(w).record();loss=[crps(t['prediction'],goal['value']) for t in trace if t['prediction'] is not None]
    return dict(case=case,profile=profile['id'],seed=seed,arm=arm,goal=goal,prefix=prefix,trace=trace,replay=True,
        actions=[t['root'] for t in trace],mean_terminal_crps=sum(loss)/len(loss) if loss else None,elapsed=time.perf_counter()-start)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--workers',str(a.workers)])
    selected=[c for c in cases('confirmation') if c['game'] in GAMES];jobs=[]
    for c in selected:
        for i in range(4):
            cc=dict(c,opening_steps=i,opening_seed=50000+i)
            if cc['game'] in ('auction_contest','hagetaka','no_thanks'):cc['deal']=cc.get('deal',0)+29*i
            for profile in profiles():
                for arm in ('proxy','rollout8'):jobs.append((cc,profile,49000+i,arm))
    pre=dict(version='frontier-wider-rollout-comparison-v1',jobs=jobs,source_snapshot=digest(json.loads((root/'source_snapshot.json').read_text())),
        hypothesis='wider reference completion at a shallower tree can improve actual choices',
        settings=dict(proxy_tree=2,rollout_tree=1,root_seeds=[401,402],tail_seeds=list(range(701,709)),tail_steps=17,regret=.15,rigidity=.5),
        limits=['whole design comparison; prefix preferences and objective time weights also differ',
            'fresh outcome/opening seeds do not make deterministic repeated starting worlds independent',
            'same authored opponent models and scenario classes; not unknown-game generalization',
            'tail reference remains reflex; actual actor replans every decision'])
    (root/'preregister.json').write_text(json.dumps(pre,indent=2)+'\n');rows,errors,elapsed=pool_run(jobs,episode,root,a.workers)
    report=dict(planned=len(jobs),complete=len(rows),errors=errors,elapsed=elapsed,comparison=[],setup_only=sum(not r['trace'] for r in rows),decisions=sum(len(r['trace']) for r in rows))
    if not errors:
        for game in GAMES:
            pairs=[]
            for r in rows:
                if r['case']['game']!=game or r['arm']!='rollout8':continue
                b=next(x for x in rows if x['case']==r['case'] and x['profile']==r['profile'] and x['seed']==r['seed'] and x['arm']=='proxy')
                if not r['trace'] or not b['trace']:
                    assert not r['trace'] and not b['trace'];continue
                pairs.append(dict(case=r['case'],profile=r['profile'],seed=r['seed'],before=b['goal']['value'],after=r['goal']['value'],delta=r['goal']['value']-b['goal']['value'],actions_changed=r['actions']!=b['actions']))
            report['comparison'].append(dict(game=game,pairs=len(pairs),better=sum(p['delta']>0 for p in pairs),worse=sum(p['delta']<0 for p in pairs),same=sum(p['delta']==0 for p in pairs),paired=pairs,
                statuses={arm:dict(Counter(r['goal']['status'] for r in rows if r['case']['game']==game and r['arm']==arm and r['trace'])) for arm in ('proxy','rollout8')}))
        report['declines']=sum(not t['tree']['complete'] for r in rows for t in r['trace'])
    (root/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='comparison'}),flush=True)
    if errors:raise RuntimeError('wider rollout failures retained')


if __name__=='__main__':main()
