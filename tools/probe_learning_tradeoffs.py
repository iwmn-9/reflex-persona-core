"""Inspect all registered objective regressions, without tuning the chooser."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy,character_key,compile_batch,digest
from reflex.decision_loop import DecisionLoop
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from tools.learning_value_games import LearningGame


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--study',required=True);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--study',str(Path(a.study).resolve())])
    old=Path(a.study);rows=[json.loads(x) for x in (old/'trajectories.jsonl').read_text().splitlines()];pairs=json.loads((old/'evaluation.json').read_text())['pairs']
    cases={digest([r['case'],r['profile']]):r for r in pairs if r['delta']< -1e-9 and r.get('candidate','adaptive')=='adaptive'}
    (root/'preregister.json').write_text(json.dumps(dict(source=old.name,source_rows=digest(rows),cases=list(cases.values()),selection='all objective regressions, first divergence in lowest-seed complete execution'),indent=2)+'\n')
    findings=[]
    for r in cases.values():
        profile=next(p for p in profiles() if p['id']==r['profile']);g=LearningGame(r['case'],profile);s=g.start();c=g.observe(s)
        policy=Policy(principle_priority='personal',commitments={character_key(c):.5});owner=DecisionLoop(c,policy=policy);ctrl=g.controller('adaptive')
        adaptive=next(x for x in rows if x['case']==r['case'] and x['profile']==r['profile'] and x['arm']=='adaptive')
        baseline=next(x for x in rows if x['case']==r['case'] and x['profile']==r['profile'] and x['arm']==r['baseline'])
        for e in sorted(adaptive['executions'],key=lambda e:e['seed']):
            other=next(x for x in baseline['executions'] if x['seed']==e['seed'])
            differing=next((i for i,(a,b) in enumerate(zip(e['steps'],other['steps'])) if a['action']!=b['action']),None)
            if differing is not None:break
        else:raise AssertionError('registered regression without realized differing choice')
        for step in e['steps'][:differing+1]:
            req=request_for(g.observe(s,owner.state),'learning-value');cap={}
            before=copy.deepcopy(owner)
            d=DecisionLoop.decide_batch([(owner,req)],False,ctrl.planner(owner,req,s,remaining=g.limit-s.public['tick'],captured=cap))[0]
            assert d['decision']['action_id']==step['action']
            if s.public['tick']==e['steps'][differing]['before']['public']['tick']:
                f=cap['node']['forecast'];batch=compile_batch(f.contexts,outcome_limit=f.outcome_limit);score=policy.decide(batch,False,return_frame=f.return_frame)
                findings.append(dict(case=r['case'],profile=r['profile'],world=g.kernel.record(s),baseline=other['steps'][differing]['action'],selected=d['decision']['action_id'],
                    plans=[dict(root=f.roots[key][0],purpose=f.purpose[key],score=float(score.scores[0,j]),
                                features=score.features[0,j].tolist(),eligible=bool(score.eligible[0,j])) for j,key in enumerate(batch.ids[0])],
                    feature_order=['endpoint','primary_need','top_value','weighted_loss','cost','familiarity','switch','confidence']))
                break
            owner.abandon(d['ticket']);s=g.kernel.restore(step['after'],lambda p:p)
    (root/'evaluation.json').write_text(json.dumps(dict(findings=findings),indent=2)+'\n');print(json.dumps(findings),flush=True)


if __name__=='__main__':main()
