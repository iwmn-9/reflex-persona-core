"""Known-model embedding equivalence across existing games; not strength data.

The two-sample kernels are a forecast bank, NOT exhaustive support for real
observations. This probe does not claim to learn those games' latent models.
"""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.belief_planning import BeliefKernel,HiddenOutcome
from reflex.closed_loop import ClosedLoop
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from tools.closed_loop_models import controller
from tools.run_frontier_value import owned
from tools.run_foundation_audit import prepare,GAMES
from tools.run_contingent_coverage import cases


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,[])
    jobs=[(c,profile) for stage in ('development','confirmation') for c in cases(stage) if c['game'] in GAMES for profile in profiles()]
    (root/'preregister.json').write_text(json.dumps(dict(jobs=jobs,kind='64 known-model forecast embeddings; no strength or real observation support claim'),indent=2)+'\n')
    rows=[]
    for case,profile in jobs:
        b,w,prefix=prepare(case,profile,64000);w=b.project(w);before=owned(b,w);base=controller(b,horizon=2,outcomes=4096)
        k=BeliefKernel(names=('known',),model=lambda h,s,key:[HiddenOutcome(t.probability,t.state,t.flow,h) for t in base.transitions(s,key)],
            public_record=b.record,model_id=base.model_id,owner='owner')
        s=k.start(w,(1.,));wrapped=ClosedLoop(observe=lambda s,m:b.observe(s.public,m),transitions=k,
            terminal=lambda s:b.terminal(s.public),assess=lambda s:b.goal(s.public).record(),record=k.record,
            request_for=base.request_for,feedback=lambda old,new,key,c,row:base.feedback(old.public,new.public,key,c,row),
            unit=base.unit,target=base.target,model_id=k.identity,budget=base.budget,max_regret=base.max_regret)
        choices=[];returns=[];states=[]
        for ctrl,state in ((base,w),(wrapped,s)):
            own=copy.deepcopy(before);rq=request_for(b.observe(w,own.state),case['game']);cap={}
            result=DecisionLoop.decide_batch([(own,rq)],False,ctrl.planner(own,rq,state,captured=cap))[0]
            assert cap['audit']['complete'];choices.append(result['decision']['action_id']);own.abandon(result['ticket']);states.append(digest(own.record()))
            # Observational regrouping can reorder the same return atoms.
            values={}
            for p,flows,actions,g,ended in cap['node']['returns']:
                atom=digest([flows,actions,g,ended]);values[atom]=values.get(atom,0.)+p
            returns.append(values)
        assert choices[0]==choices[1] and states[0]==states[1]
        assert returns[0].keys()==returns[1].keys() and all(abs(returns[0][key]-returns[1][key])<1e-12 for key in returns[0])
        rows.append(dict(case=case,profile=profile['id'],action=choices[0],same_owner=True,same_return_distribution=True))
        if len(rows)%8==0:print(len(rows),'/',len(jobs),flush=True)
    (root/'evaluation.json').write_text(json.dumps(dict(complete=len(rows),rows=rows),indent=2)+'\n');print('all equivalent',flush=True)


if __name__=='__main__':main()
