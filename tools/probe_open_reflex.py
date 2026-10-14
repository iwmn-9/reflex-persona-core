"""Observed predictive memory into ordinary mixed-game numeric batch decisions."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy, character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from tools.open_model_games import OpenGame


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--study',required=True);p.add_argument('--frozen',action='store_true');a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--study',str(Path(a.study).resolve())])
    rows=[json.loads(l) for l in (Path(a.study)/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()]
    selected=[r for r in rows if r['scenario']=='novel' and r['arm']=='adaptive' and r['seed']==71000]
    assert len(selected)==16
    (root/'preregister.json').write_text(json.dumps(dict(selection='all four games and personalities, first development seed, novel behavior; same next public start with cold or factual learned memory',
        input=digest(selected),purpose='no planner: batch/single parity, factual memory isolation, and influence on immediate choices; no population throughput claim'),indent=2)+'\n')
    findings=[]
    for profile in profiles():
        for warm in (False,True):
            prepared=[]
            for game in ('auction','resources','combat','gambling'):
                r=next(r for r in selected if r['game']==game and r['profile']==profile['id']);g=OpenGame(game,profile)
                memory=g.kernel.learner.restore(r['final_memory']) if warm else None;s=g.start(r['episodes'],memory);c=g.observe(s)
                policy=Policy(principle_priority='personal',commitments={character_key(c):.5});owner=DecisionLoop(c,policy=policy);request=request_for(c,'open-model')
                single=copy.deepcopy(owner).decide(request,False)['decision']['action_id']
                prepared.append((g,s,owner,request,single,g.kernel.record(s)))
            batch=DecisionLoop.decide_batch([(own,req) for _,_,own,req,_,_ in prepared],False)
            for (g,s,owner,request,single,before),result in zip(prepared,batch):
                chosen=result['decision']['action_id']
                assert chosen==single and chosen in g.keys(s)
                assert g.kernel.record(s)==before and owner.personality==request.context['personality'] and owner.values==request.context['values']
                owner.abandon(result['ticket'])
                findings.append(dict(game=g.game,profile=profile['id'],warm=warm,selected=chosen,single=single,
                    public=s.public,memory_observations=s.memory.observations,forecast=g.kernel.forecast(s,chosen)[4]))
    result=dict(cases=len(findings),batches=8,npcs_per_batch=4,batch_single_equal=True,no_counterfactual_memory_written=True,findings=findings)
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='findings'}),flush=True)


if __name__=='__main__':main()
