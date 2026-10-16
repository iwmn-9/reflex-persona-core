"""Revisit the previously registered tick-67 failure without new rule weights."""
import copy,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.laboratory import PROFILES
from reflex.model_observer import resolve_feedback
from reflex.closed_loop import owned_rollout
from reflex.adaptive_dynamics import PublicEvent
from reflex.purpose_plan import Goal
from reflex.decision_loop import DecisionLoop
from tools.run_prediction_recovery import RecoveryAdapter,truth,sample


def probe(root):
    rows=[json.loads(x) for x in (root/'trajectories.jsonl').read_text().splitlines()]
    r=next(x for x in rows if (x['game'],x['profile'],x['regime'],x['arm'])==('auction','steady','outside','recovery'))
    a=RecoveryAdapter('auction',PROFILES[1],r['seed']%2);w=a.start();own=a.actor_for(a.raw(w),'recovery')
    draws=np.random.default_rng(r['seed']).random(r['turns']);fb=lambda b,e,k,c:a.feedback(b,e,k,c,'channels','censored')
    for t in range(67):
        if a.terminal(w):w=a.start(w['round']+1,t)
        c=a.raw(w,own.state);res=own.decide(a.request(c),False);key=res['decision']['action_id']
        n,_,_=a.transition(w,key,sample(truth(a.symbols(),t,r['turns'],'outside'),draws[t]))
        resolve_feedback(own,res['ticket'],fb(w,n,key,c));w=n
        assert digest(own.record())==r['trace'][t]['owner_digest']
    c=a.raw(w,own.state);rq=a.request(c)
    def events(s,k):return [PublicEvent(z,*a.transition(s,k,z)[:2]) for z in a.legal(a.forecast_context(s))]
    ctrl=owned_rollout(own,observe=a.raw,events=events,terminal=a.terminal,
        assess=lambda s:Goal(a.rules.target,'scored' if a.terminal(s) else 'running',a.goal(s)).record(),
        record=lambda s:s,request_for=a.request,feedback=lambda b,e,k,cc,f:fb(b,e,k,cc),
        model_id=a.rules.model_id,target=a.rules.target,tail_steps=6,seeds=tuple(range(701,709)))
    captured={};selected=copy.deepcopy(own)
    res=DecisionLoop.decide_batch([(selected,rq)],False,ctrl.planner(own,rq,w,captured=captured))[0]
    f=captured['node']['forecast']
    print(json.dumps(dict(old=r['trace'][67]['action'],new=res['decision']['action_id'],
        purpose={f.roots[k][0]:v for k,v in f.purpose.items()},audit=res.get('deliberation')),indent=2))


if __name__=='__main__':probe(Path(sys.argv[1]))
