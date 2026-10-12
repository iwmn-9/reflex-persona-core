"""Compare repeated actual controls and probe the final strict shortcut."""
import argparse,copy,hashlib,json,sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import Policy,character_key,digest
from reflex.contingent_games import make_bridge
from reflex.decision_loop import DecisionLoop
from reflex.deliberation import select
from reflex.forecast_control import ForecastBudget,ForecastController
from reflex.integrated_trials import request_for
from reflex.model_observer import ModelFeedback
from reflex.laboratory import profiles
from tools.run_contingent_coverage import cases


def read(root,arm):
    rows={}
    with (root/'trajectories.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line)
            if r['arm']==arm:rows[digest([r['case'],r['profile'],r['seed']])]=r
    return rows


def actual(r):
    return dict(goal=r['goal'],actions=r['actions'],trace=[{k:t[k] for k in ('before','after','root','flow','goal','decision')} for t in r['trace']])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verification-root',required=True);a=ap.parse_args();v=Path(a.verification_root);replays=[]
    for first,second,arm in (('confirmation','settlement','short'),('settlement','challenger','checked')):
        x=read(v/f'forecast_reliability_{first}_20261012',arm);y=read(v/f'forecast_reliability_{second}_20261012',arm)
        assert set(y)<=set(x)
        for k in y:assert actual(x[k])==actual(y[k])
        replays.append(dict(first=first,second=second,arm=arm,episodes=len(y),whole_actual_records_equal=True))
    parity=[]
    for case in cases('confirmation'):
        if case['game'] not in ('auction_contest','gambling','cyclic_arena','shared_commons','branching_delivery','team_combat','empire_race','hagetaka'):continue
        profile=profiles()[0];b=make_bridge(profile,case);public=b.project(b.start());c=b.observe(public)
        p=Policy(principle_priority='personal',commitments={character_key(c):.5});actor=DecisionLoop(c,policy=p)
        req_for=lambda cc:request_for(cc,case['game'],learn=False)
        ctrl=ForecastController(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,
            assess=lambda s:b.goal(s).record(),request_for=req_for,feedback=lambda *args:ModelFeedback(None),unit=b.unit,target=b.target)
        saved=actor.record();budget=ForecastBudget(horizon=2,seeds=(401,402),future_levels=1)
        f,x=ctrl.evaluate(c,public,actor,req_for(c),budget);g,y=ctrl.evaluate(c,public,actor,req_for(c),replace(budget,direct_continuation=True))
        assert f.contexts==g.contexts and f.purpose==g.purpose and f.roots==g.roots
        assert select([c],f,p)[0]==select([c],g,p)[0]
        for k in x:
            assert [{q:w for q,w in r.items() if q!='modeled_observations'} for r in x[k]]==[{q:w for q,w in r.items() if q!='modeled_observations'} for r in y[k]]
            assert [r['modeled_observations'] for r in x[k]]==[r['modeled_observations'] for r in y[k]]
        assert actor.record()==saved;parity.append(dict(game=case['game'],equal=True));print(case['game'],flush=True)
    result=dict(repeated_controls=replays,final_strict_continuation_parity=parity,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'reflex').glob('*.py')})
    (v/'forecast-final-controls.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
