"""Prove the explicit unlearned continuation shortcut against owned execution."""
import argparse,copy,json,sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.contingent_games import make_bridge
from reflex.core import Policy,character_key,digest
from reflex.decision_loop import DecisionLoop
from reflex.deliberation import select
from reflex.forecast_control import ForecastBudget,ForecastController
from reflex.integrated_trials import request_for
from reflex.laboratory import profiles
from reflex.model_observer import ModelFeedback
from tools.run_contingent_coverage import cases
from tools.run_forecast_reliability import GAMES


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();rows=[]
    for case in cases('development'):
        if case['game'] not in GAMES:continue
        for profile in profiles():
            b=make_bridge(profile,case);public=b.project(b.start());c=b.observe(public)
            p=Policy(principle_priority='personal',commitments={character_key(c):.5});actor=DecisionLoop(c,policy=p)
            req_for=lambda cc:request_for(cc,case['game'],learn=False);req=req_for(c);saved=digest(actor.record())
            controller=ForecastController(observe=b.observe,advance=b.modeled_advance,terminal=b.terminal,
                assess=lambda s:b.goal(s).record(),request_for=req_for,feedback=lambda *a:ModelFeedback(None),unit=b.unit,target=b.target)
            for levels in (0,1):
                budget=ForecastBudget(horizon=2,seeds=(401,402),future_horizon=2,future_levels=levels)
                f,x=controller.evaluate(c,public,actor,req,budget)
                g,y=controller.evaluate(c,public,actor,req,replace(budget,direct_continuation=True))
                assert f.contexts==g.contexts and f.purpose==g.purpose and f.roots==g.roots and f.return_frame==g.return_frame
                assert select([c],f,p)[0]==select([c],g,p)[0]
                for key in x:
                    for arow,brow in zip(x[key],y[key]):
                        assert {k:v for k,v in arow.items() if k!='modeled_observations'}=={k:v for k,v in brow.items() if k!='modeled_observations'}
                        if levels:assert arow['modeled_observations']==brow['modeled_observations']
                assert digest(actor.record())==saved
                rows.append(dict(case=case,profile=profile['id'],future_levels=levels,roots=len(x),equal=True))
                print(len(rows),case['game'],profile['id'],levels,flush=True)
    Path(a.output).write_text(json.dumps(dict(conditions=len(rows),rows=rows,semantics='declared no-learning/no-social/no-pressure/no-progress Policy requests only'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
