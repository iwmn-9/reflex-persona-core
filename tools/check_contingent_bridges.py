"""Early integration diagnostics; NOT frozen strength evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.contingent_games import BRIDGES,make_bridge
from reflex.contingent_search import search_forecast
from reflex.laboratory import profiles
from reflex.deliberation import select
from reflex.core import compile_batch


if __name__=='__main__':
    for name in BRIDGES:
        if len(sys.argv)>1 and name not in sys.argv[1:]:continue
        bridge=make_bridge(profiles()[0],dict(game=name));w=bridge.project(bridge.start());c=bridge.observe(w)
        print(name,'roots',len([a for a in c['actions'] if a['legal'] and not a['known_failure']]),flush=True)
        f=search_forecast(c,w,observe=bridge.observe,advance=bridge.modeled_advance,terminal=bridge.terminal,
            assess=lambda s:bridge.goal(s).record(),horizon=2,seeds=(401,),width=1,revisions=1,
            target=bridge.target,unit=bridge.unit,patch_budget=2,settlement_weight='absolute')
        decisions,vote=select([c],f);key=decisions[0]['action_id']
        after,flow=bridge.actual(bridge.start(),key,1230);compile_batch([bridge.observe(bridge.project(after),decisions[0]['next_state'])]) if not bridge.terminal(after) else None
        print(name,key,bridge.goal(after).record(),'evaluations',f.audit['contingent_search']['evaluated'],flush=True)
