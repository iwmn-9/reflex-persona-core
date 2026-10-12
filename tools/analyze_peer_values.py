"""Separate lower resource score from realized persona value fulfillment."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.encounters import EncounterWorld
from tools.run_public_peer_reliability import mixed
from reflex.core import digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();runs=[]
    with (Path(a.cohort)/'trajectories.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line);w=EncounterWorld(r['seed'],r['domain'],r['layout'],'public')
            if r['variant']=='mixed':mixed(w)
            stats={own:dict(strongest_axis=max(w.profiles[own]['values'],key=w.profiles[own]['values'].get),strongest_flow=0.,weighted_flow=0.,cooperate=0,claim=0,alone=0,rest=0) for own in w.actors}
            for tick in r['trace']:
                rows,events=w.resolve(tick['chosen']);assert rows==tick['rows']
                for row in rows:
                    own=row['npc'];s=stats[own];profile=w.profiles[own];kind=row['kind']
                    effect=w.perceived(kind,row['other_kind'],row['matched'],1.)
                    if row['luck_failure']:effect['values']={}
                    s['strongest_flow']+=effect['values'].get(s['strongest_axis'],0.)
                    s['weighted_flow']+=sum(profile['values'].get(k,0.)*v for k,v in effect['values'].items())
                    if kind in ('cooperate','claim','alone','rest'):s[kind]+=1
            runs.append(dict(domain=r['domain'],layout=r['layout'],seed=r['seed'],variant=r['variant'],arm=r['peer_arm'],
                stats=stats,goals={x['npc']:x['goal']['value'] for x in r['actors']}))
    groups={}
    for r in runs:groups.setdefault(digest([r['domain'],r['layout'],r['seed'],r['variant']]),{})[r['arm']]=r
    pairs=[]
    for group in groups.values():
        if set(group)!={'neutral','public'}:raise ValueError('complete paired worlds required')
        x,y=group['neutral'],group['public']
        for own,b in x['stats'].items():
            c=y['stats'][own];pairs.append(dict(domain=x['domain'],layout=x['layout'],seed=x['seed'],variant=x['variant'],npc=own,
                goal_delta=y['goals'][own]-x['goals'][own],strongest_flow_delta=c['strongest_flow']-b['strongest_flow'],
                weighted_flow_delta=c['weighted_flow']-b['weighted_flow'],before=b,after=c))
    bad=[p for p in pairs if p['goal_delta']<-1e-12]
    result=dict(pairs=len(pairs),lower_goal=len(bad),lower_goal_higher_strongest_flow=sum(p['strongest_flow_delta']>1e-12 for p in bad),
        lower_goal_higher_weighted_flow=sum(p['weighted_flow_delta']>1e-12 for p in bad),paired=pairs,
        limitation='realized authored value flows, not whole experienced utility or proof a resource-score regression was rational')
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='paired'}))


if __name__=='__main__':main()
