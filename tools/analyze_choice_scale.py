"""Report paired purpose, real value flows, diversity, and every failed job."""
import argparse,hashlib,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.encounters import EncounterWorld
from tools.run_public_peer_reliability import mixed
from tools.analyze_forecast_reliability import comparison,stats
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summary_stats(pairs):
    result=stats(pairs);bad=[p for p in pairs if p['delta']<-1e-12]
    result['lower_goal_no_strongest_flow_gain']=sum(p['strongest_flow_delta']<=1e-12 for p in bad)
    result['lower_goal_higher_weighted_flow']=sum(p['weighted_value_flow_delta']>1e-12 for p in bad)
    return result
def load(root,partial):
    pre=json.loads((root/'preregister.json').read_text(encoding='utf-8'));file=root/'evaluation.json'
    rows=[];indices=set()
    with (root/'trajectories.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            raw=json.loads(line);index=raw['index'];assert index not in indices;indices.add(index)
            case,profile,seed,arm,horizon=pre['jobs'][index]
            assert (raw['case'],raw['profile'],raw['seed'],raw['arm'])==(case,profile['id'],seed,arm)
            values=profile['values'];top=max(values,key=values.get) if values else None;trace=raw.pop('trace')
            raw['actual_preference_flow']=dict(strongest_axis=top,strongest_flow=sum(t['flow']['values'].get(top,0.) for t in trace) if top else 0.,
                weighted_value_flow=sum(sum(values.get(k,0.)*v for k,v in t['flow']['values'].items()) for t in trace),cost=sum(t['flow']['cost'] for t in trace))
            scores=[s for t in trace for s in t['forecast_scores']];raw['forecast_loss']=statistics.mean(s['forecast_loss'] for s in scores)
            raw['purpose_tradeoffs']=sum(t['deliberation']['selected_purpose']<t['deliberation']['best_purpose']-1e-12 for t in trace)
            raw['band_decisions']=sum('purpose_band' in t.get('valuation',{}) for t in trace)
            rows.append(raw)
    if not file.exists():
        if not partial:raise ValueError('incomplete cohort cannot be final evidence')
        errors=[]
    else:
        result=json.loads(file.read_text(encoding='utf-8'));errors=result['errors']
        assert len(rows)==result['episodes'] and result['planned']==len(pre['jobs'])==len(rows)+len(errors)
        assert indices|{e['index'] for e in errors}==set(range(len(pre['jobs'])))
    receipts=dict(name=root.name,complete=file.exists(),planned=len(pre['jobs']),completed=len(rows),errors=errors,
        preregister=pre,files_sha256={p.name:sha(p) for p in (root/'preregister.json',root/'trajectories.jsonl',file) if p.exists()})
    for name,value in pre['source_hashes'].items():assert sha(root/'source'/name)==value
    return rows,receipts
def analyze_social(root,partial=False):
    pre=json.loads((root/'preregister.json').read_text(encoding='utf-8'));file=root/'evaluation.json';worlds=[];indices=set()
    if file.exists():
        e=json.loads(file.read_text(encoding='utf-8'));assert e['planned']==len(pre['jobs'])==e['worlds']+len(e['errors'])
    elif partial:e=dict(errors=[])
    else:raise ValueError('incomplete social cohort cannot be final evidence')
    for name,value in pre['source_hashes'].items():assert sha(root/'source'/name)==value
    with (root/'trajectories.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line);assert r['index'] not in indices;indices.add(r['index'])
            assert pre['jobs'][r['index']]==[r['domain'],r['layout'],r['seed'],r['variant'],r['peer_arm']]
            w=EncounterWorld(r['seed'],r['domain'],r['layout'],'public')
            if r['variant']=='mixed':mixed(w)
            counts={own:dict(strongest_axis=max(w.profiles[own]['values'],key=w.profiles[own]['values'].get),strongest_flow=0.,weighted_value_flow=0.,cost=0.,cooperate=0,claim=0,alone=0,rest=0) for own in w.actors}
            predictive=[]
            for tick in r['trace']:
                rows,_=w.resolve(tick['chosen']);assert rows==tick['rows']
                for update in tick.get('peer_updates',[]):
                    score=update['score']
                    if score['scored']:predictive.append(score['gain'])
                    elif score['reason']=='identical forecasts':predictive.append(0.)
                    else:raise ValueError('unexpected unscored categorical observation')
                for row in rows:
                    own=row['npc'];s=counts[own];profile=w.profiles[own];kind=row['kind'];flow=w.perceived(kind,row['other_kind'],row['matched'],1.)
                    if row['luck_failure']:flow['values']={}
                    s['strongest_flow']+=flow['values'].get(s['strongest_axis'],0.)
                    s['weighted_value_flow']+=sum(profile['values'].get(k,0.)*v for k,v in flow['values'].items());s['cost']+=flow['cost']
                    if kind in ('cooperate','claim','alone','rest'):s[kind]+=1
            worlds.append(dict(domain=r['domain'],layout=r['layout'],seed=r['seed'],variant=r['variant'],arm=r['peer_arm'],stats=counts,
                actors=r['actors'],goals={x['npc']:x['goal']['value'] for x in r['actors']},predictive_gain=statistics.mean(predictive) if predictive else None))
    groups={}
    for r in worlds:groups.setdefault(digest([r['domain'],r['layout'],r['seed'],r['variant']]),{})[r['arm']]=r
    comparisons=[]
    for ref,cand in [('neutral_level','neutral_gain'),('public_level','public_gain'),('neutral_level','public_level'),('neutral_gain','public_gain')]:
        pairs=[]
        for arms in groups.values():
            if ref not in arms or cand not in arms:continue
            x,y=arms[ref],arms[cand]
            for own,a in x['stats'].items():
                b=y['stats'][own];pairs.append(dict(domain=x['domain'],layout=x['layout'],seed=x['seed'],variant=x['variant'],npc=own,
                    delta=y['goals'][own]-x['goals'][own],before=x['goals'][own],after=y['goals'][own],strongest_flow_delta=b['strongest_flow']-a['strongest_flow'],
                    weighted_value_flow_delta=b['weighted_value_flow']-a['weighted_value_flow'],actions_changed=any(a[k]!=b[k] for k in ('cooperate','claim','alone','rest')),
                    before_counts=a,after_counts=b))
        comparisons.append(dict(reference=ref,candidate=cand,total=summary_stats(pairs),
            by_variant={v:summary_stats([p for p in pairs if p['variant']==v]) for v in ('known','mixed')},paired=pairs))
    if file.exists():assert len(worlds)==e['worlds'] and indices|{r['index'] for r in e['errors']}==set(range(len(pre['jobs'])))
    return dict(name=root.name,complete=file.exists(),worlds=len(worlds),npc_episodes=sum(len(w['stats']) for w in worlds),comparisons=comparisons,worlds_summary=worlds,errors=e['errors']),dict(
        name=root.name,preregister=pre,files_sha256={n:sha(root/n) for n in ('preregister.json','trajectories.jsonl','evaluation.json') if (root/n).exists()})
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohorts',nargs='+',required=True);ap.add_argument('--social');ap.add_argument('--output',required=True);ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);cohorts=[];provenance=[];comp=[];diversity=[];errors=[]
    for name in a.cohorts:
        rows,receipt=load(Path(name),a.partial);provenance.append(receipt);errors.extend(receipt['errors'])
        cohorts.append({k:receipt[k] for k in ('name','complete','planned','completed')})
        for ref,cand in [('level','gain'),('level','opportunity'),('level','gain_opportunity'),('gain','gain_opportunity'),('opportunity','gain_opportunity')]:
            pairs=comparison(rows,ref,cand)
            if not pairs:continue
            comp.append(dict(cohort=receipt['name'],reference=ref,candidate=cand,total=summary_stats(pairs),
                by_game={g:summary_stats([p for p in pairs if p['case']['game']==g]) for g in sorted({p['case']['game'] for p in pairs})},
                by_persona={pid:summary_stats([p for p in pairs if p['profile']==pid]) for pid in sorted({p['profile'] for p in pairs})},paired=pairs))
        for game in sorted({r['case']['game'] for r in rows}):
            for arm in sorted({r['arm'] for r in rows}):
                rs=[r for r in rows if r['case']['game']==game and r['arm']==arm]
                diversity.append(dict(cohort=receipt['name'],game=game,arm=arm,purpose_tradeoffs=sum(r['purpose_tradeoffs'] for r in rs),
                    steps=sum(r['steps'] for r in rs),unique_action_streams=len({digest(r['actions']) for r in rs}),
                    per_actual_seed={str(seed):len({digest(r['actions']) for r in rs if r['seed']==seed}) for seed in sorted({r['seed'] for r in rs})}))
    social=None
    if a.social:social,p=analyze_social(Path(a.social),a.partial);provenance.append(p);errors.extend(social['errors'])
    summary=dict(partial=a.partial,cohorts=cohorts,comparisons=comp,diversity=diversity,social=social,errors=errors,
        limitations='paired conditions and shared-world NPCs dependent; authored value flows are not whole utility; predictive accuracy is not causal decision benefit')
    for name,value in [('summary.json',summary),('provenance.json',provenance)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cohorts=cohorts,comparisons=[dict(cohort=c['cohort'],reference=c['reference'],candidate=c['candidate'],**c['total']) for c in comp],
        social=None if social is None else [dict(reference=c['reference'],candidate=c['candidate'],**c['total']) for c in social['comparisons']],errors=errors),ensure_ascii=False))
if __name__=='__main__':main()
