"""Retain paired regressions, personality differences and predictive failures."""
import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest


def comparison(rows,reference,candidate):
    groups={}
    for r in rows:groups.setdefault(digest([r['case'],r['profile'],r['seed']]),{})[r['arm']]=r
    pairs=[]
    for g in groups.values():
        if reference not in g or candidate not in g:continue
        a,b=g[reference],g[candidate]
        pairs.append(dict(case=a['case'],profile=a['profile'],seed=a['seed'],reference=reference,candidate=candidate,
            before=a['goal']['value'],after=b['goal']['value'],delta=b['goal']['value']-a['goal']['value'],
            actions_changed=a['actions']!=b['actions'],forecast_loss_delta=b['forecast_loss']-a['forecast_loss'] if a['budget']['horizon']==b['budget']['horizon'] else None,
            strongest_flow_delta=b['actual_preference_flow']['strongest_flow']-a['actual_preference_flow']['strongest_flow'],
            weighted_value_flow_delta=b['actual_preference_flow']['weighted_value_flow']-a['actual_preference_flow']['weighted_value_flow'],
            actual_cost_delta=b['actual_preference_flow']['cost']-a['actual_preference_flow']['cost']))
    return pairs


def stats(pairs):
    ds=[p['delta'] for p in pairs]
    return dict(pairs=len(ds),better=sum(d>1e-12 for d in ds),worse=sum(d<-1e-12 for d in ds),same=sum(abs(d)<=1e-12 for d in ds),
        mean_delta=statistics.mean(ds) if ds else None,actions_changed=sum(p['actions_changed'] for p in pairs),
        lower_goal_higher_strongest_flow=sum(p['delta']<-1e-12 and p['strongest_flow_delta']>1e-12 for p in pairs),
        preference_semantics='authored realized value flows, not whole Policy utility or proof that a goal regression was intentional')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cohorts',nargs='+',required=True);ap.add_argument('--social',nargs='*',default=[])
    ap.add_argument('--output',required=True);ap.add_argument('--partial',action='store_true');a=ap.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    cohorts=[];rows=[];provenance=[];comparisons=[];errors=[]
    contrasts=[('short',k) for k in ('long','more_samples','resampled','feedback','checked','adaptive')]+[('long','more_samples'),('long','resampled'),('resampled','feedback'),('feedback','checked'),('checked','adaptive'),('checked','challenged')]
    for value in a.cohorts:
        p=Path(value);pre=json.loads((p/'preregister.json').read_text(encoding='utf-8'));evaluated=p/'evaluation.json'
        if evaluated.exists():
            r=json.loads(evaluated.read_text(encoding='utf-8'))
            if r['episodes']+len(r['errors'])!=r['planned'] or r['planned']!=len(pre['jobs']):raise ValueError('whole cohort count mismatch')
            runs=r['runs'];errors.extend(r['errors'])
        elif a.partial:
            runs=[]
            for line in (p/'trajectories.jsonl').read_text(encoding='utf-8').splitlines():
                if not line:continue
                record=json.loads(line);scores=[s for t in record['trace'] for s in t['forecast_scores']]
                record['forecast_loss']=statistics.mean(s['forecast_loss'] for s in scores);record['persistence_loss']=statistics.mean(s['persistence_loss'] for s in scores)
                record['exact_action_branch_matches']=sum(s['matching_action_branches'] for s in scores);record['scored_branches']=sum(s['branches'] for s in scores)
                record.pop('trace');runs.append(record)
        else:raise ValueError('incomplete cohort cannot be final evidence')
        preference={}
        # Inspect actual physical flows, not only static trait preservation or
        # victory. Streaming keeps the large frozen trajectories out of RAM.
        with (p/'trajectories.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():continue
                raw=json.loads(line);profile=pre['jobs'][raw['index']][1];values=profile['values']
                top=max(values,key=values.get) if values else None
                preference[raw['index']]=dict(strongest_axis=top,
                    strongest_flow=sum(t['flow']['values'].get(top,0.) for t in raw['trace']) if top else 0.,
                    weighted_value_flow=sum(sum(values.get(k,0.)*v for k,v in t['flow']['values'].items()) for t in raw['trace']),
                    cost=sum(t['flow']['cost'] for t in raw['trace']))
        for row in runs:row['actual_preference_flow']=preference[row['index']]
        cohorts.append(dict(name=p.name,planned=len(pre['jobs']),completed=len(runs),complete=evaluated.exists()))
        rows.extend(runs);provenance.append(dict(name=p.name,preregister=pre,preregister_sha256=digest(pre)))
        for ref,cand in contrasts:
            pairs=comparison(runs,ref,cand)
            if not pairs:continue
            comparisons.append(dict(cohort=p.name,reference=ref,candidate=cand,total=stats(pairs),
                by_game={game:stats([q for q in pairs if q['case']['game']==game]) for game in sorted({q['case']['game'] for q in pairs})},
                by_persona={pid:stats([q for q in pairs if q['profile']==pid]) for pid in sorted({q['profile'] for q in pairs})},paired=pairs))
    prediction=[]
    for game in sorted({r['case']['game'] for r in rows}):
        for arm,horizon in sorted({(r['arm'],r['budget']['horizon']) for r in rows}):
            group=[r for r in rows if r['case']['game']==game and r['arm']==arm and r['budget']['horizon']==horizon]
            if group:prediction.append(dict(game=game,arm=arm,episodes=len(group),
                horizon=horizon,target_semantics='game goal at declared owner-step horizon on each controller own actual path; proxy is not win probability',
                forecast_loss=statistics.mean(r['forecast_loss'] for r in group),persistence_loss=statistics.mean(r['persistence_loss'] for r in group),
                matching_action_branches=sum(r['exact_action_branch_matches'] for r in group),scored_branches=sum(r['scored_branches'] for r in group),
                matching_semantics='whole sampled action-prefix coincidence; differences may reflect world randomness, NOT an executor mismatch rate'))
    diversity=[]
    for game in sorted({r['case']['game'] for r in rows}):
        for arm in sorted({r['arm'] for r in rows}):
            group=[r for r in rows if r['case']['game']==game and r['arm']==arm]
            if group:diversity.append(dict(game=game,arm=arm,episodes=len(group),unique_action_streams=len({digest([r['case'],r['actions']]) for r in group}),
                persona_results={pid:dict(episodes=len(rs),mean_goal=statistics.mean(x['goal']['value'] for x in rs))
                    for pid in sorted({r['profile'] for r in group}) if (rs:=[r for r in group if r['profile']==pid])}))
    social=[]
    for value in a.social:
        p=Path(value);r=json.loads((p/'evaluation.json').read_text(encoding='utf-8'));pre=json.loads((p/'preregister.json').read_text(encoding='utf-8'))
        if r['worlds']+len(r['errors'])!=r['planned']:raise ValueError('incomplete social cohort')
        errors.extend(r['errors']);pairs=r['paired'];deltas=[q['delta'] for q in pairs]
        social.append(dict(name=p.name,worlds=r['worlds'],npc_episodes=sum(len(x['actors']) for x in r['runs']),paired_npcs=len(pairs),
            better=sum(d>1e-12 for d in deltas),worse=sum(d<-1e-12 for d in deltas),same=sum(abs(d)<=1e-12 for d in deltas),
            mean_delta=statistics.mean(deltas),by_variant={v:dict(pairs=len(ds),better=sum(d>1e-12 for d in ds),worse=sum(d<-1e-12 for d in ds),same=sum(abs(d)<=1e-12 for d in ds),mean_delta=statistics.mean(ds))
                for v in sorted({q['variant'] for q in pairs}) if (ds:=[q['delta'] for q in pairs if q['variant']==v])},paired=pairs,runs=r['runs']))
        provenance.append(dict(name=p.name,preregister=pre,preregister_sha256=digest(pre)))
    result=dict(partial=a.partial,cohorts=cohorts,comparisons=comparisons,prediction=prediction,diversity=diversity,social=social,errors=errors,
        limitations='paired seeds/profile cases and shared worlds are dependent; predictive accuracy is not decision value or causal learning')
    for name,value in (('summary.json',result),('provenance.json',provenance)):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cohorts=cohorts,contrasts=[dict(cohort=c['cohort'],reference=c['reference'],candidate=c['candidate'],**c['total']) for c in comparisons],social=social[:0],errors=errors),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
