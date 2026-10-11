"""Audit complete cohorts and replay decisions/events; do not cherry-pick wins."""
import argparse,hashlib,json,sys
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.commitment_experiment import pledge_episode
from reflex.contingent_games import make_bridge
from reflex.observed_transfer import make_probe

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def analyze(folder,out,pledge_folder):
    pre=read(folder/'preregister.json');evaluation=read(folder/'evaluation.json')
    assert digest(pre)==evaluation['preregister_digest'] and not evaluation['errors']
    assert evaluation['episodes']==evaluation['planned']==len(pre['jobs'])==288
    for name,h in pre['source_hashes'].items():assert sha(folder/'source'/name)==h
    # A fixture edit or added analyzer may differ; numerical modules may not.
    for name,h in pre['source_hashes'].items():
        if name.startswith('reflex/') and name!='reflex/commitment_experiment.py':assert sha(ROOT/name)==h,name
    starts=read(folder/'starts.json');assert len(starts)==360
    assert len({r['case']['game'] for r in starts})==30
    assert all(Counter((r['profile'],r['rigidity']) for r in starts if r['case']['game']==game)==
        Counter((p,r) for p in ('growth','steady','care','ego') for r in (0.,.5,1.)) for game in {r['case']['game'] for r in starts})
    runs={};seen=set();summaries=[];steps=0;replays=Counter()
    for line in (folder/'trajectories.jsonl').open(encoding='utf-8'):
        r=json.loads(line);i=r['job_index'];assert i not in seen;seen.add(i)
        case,p,seed,rigidity=pre['jobs'][i];assert (r['case'],r['profile'],r['seed'],r['rigidity'])==(case,p['id'],seed,rigidity)
        assert r['steps']==len(r['trace']);steps+=r['steps']
        if case['game']=='mahjong_closed':
            from reflex.mahjong_adapter import engine,action_id
            from reflex.mahjong_experiment import audit
            env=engine().RiichiEnv(game_mode='4p-red-single',seed=seed);obs=env.reset()
            for t in r['trace']:
                actions={}
                for d in t['decisions']:
                    a=next(a for a in obs[d['player']].legal_actions() if action_id(a)==d['action'])
                    actions[d['player']]=a
                start=len(env.mjai_log);obs=env.step(actions);audit(env)
                assert digest(env.mjai_log[start:])==digest(t['events'])
            assert env.done() and digest(env.scores())==digest(r['scores'])
            replays['mahjong_engine']+=1;value=r['point_delta']
        elif case['game'].startswith('observed_'):
            b=make_probe(case['spec'],p);w=b.start()
            for t in r['trace']:
                assert digest(asdict(w))==digest(t['before']);w,flow=b.actual(w,t['root'],seed)
                assert digest(asdict(w))==digest(t['after']),(case['game'],i,t['root'],'after')
                assert digest(flow)==digest(t['flow']),(case['game'],i,t['root'],'flow')
                assert b.goal(w).record()==t['goal'],(case['game'],i,t['root'],'goal')
            assert b.terminal(w);replays['observed']+=1;value=r['goal']['value']
        else:
            b=make_bridge(p,case);w=b.start()
            for t in r['trace']:
                assert digest(b.record(w))==digest(t['before']),(case['game'],i)
                w,flow=b.actual(w,t['root'],seed)
                assert digest(b.record(w))==digest(t['after']) and digest(flow)==digest(t['flow']) and b.goal(w).record()==t['goal']
            assert b.terminal(w) and r['action_replay_verified'];replays['generic']+=1;value=r['goal']['value']
        summary={k:v for k,v in r.items() if k!='trace'};summaries.append(summary)
        key=(digest(case),p['id'],seed,rigidity);assert key not in runs
        actions=[t['root'] for t in r['trace']] if case['game']!='mahjong_closed' else [d['action'] for t in r['trace'] for d in t['decisions'] if 'public' in d]
        runs[key]=dict(game=case['game'],profile=p['id'],seed=seed,rigidity=rigidity,value=value,actions=actions,steps=r['steps'])
    assert len(seen)==288 and sorted(summaries,key=lambda r:r['job_index'])==evaluation['runs']
    paired=[];counts=defaultdict(Counter);by_game=defaultdict(lambda:defaultdict(Counter))
    for (case,p,seed,rigidity),base in sorted(runs.items()):
        if rigidity!=0.:continue
        for level in (.5,1.):
            other=runs[(case,p,seed,level)];delta=other['value']-base['value']
            direction='better' if delta>1e-12 else 'worse' if delta<-1e-12 else 'same'
            counts[str(level)][direction]+=1;by_game[base['game']][str(level)][direction]+=1
            paired.append(dict(game=base['game'],profile=p,seed=seed,rigidity=level,delta=delta,
                baseline=base['value'],value=other['value'],actions_changed=base['actions']!=other['actions']))
    pledges=[];groups=defaultdict(list);pledge_keys=set()
    pledge_pre=read(pledge_folder/'preregister.json')
    assert len(pledge_pre['jobs'])==768 and pledge_pre['actor_random_seed']==42
    for name,h in pledge_pre['source_hashes'].items():assert sha(pledge_folder/'source'/name)==h
    for name,h in pledge_pre['source_hashes'].items():
        if name.startswith('reflex/'):assert sha(ROOT/name)==h,name
    for line in (pledge_folder/'pledge.jsonl').open(encoding='utf-8'):
        r=json.loads(line);key=(r['changed'],r['seed'],r['rigidity'],r['inertia']);assert key not in pledge_keys;pledge_keys.add(key)
        assert pledge_episode(r['seed'],r['rigidity'],r['inertia'],changed=r['changed'])==r
        certain=0;before=1.
        for t in r['trace']:
            certain+=int(before<=.015 and t['action']!='leave');before=t['health']
            assert t['owner']==r['trace'][0]['owner']
            if t['feedback'] is not None:assert t['feedback']['before']['observations']+1==t['feedback']['after']['observations']
        summary={k:v for k,v in r.items() if k!='trace'};summary['foreseen_certain_death_choices']=certain
        pledges.append(summary);groups[(r['changed'],r['rigidity'],r['inertia'])].append(summary)
    assert len(pledges)==768 and all(len(g)==64 for g in groups.values()) and len(groups)==12
    pledge_summary=[]
    for key,g in sorted(groups.items()):
        row=dict(changed=key[0],rigidity=key[1],inertia=key[2],episodes=len(g),deaths=sum(r['died'] for r in g),
            certain_death_choices=sum(r['foreseen_certain_death_choices'] for r in g))
        for f in ('protected','leave','old_after_change','other_after_change'):row['mean_'+f]=sum(r[f] for r in g)/len(g)
        pledge_summary.append(row)
    out.mkdir(parents=True,exist_ok=True)
    for name in ('preregister.json','evaluation.json','starts.json'):(out/name).write_bytes((folder/name).read_bytes())
    (out/'pledge_preregister.json').write_bytes((pledge_folder/'preregister.json').read_bytes())
    result=dict(format='personal-commitment-audit-v1',whole_episodes=288,whole_protocols=12,connection_starts=360,start_protocols=30,
        paired_contrasts=192,counts={k:dict(v) for k,v in counts.items()},by_game={g:{k:dict(v) for k,v in levels.items()} for g,levels in by_game.items()},
        comparisons=paired,pledge_episodes=768,pledge_summary=pledge_summary,replays=dict(replays),trace_intervals=steps,
        default_changed=False,cloud_runtime_used=False)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'pledge_summary.json').write_text(json.dumps(pledges,indent=2)+'\n',encoding='utf-8')
    prov=dict(timing='post-study descriptive audit; not statistical significance',source_hashes=pre['source_hashes'],
        raw_sha256={'trajectories.jsonl':sha(folder/'trajectories.jsonl'),'pledge.jsonl':sha(pledge_folder/'pledge.jsonl')},
        pledge_source_hashes=pledge_pre['source_hashes'],
        analyzer_sha256=sha(Path(__file__)),source_differences=[n for n,h in pre['source_hashes'].items() if not (ROOT/n).exists() or sha(ROOT/n)!=h],
        archived_invalid_subjective_v1='v1 retained privately; prior concentration contaminated evidence confidence; entire cohort replaced, no double counting',
        replaced_pledge_v2='world/actor RNG used same seed and diagnostic mode was not initialized; all768 replaced by isolated pledge-v3; unaffected288 games retained')
    (out/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('whole_episodes','start_protocols','counts','pledge_summary','replays','trace_intervals')},ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--pledge-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    analyze(Path(a.root).resolve(),Path(a.output).resolve(),Path(a.pledge_root).resolve())
