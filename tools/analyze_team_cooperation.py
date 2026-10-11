"""Independent transition ledgers and paired, seed-cluster team comparisons."""
import argparse
from collections import Counter,defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def seal(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(x):return hashlib.sha256(seal(x).encode()).hexdigest()


def check_projects(before,choices,after,audit):
    expected=copy.deepcopy(before);ps=expected['people'];stock=expected['stock'];points=expected['points']
    aid=[0]*len(ps);failed=[];overflow=0
    for team in (0,1):
        ids=[i for i,p in enumerate(ps) if p['team']==team]
        claims=[i for i in ids if choices[i] in ('build','train')]
        fail=len(claims)*2>before['stock'][team]
        if fail:failed+=claims
        for i in ids:
            p=ps[i];key=choices[i];e=p['energy']
            assert 0<=e<=6
            if key=='rest':p['energy']=min(6,e+3)
            elif key=='gather':
                assert e>=1;p['energy']=e-1;p['prestige']+=.5;stock[team]+=1 if before['scarce'] else 2
            elif key in ('build','train'):
                assert e>=2 and before['stock'][team]>=2
                if fail:continue
                stock[team]-=2;p['energy']=e-2
                if key=='build':gain=3+p['skill'];points[team]+=gain;p['prestige']+=gain
                else:assert p['skill']<2;p['skill']+=1
            else:
                assert key.startswith('aid:') and e>=1
                j=int(key.split(':')[1]);assert j!=i and ps[j]['team']==team and before['people'][j]['energy']<=4
                p['energy']=e-1;aid[j]+=2
        overflow+=max(0,stock[team]-8);stock[team]=min(8,stock[team])
    for i,p in enumerate(ps):p['energy']=min(6,p['energy']+aid[i])
    expected['tick']+=1
    if before['shock'] and expected['tick']==5:expected['stock']=[x//2 for x in stock]
    assert seal(expected)==seal(after)
    assert audit==dict(material_conflicts=failed,aid=aid,overflow=overflow)


def line_clear(w,a,b):
    for x,y in w['walls']:
        lo,hi=0.,1.
        for start,finish,c in ((a[0],b[0],x),(a[1],b[1],y)):
            if finish==start:
                if c-.5<=start<=c+.5:continue
                lo,hi=1.,0.;break
            v=sorted(((c-.5-start)/(finish-start),(c+.5-start)/(finish-start)))
            lo=max(lo,v[0]);hi=min(hi,v[1])
        if lo<=hi:return False
    return True


def shot_chance(w,ps,i,j):
    a,b=ps[i],ps[j];pa=(a['x'],a['y']);pb=(b['x'],b['y']);dist=abs(pa[0]-pb[0])+abs(pa[1]-pb[1])
    if min(a['hp'],b['hp'])<=0 or a['team']==b['team'] or a['ammo']<=0 or dist>4 or not line_clear(w,pa,pb):return 0.
    return max(.12,.9-.08*dist-.22*(list(pb) in w['covers'])-.2*b['guard'])


def check_combat(before,choices,after,audit,seed):
    expected=copy.deepcopy(before);ps=expected['units'];living=[i for i,p in enumerate(ps) if p['hp']>0]
    assert set(choices)==set(living);occupied={(p['x'],p['y']) for p in ps if p['hp']>0}
    destinations={i:tuple(map(int,k.split(':')[1:])) for i,k in choices.items() if k.startswith('move:')}
    collisions=0;heal=[0]*len(ps)
    for i,p in enumerate(ps):p['guard']=choices.get(i)=='guard'
    for i,d in destinations.items():
        p=before['units'][i]
        assert 0<=d[0]<9 and 0<=d[1]<5 and list(d) not in before['walls'] and d not in occupied
        assert abs(d[0]-p['x'])+abs(d[1]-p['y'])==1
        if list(destinations.values()).count(d)>1:collisions+=1
        else:ps[i]['x'],ps[i]['y']=d
    for i,k in choices.items():
        old=before['units'][i]
        if k=='reload':assert old['ammo']<3;ps[i]['ammo']=3
        elif k.startswith('heal:'):
            j=int(k.split(':')[1]);target=before['units'][j]
            assert old['kit'] and target['hp']>0 and target['hp']<9 and target['team']==old['team']
            assert abs(old['x']-target['x'])+abs(old['y']-target['y'])<=1
            ps[i]['kit']-=1;heal[j]+=4
        elif k.startswith('shoot:'):assert shot_chance(before,before['units'],i,int(k.split(':')[1]))>0
        else:assert k=='guard' or k.startswith('move:')
    for i,p in enumerate(ps):p['hp']=min(9,p['hp']+heal[i])
    phase=copy.deepcopy(ps);loss=[0]*len(ps);shots=[]
    for i,k in choices.items():
        if not k.startswith('shoot:'):continue
        j=int(k.split(':')[1]);p=shot_chance(before,phase,i,j)
        rng=np.random.default_rng(int(digest([seed,before['tick'],i,'actual-hit'])[:16],16))
        hit=bool(rng.random()<p);loss[j]+=3*hit;ps[i]['ammo']-=1
        shots.append(dict(shooter=i,target=j,probability=p,hit=hit))
    for i,p in enumerate(ps):p['hp']=max(0,p['hp']-loss[i])
    counts=[sum(p['hp']>0 and p['team']==t and p['x']==4 and 1<=p['y']<=3 for p in ps) for t in (0,1)]
    expected['hold']=[min(3,before['hold'][t]+1) if counts[t]>counts[1-t] and counts[t]>0 else 0 for t in (0,1)]
    expected['secured']=[bool(before['secured'][t] or expected['hold'][t]>=3) for t in (0,1)]
    expected['tick']+=1
    assert seal(expected)==seal(after)
    assert seal(audit)==seal(dict(shots=shots,collisions=collisions,healing=heal))


def summary(rows,modes):
    groups=[];pairs=[];index={}
    key=lambda r:(r['genre'],r['scenario'],r['roster'],r['seed'],r['participation'])
    for r in rows:index.setdefault(key(r),{})[r['mode']]=r
    for identity,arms in sorted(index.items()):
        assert set(arms)==set(modes)
        for mode in (m for m in modes if m!='independent'):
            a,b=arms['independent'],arms[mode]
            pairs.append(dict(identity=identity,mode=mode,credit_delta=b['win_credit']-a['win_credit'],
                health_delta=b['health']-a['health'],ticks_delta=b['ticks']-a['ticks']))
    for genre in ('combat','projects'):
        for participation in ('all','partial'):
            rs=[r for r in rows if r['genre']==genre and r['participation']==participation]
            if not rs:continue
            arms={}
            for mode in modes:
                group=[r for r in rs if r['mode']==mode];work=Counter()
                for r in group:work.update(r['work'])
                arms[mode]=dict(games=len(group),mean_credit=float(np.mean([r['win_credit'] for r in group])),
                    wins=sum(r['won'] for r in group),losses=sum(r['lost'] for r in group),
                    mean_health=float(np.mean([r['health'] for r in group])),work=dict(work))
            comparisons=[]
            contrasts=[(m,'independent') for m in modes if m!='independent']
            contrasts += [(c,r) for c,r in (('consent','sum'),('bargain','consent'),
                ('sum_uncertain','sum'),('bargain_uncertain','bargain'),('bargain_uncertain','sum_uncertain'))
                if c in modes and r in modes]
            for candidate,reference in contrasts:
                selected=[arms for identity,arms in index.items() if identity[0]==genre and identity[-1]==participation]
                ds=[(a[candidate]['seed'],a[candidate]['win_credit']-a[reference]['win_credit']) for a in selected]
                seeds=sorted({s for s,d in ds});means=np.array([np.mean([d for ss,d in ds if ss==s]) for s in seeds])
                rng=np.random.default_rng(9640701);boot=means[rng.integers(len(means),size=(10000,len(means)))].mean(1)
                comparisons.append(dict(candidate=candidate,reference=reference,paired_cases=len(ds),seed_clusters=len(seeds),
                    mean_credit_delta=float(means.mean()),seed_cluster_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                    better=sum(d>0 for s,d in ds),same=sum(d==0 for s,d in ds),worse=sum(d<0 for s,d in ds)))
            groups.append(dict(genre=genre,participation=participation,arms=arms,comparisons=comparisons))
    return groups,pairs


def main(args):
    root=Path(args.root);assert (root/'completed.json').exists(),'incomplete study'
    sys.path.insert(0,str(root/'_source'))
    from reflex.core import Policy,TRAITS,VALUES
    from reflex.laboratory import PROFILES
    from reflex.team_choice import select,concession
    from reflex import team_projects as tp,team_combat as tc
    from reflex.combat import battle_from_record
    from tools.run_team_cooperation import ROSTERS
    hashes=json.loads((root/'source_snapshot.json').read_text())
    for name,h in hashes.items():assert hashlib.sha256((root/'_source'/name).read_bytes()).hexdigest()==h
    rows=[];checks=Counter();by_profile=defaultdict(Counter);examples=[]
    with (root/'trajectories.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);prior=row['initial'];memories={};team=row['team']
            own=[i for i,p in enumerate(prior['units'] if row['genre']=='combat' else prior['people']) if p['team']==team]
            for turn in row['trace']:
                assert seal(turn['before'])==seal(prior);choices={int(k):v for k,v in turn['choices'].items()}
                if row['genre']=='combat':check_combat(prior,choices,turn['after'],turn['world_audit'],int(digest(['team-cooperation-world',row['seed']])[:16],16))
                else:check_projects(prior,choices,turn['after'],turn['world_audit'])
                checks['independent_world_transitions']+=1;checks['actual_actions']+=len(choices)
                for i,c,d in zip(turn['signed'],turn['contexts'],turn['decisions']):
                    p=PROFILES[ROSTERS[row['roster']][own.index(i)%3]]
                    assert c['personality']==dict(zip(TRAITS,p['traits'])) and c['values']=={k:float(p['values'].get(k,0)) for k in VALUES}
                    assert c['scope']['npc'] in (f'unit-{i}',f'worker-{i}') and c['tick']==prior['tick']
                    assert d['action_id']==choices[i] and d['context_hash']==digest(c)
                    if i in memories:assert c['state']==memories[i]
                    memories[i]=d['next_state'];checks['owner_receipts']+=1
                    by_profile[(row['genre'],row['mode'],p['id'])]['decisions']+=1
                    by_profile[(row['genre'],row['mode'],p['id'])][d['action_id'].split(':')[0]]+=1
                n=turn['negotiation']
                if n and n['cooperation_mode']=='consent' and n['adopted']:
                    assert set(n['members'])=={c['scope']['npc'] for c in turn['contexts']}
                    for member,regret in n['member_regrets'].items():assert -1e-12<=regret<=n['concessions'][member]+1e-12
                    checks['accepted_consent_receipts']+=1
                if args.replay_policies:
                    ds=[Policy(principle_priority='finite').choose(c,False) for c in turn['contexts']]
                    assert ds==turn['independent']
                    if turn['contexts'] and row['mode']!='independent':
                        kw={'partner_model':'uncertain'} if row['mode'].endswith('_uncertain') else {}
                        if row['genre']=='combat':w=battle_from_record(prior);forecast=tc.forecast(w,tuple(turn['signed']),turn['contexts'],ds,**kw)
                        else:
                            w=tp.Workshop(**{**prior,'people':tuple(tp.Worker(**p) for p in prior['people']),
                                'stock':tuple(prior['stock']),'points':tuple(prior['points'])})
                            forecast=tp.forecast(w,tuple(turn['signed']),turn['contexts'],ds,**kw)
                        skw={'outside':tuple(d['action_id'] for d in ds)} if row['mode'].startswith('bargain') else {}
                        selection='sum' if row['mode'].startswith('sum') else 'consent'
                        selected,n=select(turn['contexts'],forecast,group=f'{row["genre"]}-team-{team}',mode=selection,**skw)
                        assert seal(n)==seal(turn['negotiation'])
                        assert (ds if selected is None else selected)==turn['decisions']
                    else:assert ds==turn['decisions']
                    checks['frozen_policy_tick_replays']+=1
                prior=turn['after']
            assert seal(prior)==seal(row['final']);assert row['ticks']==prior['tick']
            if row['genre']=='projects':
                ws=[t for t,x in enumerate(prior['points']) if x>=prior['quota']]
                value=1/len(ws) if team in ws else 0. if ws else .5
            else:
                ps=prior['units'];ws=[]
                for t in (0,1):
                    living=any(p['hp']>0 and p['team']==t for p in ps)
                    kill=not any(p['hp']>0 and p['team']==1-t for p in ps);capture=prior['secured'][t]
                    if living and {'eliminate':kill,'secure':capture,'either':kill or capture,'both':kill and capture}[prior['goal']]:ws.append(t)
                value=1/len(ws) if team in ws else 0. if ws else .5
            assert row['win_credit']==value and row['won']==(team in ws) and row['lost']==(1-team in ws)
            rows.append(row)
    modes=json.loads((root/'preregister.json').read_text())['modes']
    groups,pairs=summary(rows,modes)
    result=dict(games=len(rows),groups=groups,pairs=pairs,checks=dict(checks),
        action_profiles=[dict(genre=k[0],mode=k[1],profile=k[2],actions=dict(v)) for k,v in sorted(by_profile.items())],
        source_snapshot_sha256=hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest(),
        trajectories_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest(),
        limits=['intervals condition on frozen authored scenarios; eight seeds are not eight unknown games',
                'modeled concessions are not measured real-life psychology or a promise of personal survival',
                'frozen numerical policy replay is separate from independently implemented world ledgers'])
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(games=result['games'],checks=result['checks'],groups=groups),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--replay-policies',action='store_true')
    main(p.parse_args())
