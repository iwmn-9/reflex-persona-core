"""Replay consent and real transitions; summarize losses as well as successes."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
from reflex.core import digest
from reflex.encounters import EncounterWorld
from reflex.cooperation import Ballot,negotiate
from tools.cooperation_experiment import catalog,configure


def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),(a.keys(),b.keys())
        for k in a:close(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
    else:assert a==b,(a,b)


def audit(root):
    root=Path(root);data=json.loads((root/'results.json').read_text());counts=Counter();groups={};profile=defaultdict(Counter);blocks=defaultdict(Counter)
    for r in data['runs']:
        raw=gzip.decompress((root/r['trajectory']).read_bytes());assert hashlib.sha256(raw).hexdigest()==r['sha256']
        saved=json.loads(raw);trace=saved['trace'];assert len(trace)==r['turns']
        w=EncounterWorld(r['seed'],r['domain'],r['layout'],r['contact']);configure(w,r['scenario'])
        if r['scenario']=='asymmetric':
            for i,own in enumerate(w.actors):w.health[own]=3. if i%2==0 else 8.;w.energy[own]=.4 if i%2==0 else .9
        seen=set()
        for tick,t in enumerate(trace):
            assert t['tick']==w.tick==tick;configure(w,r['scenario']);cs,_=w.contexts()
            offers,_=catalog(w) if r['arm'] in ('communicate','combined','continued') else ([],{})
            scope=(w.domain,cs[0]['scope']['episode'],tick,digest([w.rules,[w.state(a) for a in w.actors]]))
            ballots=[Ballot(b['owner'],scope,b['outside'],tuple(b['ranked'])) for b in t['ballots']]
            legal={c['scope']['npc']:{a['id'] for a in c['actions'] if a['legal']} for c in cs}
            continuing=tuple(t['negotiation'].get('continuing',()))
            if r['arm']=='continued' and tick:
                previous=trace[tick-1];previous_choices={r['npc']:r['action'] for r in previous['rows']}
                expected=tuple(a['offer'] for a in previous['negotiation']['agreements'] if all(previous_choices[m]=='rest' for m in a['members']))
                assert continuing==expected
            else:assert not continuing
            selected,receipt=negotiate(offers,ballots,t['negotiation']['order'],scope=scope,legal=legal,continuing=continuing)
            close(receipt,t['negotiation'])
            assert selected=={row['npc']:row['action'] for row in t['rows']}
            for b in ballots:
                e=t['evaluations'][b.owner]
                assert tuple(e['accepted'])==b.ranked and e['outside_root']==b.outside
                for key in b.ranked:assert e['scores'][key]>e['scores'][e['outside_option']]+1e-9
            signed=set()
            for agreement in receipt['agreements']:
                for own in agreement['members']:
                    assert own not in signed;signed.add(own)
                    assert agreement['offer'] in next(b.ranked for b in ballots if b.owner==own)
                counts['agreements']+=1
            rows,_=w.resolve(selected)
            for actual,stored in zip(rows,t['rows']):
                close(actual,{k:stored[k] for k in actual})
                # Independent arithmetic check, not just the production resolver.
                old=stored['before'];after=stored['after'];kind=stored['kind']
                if kind!='finished':
                    energy=(min(1,old['energy']+.4) if kind=='rest' else max(0,old['energy']-.18))-.025
                    assert math.isclose(after['energy'],max(0,energy),abs_tol=1e-10)
                    assert math.isclose(after['stock'],max(0,old['stock']+stored['gain']-.08),abs_tol=1e-10)
                    counts['active_decisions']+=1
                counts['actor_ticks']+=1
                key=(r['domain'],r['scenario'],r['arm'],stored['profile'])
                profile[key][kind]+=1
                block=(r['domain'],r['scenario'],r['arm'],'before160' if tick<160 else 'after160')
                blocks[block][kind]+=1
                if stored['matched'] and stored['kind']==stored['other_kind']=='cooperate':
                    seen.add((stored['npc'],stored['target']))
            counts['world_ticks']+=1
        for a in r['actors']:close(w.state(a['npc']),{k:a[k] for k in w.state(a['npc'])})
        key='|'.join((r['domain'],r['scenario'],r['arm']))
        group=groups.setdefault(key,dict(worlds=0,actors=0,alive=0,gain=0.,agreements=0,cooperation=0,rejections=0))
        group['worlds']+=1;group['actors']+=len(r['actors']);group['alive']+=r['alive'];group['gain']+=r['gain']
        for k in ('agreements','cooperation','rejections'):group[k]+=r[k]
        counts['worlds']+=1
    lookup={(r['seed'],r['domain'],r['layout'],r['contact'],r['scenario'],r['arm']):r for r in data['runs']}
    paired={}
    for new,old in [('combined','foresight'),('communicate','independent'),('combined','independent'),('continued','combined')]:
        total=Counter();by=defaultdict(Counter)
        for k,r in lookup.items():
            if k[-1]!=new or k[:-1]+(old,) not in lookup:continue
            base=lookup[k[:-1]+(old,)];tag=k[1]+'|'+k[4]
            total['world_pairs']+=1;by[tag]['world_pairs']+=1
            for a,b in zip(r['actors'],base['actors']):
                assert (a['npc'],a['profile'])==(b['npc'],b['profile'])
                delta=a['score']-b['score'];out='gain_better' if delta>1e-9 else 'gain_worse' if delta< -1e-9 else 'gain_equal'
                for c in (total,by[tag]):
                    c[out]+=1;c['gain_delta']+=delta;c['avoided_deaths']+=int(b['health']<=0<a['health']);c['new_deaths']+=int(a['health']<=0<b['health'])
        paired[new+'_vs_'+old]=dict(total=dict(total),by_condition={k:dict(v) for k,v in by.items()})
    report=dict(verified=dict(counts),groups=groups,paired=paired,
        profile_actions={'|'.join(k):dict(v) for k,v in profile.items()},
        phase_actions={'|'.join(k):dict(v) for k,v in blocks.items()},
        notes=['all worlds replayed: physics, legality, consent, score floor, scope, exclusivity',
            'actor results within the same world are dependent',
            'private score comparison is model-relative, not proof of actual future utility'])
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified=report['verified'],paired=paired),indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();audit(a.root)
