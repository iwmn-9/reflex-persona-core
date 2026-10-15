"""Controlled study of voluntary coordination, with independent private ballots.

The adapter offers every pair of public encounter roles, including asymmetric
and mutually harmful ones. It does not prescribe partners or label winners.
The optional forecast is a conditional repeated method with synchronized
recovery, NOT a promise that future independent people will keep consenting.
Only the first action is executed; every subsequent tick requires fresh consent.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import itertools
import json
import time
import numpy as np
from reflex.core import Policy, compile_batch, digest, TRAITS, NEEDS
from reflex.cooperation import Offer, Ballot, evaluate, negotiate
from reflex.encounters import EncounterWorld, after_action, RULES, LAYOUTS
from reflex.examples import action, effect
from reflex.social import SocialBinding
from reflex.social_withdrawal import WithdrawalSocialMemory

ARMS = ('independent', 'communicate', 'foresight', 'combined', 'continued')
SCENARIOS = ('standard', 'solo_advantage', 'costly_joint', 'change', 'asymmetric')


def configure(world, scenario):
    # Public rule variations, declared before evaluation; common controller and
    # coefficients are identical across worlds. No hidden future shock is read.
    rules = dict(RULES[world.domain])
    if scenario == 'solo_advantage' or scenario == 'change' and world.tick >= 160:
        rules['alone'] = (.7, .35)
    if scenario == 'costly_joint':rules['cc'] = (.05, -.2)
    world.rules = rules


def catalog(world):
    offers=[]; methods={}
    for i, a in enumerate(world.actors):
        for b in world.actors[i+1:]:
            if min(world.health[a], world.health[b]) <= 0:continue
            if world.contact == 'public' and b != world.available_partner(a):continue
            for ka, kb in itertools.product(('cooperate', 'claim'), repeat=2):
                key=f'joint-{a}-{b}-{ka}-{kb}'
                recover=min(world.energy[a], world.energy[b]) < .18
                roots=tuple((own, 'rest' if recover else f'{kind}:{target}')
                    for own,kind,target in ((a,ka,b),(b,kb,a)))
                offers.append(Offer(key,roots));methods[key]={a:(ka,b,kb),b:(kb,a,ka)}
    return offers,methods


def immediate_options(world,own,c,offers,methods):
    """Exact public one-step probabilities, including the 5% environmental loss."""
    specs=[];roots={};outside=[];bindings={};before=world.state(own)
    for a in c['actions']:
        if not a['legal'] or a['known_failure']:continue
        key='outside-'+a['id'];kind,target=world.catalog[own][a['id']]
        specs.append((key,a['id'],kind,target,copy.deepcopy(a['outcomes'])))
        outside.append(key)
    for offer in offers:
        rr=dict(offer.roots)
        if own not in rr:continue
        kind,target,other=methods[offer.key][own]
        root=rr[own]
        if root=='rest':kind='rest';outcomes=[world.perceived('rest',None,False,1.)];target=None
        elif kind==other=='cooperate':
            outcomes=[world.perceived(kind,other,True,.95),effect(-.1,p=.05,needs={'safety':-.15,'physiology':-.12},cost=.05)]
        else:outcomes=[world.perceived(kind,other,True,1.)]
        specs.append((offer.key,root,kind,target,outcomes))
    out=[]
    safety=lambda s:max(1-s['health']/8,max(0,(2-s['stock'])/2))
    for key,root,kind,target,outcomes in specs:
        for flow in outcomes:
            after=after_action(before,kind,flow['objective'],flow['needs'].get('safety',0),world.rules['upkeep'])
            flow['needs'].update(physiology=after['energy']-before['energy'],safety=safety(before)-safety(after))
            if after['health']<=0:flow['objective']=-1.
        a=action(key,*outcomes);a['target']=target;out.append(a);roots[key]=root
        if target:bindings[key]=SocialBinding(target,1 if kind=='cooperate' else -1,max(0,1-world.health[target]/8))
    return out,roots,tuple(outside),bindings


def forecast_options(world, own, c, offers, methods, horizon, samples=8):
    """Vectorized conditional options; only public rules/state/history are read.

    Intrinsic style/value effects are time averages; physiology/safety are net
    state changes; objective is average earned resources if alive at the horizon,
    and -1 on death. This game explicitly values survival, unlike the old gain-only
    objective. Both horizons and both communication arms use this same contract.
    """
    if horizon==1:return immediate_options(world,own,c,offers,methods)
    legal=[a for a in c['actions'] if a['legal'] and not a['known_failure']]
    specs=[]; roots={}; outside=[]
    for a in legal:
        key='outside-'+a['id'];kind,target=world.catalog[own][a['id']]
        specs.append((key,kind,target,None,False,a['id']))
        roots[key]=a['id'];outside.append(key)
    for offer in offers:
        if own not in dict(offer.roots):continue
        kind,target,other=methods[offer.key][own]
        specs.append((offer.key,kind,target,other,True,dict(offer.roots)[own]))
        roots[offer.key]=dict(offer.roots)[own]
    n=len(specs);shape=(n,samples)
    energy=np.full(shape,world.energy[own]);health=np.full(shape,world.health[own]);stock=np.full(shape,world.stock[own])
    other_energy=np.array([world.energy[t] if t else 1 for _,_,t,_,_,_ in specs])[:,None]*np.ones(shape)
    other_health=np.array([world.health[t] if t else 8 for _,_,t,_,_,_ in specs])[:,None]*np.ones(shape)
    gain_sum=np.zeros(shape);style=np.zeros((n,samples,5));values=np.zeros((n,samples,10));other_needs=np.zeros((n,samples,5));cost=np.zeros(shape)
    from reflex.core import VALUES
    # Common random numbers across candidates; distinct from real world RNG.
    rng=np.random.default_rng(np.random.SeedSequence([world.seed,world.tick,72131]))
    draws=rng.random((horizon,samples,2))
    probabilities=np.array([world.forecast(own,t) if t else (0,0,1) for _,_,t,_,_,_ in specs])
    kinds=np.array([k for _,k,_,_,_,_ in specs]);joint=np.array([j for *_,j,r in specs])
    responses=np.array([k or '' for _,_,_,k,_,_ in specs]);first=np.array([world.catalog[own][r][0] for *_,r in specs])
    public_clock=copy.copy(world)
    # Perceived fixed meanings are read from the existing game adapter.
    table={}
    for k in ('alone','rest','finished','cooperate','claim'):
        for response in ('cooperate','claim','unmatched','failure'):
            matched=response!='unmatched'
            flow=world.perceived(k,None if not matched else response,matched,1.)
            if response=='failure' and k=='cooperate':
                flow=effect(-.1,needs={'safety':-.15,'physiology':-.12},cost=.05)
            gh=(-.1,-.15) if response=='failure' and k=='cooperate' else world.consequences(k,response,matched)
            table[k,response]=(gh,np.array([flow['values'].get(v,0) for v in VALUES]),
                np.array([flow['style'].get(t,0) for t in TRAITS]),
                np.array([flow['needs'].get(t,0) for t in NEEDS]),flow['cost'])
    for step in range(horizon):
        alive=health>0
        # Each option is a method with recovery, not an instruction to rest forever.
        kk=np.broadcast_to(kinds[:,None],shape).copy()
        kk[kinds=='rest']='alone'
        resting=(energy<.18)|(joint[:,None]&(other_energy<.18))
        kk[resting]='rest'
        contact=np.ones((n,1),dtype=bool)
        if world.contact=='public':
            public_clock.tick=world.tick+step
            partner=public_clock.available_partner(own)
            contact=np.array([t is None or t==partner for _,_,t,_,_,_ in specs])[:,None]
            # A scheduled opportunity can close. During that interval each actor
            # uses legal independent work/recovery, not an impossible joint role.
            kk=np.where(joint[:,None]&~contact,np.where(energy<.18,'rest','alone'),kk)
        if step==0:kk=np.broadcast_to(first[:,None],shape).copy()
        kk[~alive]='finished'
        rr=np.where(draws[step,:,0][None,:]<probabilities[:,0,None],'cooperate',
            np.where(draws[step,:,0][None,:]<(probabilities[:,0]+probabilities[:,1])[:,None],'claim','unmatched'))
        joint_response=np.where((other_health>0)&(other_energy>=.18)&(kk!='rest'),responses[:,None],'unmatched')
        rr=np.where(joint[:,None],joint_response,rr)
        rr=np.where(contact,rr,'unmatched')
        failed=(kk=='cooperate')&(rr=='cooperate')&(draws[step,:,1][None,:]<.05)
        rr[failed]='failure'
        gain=np.zeros(shape);hp=np.zeros(shape)
        for (kind,response),(gh,v,st,ne,co) in table.items():
            mask=(kk==kind)&(rr==response)
            gain[mask]=gh[0];hp[mask]=gh[1];values[mask]+=v;style[mask]+=st;other_needs[mask]+=ne;cost[mask]+=co
        gain_sum+=gain
        energy=np.where(alive,np.maximum(0,np.where(kk=='rest',np.minimum(1,energy+.4),np.maximum(0,energy-.18))-.025),energy)
        stock=np.where(alive,np.maximum(0,stock+gain-.08),stock)
        health=np.where(alive,np.maximum(0,np.minimum(8,health+hp)-world.rules['upkeep']),health)
        # Conditional partner follows the declared method/recovery schedule.
        pk=np.where(kk=='rest','rest',np.where((kk=='finished')|(kk=='alone'),'alone',responses[:,None]))
        pk=np.where(joint[:,None]&~contact,np.where(other_energy<.18,'rest','alone'),pk)
        pr=np.where((kk=='cooperate')|(kk=='claim'),kk,'unmatched')
        pr=np.where(failed,'failure',pr);partner_hp=np.zeros(shape)
        for (kind,response),(gh,*_) in table.items():partner_hp[(pk==kind)&(pr==response)]=gh[1]
        partner_alive=joint[:,None]&(other_health>0)
        other_energy=np.where(partner_alive,np.maximum(0,np.where(pk=='rest',np.minimum(1,other_energy+.4),np.maximum(0,other_energy-.18))-.025),other_energy)
        other_health=np.where(partner_alive,np.maximum(0,np.minimum(8,other_health+partner_hp)-world.rules['upkeep']),other_health)
    before_safety=max(1-world.health[own]/8,max(0,(2-world.stock[own])/2))
    safety=np.maximum(1-health/8,np.maximum(0,(2-stock)/2))
    out=[];bindings={}
    for idx,(key,kind,target,other,is_joint,root) in enumerate(specs):
        outcomes=[]
        for s in range(samples):
            ne={k:float(other_needs[idx,s,j]/horizon) for j,k in enumerate(NEEDS)}
            ne.update(physiology=float(energy[idx,s]-world.energy[own]),safety=float(before_safety-safety[idx,s]))
            outcomes.append(effect(float(gain_sum[idx,s]/horizon) if health[idx,s]>0 else -1.,needs=ne,
                values=dict(zip(VALUES,(values[idx,s]/horizon).tolist())),style=dict(zip(TRAITS,(style[idx,s]/horizon).tolist())),
                cost=float(cost[idx,s]/horizon),p=1/samples))
        a=action(key,*outcomes);a['target']=target;out.append(a)
        if target:bindings[key]=SocialBinding(target,1 if kind=='cooperate' else -1,max(0,1-world.health[target]/8))
    return out,roots,tuple(outside),bindings


def run_one(job):
    root,seed,domain,layout,contact,scenario,arm,turns,horizon=job
    world=EncounterWorld(seed,domain,layout,contact);configure(world,scenario)
    if scenario=='asymmetric':
        for i,own in enumerate(world.actors):
            world.health[own]=3. if i%2==0 else 8.;world.energy[own]=.4 if i%2==0 else .9
    initial,_=world.contexts();mem={a:WithdrawalSocialMemory(c) for a,c in zip(world.actors,initial)}
    states={a:copy.deepcopy(c['state']) for a,c in zip(world.actors,initial)}
    fixed=digest([(c['personality'],c['values']) for c in initial]);trace=[];unfinished=()
    for tick in range(turns):
        configure(world,scenario);cs,_=world.contexts()
        offers,methods=catalog(world) if arm in ('communicate','combined','continued') else ([],{})
        revision=digest([world.rules,[world.state(a) for a in world.actors]])
        scope=(world.domain,initial[0]['scope']['episode'],tick,revision)
        ballots=[];audits={};legal={}
        for own,c in zip(world.actors,cs):
            c['state']=copy.deepcopy(states[own]);legal[own]={a['id'] for a in c['actions'] if a['legal']}
            opts,roots,outside,bindings=forecast_options(world,own,c,offers,methods,
                min(horizon,turns-tick) if arm in ('foresight','combined','continued') else 1)
            app,_=mem[own].scores([a['id'] for a in opts],[a['target'] for a in opts],bindings,tick)
            ranked,audit=evaluate(c,opts,roots,outside,appraisal=dict(zip([a['id'] for a in opts],app)))
            ballots.append(Ballot(own,scope,audit['outside_root'],ranked));audits[own]=audit
        order=world.actors[tick%len(world.actors):]+world.actors[:tick%len(world.actors)]
        chosen,negotiation=negotiate(offers,ballots,order,scope=scope,legal=legal,continuing=unfinished)
        rows,events=world.resolve(chosen)
        # A method that only performed its recovery step is unfinished. Once
        # its actual role actions run (even with an unlucky outcome), release it.
        # This adapter's recovery lasts one step; future consent remains required.
        unfinished=tuple(a['offer'] for a in negotiation['agreements']
            if all(chosen[m]=='rest' for m in a['members'])) if arm=='continued' else ()
        for own,row,ev in zip(world.actors,rows,events):
            for event in ev:mem[own].observe(event)
            audit=audits[own];old=states[own]
            states[own]=dict(primary_need=NEEDS[audit['primary']] if audit['primary']>=0 else None,
                mode=('need','principle')[audit['mode']],mode_urgency=audit['mode_urgency'],
                intent_action=chosen[own],age=old['age']+1 if old['intent_action']==chosen[own] else 0)
            row['outside']=audit['outside_root'];row['willing_offers']=len(audit['accepted'])
        assert fixed==digest([(m.personality,m.values) for m in mem.values()])
        trace.append(dict(tick=tick,rows=rows,negotiation=negotiation,
            # Private evaluation is for the offline audit, never a peer input.
            ballots=[dict(owner=b.owner,outside=b.outside,ranked=b.ranked) for b in ballots],
            evaluations=audits))
    actors=[dict(npc=a,profile=world.profiles[a]['id'],**world.state(a)) for a in world.actors]
    result=dict(seed=seed,domain=domain,layout=layout,contact=contact,scenario=scenario,arm=arm,turns=turns,
        actors=actors,alive=sum(a['health']>0 for a in actors),gain=sum(a['score'] for a in actors),
        agreements=sum(len(t['negotiation']['agreements']) for t in trace),
        proposals=sum(len(t['negotiation']['attempts']) for t in trace),
        rejections=sum(bool(a['refused']) for t in trace for a in t['negotiation']['attempts']),
        cooperation=sum(r['matched'] and r['kind']==r['other_kind']=='cooperate' for t in trace for r in t['rows'])//2,
        actions=dict(Counter(r['kind'] for t in trace for r in t['rows'])),personality_fixed=True)
    if root:
        name=f'{domain}-{layout}-{contact}-{scenario}-{seed}-{arm}.json.gz'
        raw=json.dumps(dict(result=result,trace=trace),separators=(',',':')).encode()
        with (Path(root)/name).open('wb') as f:
            with gzip.GzipFile(fileobj=f,mode='wb',mtime=0,filename='') as z:z.write(raw)
        result['trajectory']=name;result['sha256']=hashlib.sha256(raw).hexdigest()
    return result,trace


def worker(job):return run_one(job)[0]


def experiment(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    sources={p.relative_to(repo).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('reflex','tools') for p in (repo/folder).glob('*.py')}
    registration=dict(version='voluntary-coordination-v1',arguments=vars(args),sources=sources,
        acceptance=['all agreements require independent consent and legal, disjoint roots',
            'no personality edits or game/action-name branches in common protocol',
            'compare alive and gain by scenario/world/profile, not cooperation rate alone',
            'any new deaths or negative transfer prevent unconditional default adoption',
            'solo advantage, costly joint, changing opportunity and asymmetric resources are mandatory controls'],
        limits=['authored public worlds','bounded conditional methods','future consent not guaranteed','not a stable/global matching optimum'])
    (root/'registration.json').write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    for name in sources:
        dest=root/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((repo/name).read_bytes())
    jobs=[(str(root),seed,domain,layout,contact,scenario,arm,args.turns,args.horizon)
        for seed,domain,layout,contact,scenario,arm in itertools.product(range(args.seed_start,args.seed_start+args.seeds),
            args.domains,args.layouts,args.contacts,args.scenarios,args.arms)]
    start=time.perf_counter();runs=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(worker,jobs):
            runs.append(r)
            print(f'{len(runs)}/{len(jobs)} {r["domain"]} {r["scenario"]} {r["arm"]} alive={r["alive"]} gain={r["gain"]:.1f} agreements={r["agreements"]} {time.perf_counter()-start:.1f}s',flush=True)
    (root/'results.json').write_text(json.dumps(dict(registration=registration,runs=runs,seconds=time.perf_counter()-start),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed-start',type=int,default=92000)
    p.add_argument('--seeds',type=int,default=1);p.add_argument('--turns',type=int,default=320);p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--domains',nargs='+',default=list(RULES),choices=list(RULES))
    p.add_argument('--stage',default='development',choices=['development','confirmation','transfer','sensitivity'])
    p.add_argument('--layouts',nargs='+',default=['balanced4'],choices=list(LAYOUTS));p.add_argument('--contacts',nargs='+',default=['free'],choices=['free','public'])
    p.add_argument('--scenarios',nargs='+',default=['standard'],choices=SCENARIOS);p.add_argument('--arms',nargs='+',default=list(ARMS),choices=ARMS)
    experiment(p.parse_args())
