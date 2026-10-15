"""Long-lived endogenous NPC observation and matched-state experience ablations.

This tests an existing social/reflex configuration, NOT all optional modules.
No prescribed betrayal, reconciliation, new personality or future revelation.
Material-greedy is a narrow comparator, not GOAP/PsychSim or all Utility AI.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import itertools
import json
import time
import numpy as np

from reflex.core import Policy, compile_batch, digest
from reflex.encounters import EncounterWorld, after_action, RULES, LAYOUTS
from reflex.encounter_experiment import interaction_counts
from reflex.social_withdrawal import WithdrawalSocialPopulation, WithdrawalSocialMemory

ARMS=('amnesic','prediction','social','net_effects','material_greedy')
FIELDS=('needs','effects','probability','legal')


def need_state(state):
    return dict(physiology=1-state['energy'],
        safety=max(1-state['health']/8,max(0,(2-state['stock'])/2)))


def inputs(world, *, forget=False, net_effects=False):
    source=copy.deepcopy(world) if forget else world
    if forget:
        for history in source.history.values():history.clear()
    cs,bindings=source.contexts()
    if net_effects:
        for own,c in zip(source.actors,cs):
            before=source.state(own); deficits=need_state(before)
            for a in c['actions']:
                kind,_=source.catalog[own][a['id']]
                for outcome in a['outcomes']:
                    # Raw rules already supply gain and pre-maintenance health.
                    # Use the very same full-tick transition as real execution.
                    after=after_action(before,kind,outcome['objective'],outcome['needs'].get('safety',0),source.rules['upkeep'])
                    future=need_state(after)
                    for name in deficits:outcome['needs'][name]=deficits[name]-future[name]
    return cs,bindings


class MaterialGreedy(Policy):
    """One-step expected material gain, with exactly the same legal options."""
    def decide(self,b,stochastic=True,**kwargs):
        d=super().decide(b,False)
        scores=np.einsum('nak,nak->na',b.probability,b.effects[:,:,:,0])
        eligible=b.legal&~b.known_failure
        return replace(d,action=np.argmax(np.where(eligible,scores,-np.inf),1),scores=scores,eligible=eligible)


def choose(pop, cs, bindings, social):
    b=compile_batch(cs)
    d=pop.step(bindings=bindings if social else [{} for _ in cs],
        **{k:getattr(b,k) for k in FIELDS})
    return pop.action_ids(d),d


def comparison_outcome(world, own, choice, joint):
    """Evaluator only: unilateral alternative against already chosen opponents.

    The alternative was selected before reveal. This evaluator is never passed
    to a chooser or memory, and is only a one-step diagnostic, not future value.
    """
    kind,target=world.catalog[own][choice]
    other,recipient=world.catalog[target][joint[target]] if target else (None,None)
    matched=target is not None and recipient==own
    failed=False
    if matched and kind==other=='cooperate':
        index={a:i for i,a in enumerate(world.actors)}
        luck=np.random.default_rng(np.random.SeedSequence([world.seed,world.tick])).random((len(index),len(index)))
        failed=luck[min(index[own],index[target]),max(index[own],index[target])]<.05
    gain,health=world.consequences(kind,other,matched,failed)
    return after_action(world.state(own),kind,gain,health,world.rules['upkeep'])


def run_one(job):
    root,seed,domain,layout,contact,arm,turns=job
    world=EncounterWorld(seed,domain,layout,contact)
    initial,_=inputs(world,net_effects=arm=='net_effects')
    policy=MaterialGreedy(principle_priority='finite') if arm=='material_greedy' else Policy(principle_priority='finite')
    pop=WithdrawalSocialPopulation(initial,policy)
    axes=digest([dict(personality=c['personality'],values=c['values']) for c in initial])
    trace=[]; probes=[]; prediction=Counter(); harms=set(); repaired=set()
    for tick in range(turns):
        cs,bindings=inputs(world,forget=arm=='amnesic',net_effects=arm=='net_effects')
        forecast_world=world
        if arm=='amnesic':
            forecast_world=copy.deepcopy(world)
            for history in forecast_world.history.values():history.clear()
        pre_forecast={a:{b:list(forecast_world.forecast(a,b)) for b in world.actors if b!=a} for a in world.actors}
        # Same actor, board, needs, RNG, and intent; only remembered experience removed.
        probe_actions=None
        if arm in ('social','net_effects') and tick>0 and tick%16==0:
            empty=copy.deepcopy(pop)
            empty.social=[WithdrawalSocialMemory(c) for c in initial]
            blank,bb=inputs(world,forget=True,net_effects=arm=='net_effects')
            probe_actions,_=choose(empty,blank,bb,True)
        actions,decision=choose(pop,cs,bindings,arm in ('social','net_effects'))
        joint=dict(zip(world.actors,actions))
        if probe_actions is not None:
            for own,selected,alternative in zip(world.actors,actions,probe_actions):
                if world.health[own]<=0:continue
                actual=comparison_outcome(world,own,selected,joint)
                counter=comparison_outcome(world,own,alternative,joint)
                probes.append(dict(tick=tick,npc=own,profile=world.profiles[own]['id'],
                    chosen=selected,without_experience=alternative,
                    gain_delta=actual['score']-counter['score'],health_delta=actual['health']-counter['health']))
        rows,events=world.resolve(joint)
        if arm!='amnesic':pop.observe_batch(events)
        for row in rows:
            own=row['npc']; target=row['target']
            if target is not None:
                p=pre_forecast[own][target]
                truth=[0.,0.,0.];truth[(0 if row['other_kind']=='cooperate' else 1) if row['matched'] else 2]=1.
                prediction['brier_sum']+=sum((x-y)**2 for x,y in zip(p,truth));prediction['count']+=1
                if row['kind']=='cooperate' and row['other_kind']=='claim':harms.add((own,target))
                if row['kind']==row['other_kind']=='cooperate' and not row['luck_failure'] and (own,target) in harms:
                    repaired.add((own,target))
            row['primary_need']=int(decision.primary[world.actors.index(own)])
            row['selected_appraisal']=copy.deepcopy(pop.social_audit[world.actors.index(own)].get(row['action']))
        fixed=axes==digest([dict(personality=m.personality,values=m.values) for m in pop.social])
        assert fixed
        trace.append(dict(tick=tick,rows=rows,fixed_personality=fixed))
    actors=[dict(npc=a,profile=world.profiles[a]['id'],**world.state(a),
        active_turns=sum(t['rows'][i]['kind']!='finished' for t in trace)) for i,a in enumerate(world.actors)]
    counts,_=interaction_counts(trace,contact)
    blocks=[]
    for start in range(0,turns,80):
        rs=[r for t in trace[start:start+80] for r in t['rows']]
        blocks.append(dict(start=start,gain=sum(r['gain'] for r in rs),active=sum(r['kind']!='finished' for r in rs),
            cooperative=sum(r['matched'] and r['kind']==r['other_kind']=='cooperate' for r in rs)//2,
            exploited=sum(r['matched'] and r['kind']=='cooperate' and r['other_kind']=='claim' for r in rs),
            actions=dict(Counter(r['kind'] for r in rs)),
            primary_needs=dict(Counter(str(r['primary_need']) for r in rs))))
    result=dict(seed=seed,domain=domain,layout=layout,contact=contact,arm=arm,turns=turns,
        actors=actors,interaction=counts,blocks=blocks,prediction=dict(prediction),
        repaired_directed_relations=len(repaired),harm_directed_relations=len(harms),
        probes=dict(count=len(probes),changed=sum(p['chosen']!=p['without_experience'] for p in probes),
            material_better=sum(p['gain_delta']>1e-9 for p in probes),material_worse=sum(p['gain_delta']< -1e-9 for p in probes),
            gain_delta=sum(p['gain_delta'] for p in probes),health_delta=sum(p['health_delta'] for p in probes)),
        axes_digest=axes,personality_fixed=True)
    name=f'{domain}-{layout}-{contact}-{seed}-{arm}.json.gz'
    raw=json.dumps(dict(run=result,trace=trace,probes=probes),ensure_ascii=False,separators=(',',':')).encode()
    with (Path(root)/name).open('wb') as dest:
        with gzip.GzipFile(fileobj=dest,mode='wb',mtime=0,filename='') as f:f.write(raw)
    result['trajectory']=name;result['content_sha256']=hashlib.sha256(raw).hexdigest()
    return result


def summarize(runs):
    totals={};paired={}
    for arm in ARMS:
        rs=[r for r in runs if r['arm']==arm];actors=[a for r in rs for a in r['actors']]
        if not rs:continue
        totals[arm]=dict(worlds=len(rs),actors=len(actors),alive=sum(a['health']>0 for a in actors),
            mean_actor_gain=float(np.mean([a['score'] for a in actors])),
            active_turns=sum(a['active_turns'] for a in actors),
            cooperation_pairs=sum(r['interaction']['cooperation_pairs'] for r in rs),
            exploited=sum(r['interaction'].get('exploited',0) for r in rs),
            repaired_relations=sum(r['repaired_directed_relations'] for r in rs),
            repair_worlds=sum(r['repaired_directed_relations']>0 for r in rs))
    lookup={(r['domain'],r['layout'],r['contact'],r['seed'],r['arm']):r for r in runs}
    for new,old in [('prediction','amnesic'),('social','prediction'),('net_effects','social'),('social','material_greedy')]:
        counts=Counter();by_domain={};deltas=[]
        for key,r in lookup.items():
            if key[-1]!=new or key[:-1]+(old,) not in lookup:continue
            baseline=lookup[key[:-1]+(old,)];world_delta=0
            for a,b in zip(r['actors'],baseline['actors']):
                assert a['npc']==b['npc'] and a['profile']==b['profile']
                delta=a['score']-b['score'];tag='better' if delta>1e-9 else 'worse' if delta< -1e-9 else 'same'
                counts[tag]+=1;by_domain.setdefault(key[0],Counter())[tag]+=1;world_delta+=delta
                counts['new_deaths']+=a['health']<=0<b['health'];counts['avoided_deaths']+=b['health']<=0<a['health']
            deltas.append(world_delta/len(r['actors']))
        paired[new+'_vs_'+old]=dict(actor_counts=dict(counts),by_domain=by_domain,world_pairs=len(deltas),
            world_mean_delta=float(np.mean(deltas)) if deltas else None)
    return dict(totals=totals,paired=paired)


def experiment(root, seeds, turns=320, workers=4, arms=ARMS):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    names=['reflex/core.py','reflex/runtime.py','reflex/encounters.py','reflex/social.py',
        'reflex/social_projection.py','reflex/social_withdrawal.py','reflex/laboratory.py',
        'reflex/examples.py','reflex/encounter_experiment.py','tools/long_lived_npcs.py']
    plan=dict(version='long-lived-npcs-v1',seeds=list(seeds),turns=turns,arms=list(arms),
        domains=list(RULES),layouts=list(LAYOUTS),contacts=['free','public'],
        same_state_probe_every=16,window=80,rules=RULES,workers=workers,
        acceptance='diagnose long-lived integration; do not promote by aggregate gain alone or change fixed personality',
        sources={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in names},
        net_effect_adoption_gate='no additional deaths, actor material improvements at least regressions; otherwise diagnostic only',
        limitations=['two authored worlds','no future planning or general emotion updater','no autonomous goal replacement',
            'all publicly observed history','same-world actors are correlated','material-greedy is not representative of all existing methods'])
    (root/'registration.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    for name in names:
        dest=root/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((repo/name).read_bytes())
    jobs=[(str(root),seed,domain,layout,contact,arm,turns) for domain,layout,contact,seed,arm in
        itertools.product(plan['domains'],plan['layouts'],plan['contacts'],seeds,arms)]
    start=time.perf_counter();runs=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,r in enumerate(pool.map(run_one,jobs)):
            runs.append(r)
            if (i+1)%12==0:print(f'{i+1}/{len(jobs)} worlds, {time.perf_counter()-start:.1f}s',flush=True)
    summary=summarize(runs)
    result=dict(plan=plan,runs=runs,summary=summary,seconds=time.perf_counter()-start)
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--seed-start',type=int,default=91100);parser.add_argument('--seeds',type=int,default=8)
    parser.add_argument('--turns',type=int,default=320);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=ARMS);args=parser.parse_args()
    experiment(args.output,range(args.seed_start,args.seed_start+args.seeds),args.turns,args.workers,args.arms)
