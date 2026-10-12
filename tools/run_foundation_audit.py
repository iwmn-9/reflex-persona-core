"""Compare complete episodes before changing the production decision policy.

purpose4 is a diagnostic root selector, NOT a personality-preserving NPC or
an optimal opponent. All planning arms model the same future persona reflex.
The generated graph oracle knows public rules, never future chance outcomes.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reflex.core import Policy, character_key, digest
from reflex.decision_loop import DecisionLoop
from reflex.contingent_games import make_bridge
from reflex.forecast_control import ForecastBudget, restrict
from reflex.integrated_trials import request_for
from reflex.deliberation import select
from reflex.purpose_plan import goal_forecast
from reflex.flow_rollout import rollout
from reflex.laboratory import profiles
from tools.run_contingent_coverage import cases

GAMES = ('auction_contest', 'hagetaka', 'gambling', 'empire_race',
         'team_combat', 'team_projects', 'no_thanks', 'goofspiel')
ARMS = ('reflex', 'persona4', 'purpose4', 'persona8')


def graph_values(trial):
    """Independent exact probability backup, no sampled future or persona fit."""
    v = {(trial.length, n): float(n in trial.winning) for n in range(trial.width)}
    q = {}
    for t in reversed(range(trial.length)):
        for n in range(trial.width):
            for a in ('option-u', 'option-v'):
                q[t, n, a] = sum(v[t+1, x] for x in trial.edges[t, n, a]) / 2
            v[t, n] = max(q[t, n, a] for a in ('option-u', 'option-v'))
    return v, q


def prepare(case, profile, seed):
    b = make_bridge(profile, case); w = b.start(); prefix = []
    # Valid public midgames expose more than the already-saturated starts.
    # This independent legal opening is identical across arms and personas.
    opening = random.Random(case.get('opening_seed', 0))
    for _ in range(case.get('opening_steps', 0)):
        if b.terminal(w): break
        c = b.observe(b.project(w))
        legal = sorted(a['id'] for a in c['actions'] if a['legal'] and not a['known_failure'])
        key = opening.choice(legal); before = b.record(w)
        w, row = b.actual(w, key, seed)
        prefix.append(dict(before=before, root=key, after=b.record(w), flow=row))
    return b, w, prefix


def play(job):
    case, profile, seed, arm = job
    started = time.perf_counter(); b, w, prefix = prepare(case, profile, seed)
    initial = b.record(w)
    if b.terminal(w):
        return dict(job=job, case=case, profile=profile['id'], seed=seed, arm=arm,
                    initial=initial, prefix=prefix, trace=[], actions=[], goal=b.goal(w).record(),
                    pre_settled=True, model_steps=0, replay=True, elapsed_seconds=time.perf_counter()-started)
    c = b.observe(b.project(w)); fixed = {k:copy.deepcopy(c[k]) for k in ('scope','seed','personality','values','objective')}
    p = Policy(principle_priority='personal', commitments={character_key(c):.5})
    actor = DecisionLoop(c, policy=p); trace=[]
    budget = ForecastBudget(horizon=8 if arm=='persona8' else 4, direct_continuation=True)
    model_steps=0
    oracle = graph_values(b.rules) if case['game']=='public_graph' else None
    while not b.terminal(w):
        projected=b.project(w); c=b.observe(projected, actor.state); req=request_for(c, case['game']); capture={}
        assert all(c[k]==v for k,v in fixed.items())
        saved=digest(actor.record())
        def planner(cs, ds):
            # Recover every modeled root before persona-tier pruning, using
            # identical branches. Finite policy here ONLY exposes root scores.
            # It is not substituted into the modeled future person's choices.
            paths, audit=rollout(cs[0], projected, observe=b.observe, advance=b.modeled_advance,
                terminal=b.terminal, assess=lambda s:b.goal(s).record(), horizon=budget.horizon,
                seeds=budget.seeds, policy=p)
            f=goal_forecast(cs[0],paths,audit,horizon=budget.horizon,unit=b.unit,
                target=b.target,policy=p,settlement_weight='absolute')
            all_f=goal_forecast(cs[0], paths, audit, horizon=budget.horizon, unit=b.unit,
                target=b.target, policy=Policy(principle_priority='finite'), settlement_weight='absolute')
            _, vote=select(cs, f, p)
            persona=vote.get('selected_plan'); best=max(all_f.purpose.values())
            # Keep persona choice on an exact purpose tie to avoid treating
            # a different tie convention as newly acquired competence.
            wanted=persona if persona in all_f.purpose and abs(all_f.purpose[persona]-best)<1e-12 else min(
                k for k,v in all_f.purpose.items() if abs(v-best)<1e-12)
            selected=wanted if arm=='purpose4' else persona
            capture.update(predicted={all_f.roots[k][0]:v for k,v in all_f.purpose.items()},
                persona=f.roots[persona][0], purpose=all_f.roots[wanted][0],
                persona_tier=[f.roots[k][0] for k in f.purpose], planned_purpose_cost=best-all_f.purpose[selected],
                settled_branches=sum(x['terminal'] for rows in audit.values() for x in rows),
                branches=sum(len(rows) for rows in audit.values()),
                model_steps=sum(len(x['actions']) for rows in audit.values() for x in rows))
            assert digest(actor.record())==saved
            return restrict(all_f,wanted) if arm=='purpose4' else f
        if arm=='oracle':
            q=oracle[1]; t,n=w.position
            key=min(('option-u','option-v'),key=lambda a:(-q[t,n,a],a))
            # This diagnostic bypasses personality scoring, but uses legal
            # public rules. No controller sees the eventual chance bit.
            chosen=copy.deepcopy(c); chosen['actions']=[a for a in c['actions'] if a['id']==key]
            req=request_for(chosen, case['game']); result=actor.decide(req,False)
        else:
            result=DecisionLoop.decide_batch([(actor,req)],False,None if arm=='reflex' else planner)[0]
            key=result['decision']['action_id']
        before=b.record(w); qrow=None
        if oracle:
            t,n=w.position; qrow={a:oracle[1][t,n,a] for a in ('option-u','option-v')}
        w,flow=b.actual(w,key,seed); actor.abandon(result['ticket'])
        model_steps+=capture.get('model_steps',0)
        assert not actor.memory.entries
        trace.append(dict(before=before,after=b.record(w),root=key,flow=flow,goal=b.goal(w).record(),
            forecast=capture,oracle_q=qrow,oracle_regret=None if qrow is None else max(qrow.values())-qrow[key],
            decision=result['decision'],model_steps=capture.get('model_steps',0)))
        if len(trace)>1000: raise AssertionError('unsettled full episode')
    check=b.start()
    for row in prefix+trace:
        assert b.record(check)==row['before']; check,flow=b.actual(check,row['root'],seed)
        assert b.record(check)==row['after'] and flow==row['flow']
    assert b.terminal(check) and b.goal(check).record()==b.goal(w).record()
    assert DecisionLoop.from_record(c,actor.record(),policy=p).record()==actor.record()
    return dict(job=job,case=case,profile=profile['id'],seed=seed,arm=arm,initial=initial,prefix=prefix,
        actions=[t['root'] for t in trace],trace=trace,goal=b.goal(w).record(),pre_settled=False,
        oracle_initial=None if oracle is None else oracle[0][0,0],model_steps=model_steps,replay=True,
        elapsed_seconds=time.perf_counter()-started)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--stage',choices=('pilot','development','confirmation','graphs'),required=True)
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--frozen',action='store_true');a=ap.parse_args()
    root=Path(a.root).resolve()
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        return dispatch(root,Path(__file__).name,['--stage',a.stage,'--workers',str(a.workers)])
    if (root/'preregister.json').exists():raise FileExistsError('fresh cohort required')
    if a.stage=='graphs':
        selected=[dict(game='public_graph',graph=g,length=length) for length in (5,12) for g in range(23100,23108)]
        seeds=range(33100,33104); arms=('reflex','persona4','purpose4','oracle')
    else:
        selected=[c for c in cases('confirmation' if a.stage=='confirmation' else 'development') if c['game'] in GAMES]
        if a.stage=='confirmation':
            selected=[dict(c,opening_steps=3,opening_seed=34100+i) for i,c in enumerate(selected)]
        seeds=range(31100,31104) if a.stage=='confirmation' else range(30100,30102)
        arms=ARMS
        if a.stage=='pilot':selected=selected[:1];seeds=range(30100,30101)
    jobs=[]
    for c in selected:
        for s in seeds:
            scenario=copy.deepcopy(c)
            if c['game'] in ('auction_contest','hagetaka','no_thanks'):
                scenario['deal']=c.get('deal',0)+(s-min(seeds))*19
            if 'opening_seed' in scenario:scenario['opening_seed']+=s-min(seeds)
            for p in profiles():
                for arm in arms:jobs.append((scenario,p,s,arm))
    pre=dict(format='foundation-audit-v1',stage=a.stage,jobs=jobs,
        controls=['production core unchanged','same four fixed profiles and rigidity .5',
            'same public model banks401..404, no actual future RNG input','no immediate learned outcomes; shared planning diagnosis',
            'all game errors, losses and settled openings retained','all actual transitions replayed'],
        hypotheses=['same-model purpose selection can expose preference cost separately from model inadequacy',
            'a longer horizon alone is insufficient for robust competence','flat cutoff proxies can conceal valuable preparation'],
        acceptance=['report each game/persona separately; do not pool unlike scores',
            'no default promotion from this diagnosis','graph DP oracle uses full public transition probabilities',
            'saturated and deterministic duplicate conditions identified, not counted as independent proof'],
        limitations=['authored game adapters and opponent policies','purpose4 still models future persona reflex; not optimal CPU',
            'oracle regret measures objective opportunity, not morally correct personality behavior',
            'no new human-level, social-fun or combined-learning claim'],
        source_snapshot_sha256=hashlib.sha256((root/'source_snapshot.json').read_bytes()).hexdigest())
    (root/'preregister.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[]; errors=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(root/'trajectories.jsonl').open('x',encoding='utf-8') as stream:
        fs={pool.submit(play,j):i for i,j in enumerate(jobs)}
        for f in as_completed(fs):
            i=fs[f]
            try:r=f.result()
            except Exception as e:
                errors.append(dict(index=i,job=jobs[i],error=repr(e)));print('ERROR',i,repr(e),flush=True);continue
            r['index']=i;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            slim={k:v for k,v in r.items() if k!='trace'}
            slim.update(steps=len(r['trace']),oracle_regret=sum(t['oracle_regret'] or 0 for t in r['trace']),
                purpose_disagreements=sum(t['forecast'].get('persona')!=t['forecast'].get('purpose') for t in r['trace']))
            rows.append(slim)
            print(len(rows)+len(errors),'/',len(jobs),r['case']['game'],r['profile'],r['arm'],r['goal'],round(r['elapsed_seconds'],1),flush=True)
    result=dict(format=pre['format'],stage=a.stage,planned=len(jobs),episodes=len(rows),errors=errors,runs=rows,
        elapsed_seconds=time.perf_counter()-start,preregister_sha256=digest(pre),
        trajectories_sha256=hashlib.sha256((root/'trajectories.jsonl').read_bytes()).hexdigest())
    (root/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if errors:raise RuntimeError(f'{len(errors)} incomplete jobs retained')


if __name__=='__main__':main()
