"""Fixed cross-rule study of recovery beyond two inaccurate forecasts.

No hidden law enters a controller. Full public response is revealed every turn,
numerical effect only every fourth turn. Actual and simulated owner updates,
checkpoint continuation, preferences and branch isolation are audited.
All parameters are shared across games. Repeated short bouts retain one owner;
this is not a continuous-world or human-level intelligence benchmark.
"""
import argparse, copy, json, math, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reflex.core import digest, character_key, compile_batch
from reflex.event_reader import EventReader
from reflex.loop_predictor import CategoricalReader
from reflex.decision_loop import DecisionLoop
from reflex.actor_model import ActorModel
from reflex.model_observer import resolve_feedback
from reflex.flow_rollout import rollout
from reflex.intertemporal import forecast
from reflex.laboratory import PROFILES
from tools.run_lifecycle_soak import Adapter, public_context, sample, GAMES


class RecoveryAdapter(Adapter):
    def forecast_context(self, s):
        # Probability callbacks inspect only this public projection. Avoid
        # re-enumerating every action/effect just to obtain a response law.
        encoded=json.dumps(s,sort_keys=True,separators=(',',':'))
        facts={f'public_{i:03d}':encoded[i:i+128] for i in range(0,len(encoded),128)}
        facts['visible_cue']=str((s['tick']//3)%2)
        return dict(scope=copy.deepcopy(self.scope),tick=s['tick'],facts=facts)

    def raw(self, s, state=None):
        c = super().raw(s, state)
        c['facts']['visible_cue'] = str((s['tick']//3) % 2)
        return c

    def legal(self, c):
        return tuple(sorted(self.rules.symbols(public_context(c)['world'])))

    def models(self, c):
        # Same incomplete library for EVERY arm: no hypothesis favors the last
        # response. All nevertheless allow every rule-permitted response.
        full = super().models(c)
        return {k: full[k] for k in self.symbols()[:2]}

    def supported_models(self, c):
        legal = self.legal(c)
        return {k: {z: sum(p for symbol, p in row.items()
                           if str(min(int(symbol), len(legal)-1)) == z) for z in legal}
                for k, row in self.models(c).items()}

    def actor_for(self, c, arm):
        actor = super().actor(c)
        if arm in ('joint', 'choice'):
            actor.predictor = CategoricalReader(c['scope'], self.symbols()[:2], self.models,
                self.outcome, known_conditionals=True)
            actor.decision_metric = 'persona'
            actor.response_adoption = 'persona_choice' if arm == 'choice' else 'joint'
        else:
            actor.predictor = EventReader(c['scope'], self.symbols()[:2], self.supported_models,
                self.outcome, legal_for=self.legal, model_id='recovery-v1:'+self.rules.model_id,
                known_conditionals=True, factors=('public-cue',),
                features_for=lambda cc: (cc['facts']['visible_cue'],), learned=arm != 'library')
        return actor


def truth(symbols, tick, turns, regime):
    stage = min(3, tick*4//turns)
    main = 0
    if regime == 'shift-return': main = (0, len(symbols)-1, 1, 0)[stage]
    elif regime == 'outside': main = len(symbols)-1
    elif regime == 'cue-change':
        cue = (tick//3) % 2
        main = len(symbols)-1 if cue == int(stage >= 2) else 0
    return {z: (.85 if int(z) == main else .15/(len(symbols)-1)) for z in symbols}


def predictive_law(reader, c):
    return reader.predict(c) if isinstance(reader, EventReader) else reader.forecasts(c)[0]


def run(job, arm):
    game, pid, seed, regime, turns, plan_every = job
    a = RecoveryAdapter(game, PROFILES[pid], seed % 2)
    s = a.start(); c = a.raw(s); actor = a.actor_for(c, arm)
    identity = character_key(c); checks = Counter(); traces = []; scores = []
    uniforms = np.random.default_rng(seed).random(turns); modeled_count = 0
    for tick in range(turns):
        if a.terminal(s): scores.append(a.goal(s)); s = a.start(s['round']+1, tick)
        c = a.raw(s, actor.state); req = a.request(c); before = digest(actor.record())
        assert character_key(c) == identity
        law = predictive_law(actor.predictor, c)
        actual = truth(a.symbols(), tick, turns, regime)
        # Diagnostics use the true law AFTER the controller inputs are built.
        # They never modify the owner, request, predictor or planner.
        legal = a.legal(c)
        projected = lambda p: {z: sum(v for k, v in p.items()
            if str(min(int(k), len(legal)-1)) == z) for z in legal}
        ptrue = projected(actual); plaw = projected(law)
        fb = lambda b,e,k,cc: a.feedback(b,e,k,cc,'channels','censored')
        plan = {}
        def planner(cs, ds):
            nonlocal modeled_count
            model = ActorModel(actor, root_request=req, request_for=a.request, feedback=fb)
            def advance(w, k, model_seed):
                u = np.random.default_rng(np.random.SeedSequence([model_seed, w['tick']])).random()
                # Root factual memory is fixed, but public features/legal sets
                # are those of this future state. No future real reveal enters.
                future_law = predictive_law(actor.predictor, a.forecast_context(w))
                end, flow, _ = a.transition(w, k, sample(future_law, u))
                return end, flow
            paths, audit = rollout(cs[0], s, observe=a.raw, advance=advance,
                terminal=a.terminal, horizon=3, seeds=(1801,2702,3603),
                observer=model, policy=actor.policy)
            assert digest(actor.record()) == before
            checks['isolated_plans'] += 1
            modeled_count += sum(len(row['actions']) for rows in audit.values() for row in rows)
            plan['actions'] = {k: [r['actions'] for r in rows] for k, rows in audit.items()}
            return forecast(cs[0], paths, horizon=3, unit='decisions', target='incremental-purpose')
        mirror = ActorModel(actor, root_request=req, request_for=a.request, feedback=fb)
        prepared, mreq, _, _ = mirror.prepare(s,c,prepared=True); mirror.preview()
        # Same-person one-step oracle is diagnostic only, never a policy target.
        probe = copy.deepcopy(prepared)
        for row in probe['actions']:
            row['outcomes'] = [dict(a.outcome(c,row['id'],z),p=p) for z,p in ptrue.items()]
        batch = compile_batch([probe]); oracle = actor.policy.decide(batch,False)
        result = DecisionLoop.decide_batch([(actor,req)],False,planner if plan_every and tick % plan_every == 0 else None)[0]
        key = result['decision']['action_id']; mirror.commit_choice(key)
        mirror.adopt_state(result['decision']['next_state'])
        chosen = batch.ids[0].index(key)
        best = int(oracle.action[0])
        regret = float(oracle.scores[0,best] - oracle.scores[0,chosen])
        end, flow, symbol = a.transition(s,key,sample(actual,uniforms[tick]))
        resolve_feedback(actor,result['ticket'],fb(s,end,key,c)); mirror.advance(s,end,key,prepared,mreq)
        assert digest(actor.record()) == digest(mirror.actor.record())
        checks['same_feedback_same_owner'] += 1
        if tick % 29 == 0:
            record = json.loads(json.dumps(actor.record()))
            actor = DecisionLoop.from_record(c,record,predictor=actor.predictor)
            assert record == actor.record(); checks['checkpoint_roundtrips'] += 1
        observed = actor.predictor.observations if isinstance(actor.predictor,EventReader) else actor.predictor.tracker.observations
        assert observed == tick+1
        checks['preferences_and_observations'] += 1
        traces.append(dict(tick=tick,round=s['round'],action=key,response=symbol,
            goal=a.goal(end),law=law,truth_diagnostic=ptrue,
            log_loss=-math.log(max(plaw[symbol],1e-300)),
            expected_brier=sum((plaw[z]-ptrue[z])**2 for z in legal)/len(legal),
            immediate_persona_regret=regret,oracle_action=batch.ids[0][best],
            reading=result['reading'],planned=bool(plan),
            owner_digest=digest(actor.record()),plan=plan,
            learned_weight=actor.predictor.learner.diagnostics(actor.predictor.memory)['learned_weight']
                if isinstance(actor.predictor,EventReader) else None))
        s=end
    if a.terminal(s): scores.append(a.goal(s))
    return dict(game=game,profile=PROFILES[pid]['id'],seed=seed,regime=regime,arm=arm,
        turns=turns,plan_every=plan_every,score=float(np.mean(scores)),scores=scores,
        modeled_decisions=modeled_count,checks=dict(checks),trace=traces,
        log_loss=float(np.mean([t['log_loss'] for t in traces])),
        expected_brier=float(np.mean([t['expected_brier'] for t in traces])),
        immediate_persona_regret=float(np.mean([t['immediate_persona_regret'] for t in traces if not t['planned']])))


def evaluate(job, arms): return [run(job,arm) for arm in arms]


def summary(rows):
    pairs=[]
    keys=sorted({(r['game'],r['profile'],r['seed'],r['regime']) for r in rows})
    for key in keys:
        by={r['arm']:r for r in rows if (r['game'],r['profile'],r['seed'],r['regime'])==key}
        pairs.append(dict(game=key[0],profile=key[1],seed=key[2],regime=key[3],
            **{metric:{arm:r[metric] for arm,r in by.items()}
                for metric in ('score','log_loss','expected_brier','immediate_persona_regret')}))
    return dict(runs=len(rows),decisions=sum(r['turns'] for r in rows),
        modeled_decisions=sum(r['modeled_decisions'] for r in rows),pairs=pairs,
        checks={k:sum(r['checks'].get(k,0) for r in rows) for k in rows[0]['checks']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',action='store_true')
    p.add_argument('--turns',type=int,default=256);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--pilot',action='store_true');p.add_argument('--holdout',action='store_true')
    p.add_argument('--games',nargs='+',choices=GAMES,default=list(GAMES))
    p.add_argument('--arms',nargs='+',choices=('joint','choice','library','recovery'))
    p.add_argument('--plan-every',type=int,default=12);a=p.parse_args();root=Path(a.root)
    if not a.frozen:
        from tools.freeze_experiment import dispatch
        args=['--turns',str(a.turns),'--workers',str(a.workers),'--plan-every',str(a.plan_every)]
        if a.pilot:args+=['--pilot']
        if a.holdout:args+=['--holdout']
        args+=['--games',*a.games]
        if a.arms:args+=['--arms',*a.arms]
        dispatch(root,Path(__file__).name,args);return
    root.mkdir(parents=True,exist_ok=True)
    seeds=(70,103) if a.holdout else (43,)
    jobs=[(g,p,s,r,a.turns,a.plan_every) for g in a.games for p in range(4) for s in seeds
          for r in ('steady','shift-return','outside','cue-change')]
    arms=('choice','library','recovery') if a.pilot else ('joint','choice','library','recovery')
    if a.arms:arms=tuple(a.arms)
    if a.pilot:jobs=[(g,0,43,r,a.turns,a.plan_every) for g in GAMES for r in ('steady','outside')]
    (root/'registration.json').write_text(json.dumps(dict(jobs=jobs,arms=arms,semantics=__doc__,
        holdout_seeds_reserved=[70,103],parameters='unchanged EventLearner defaults; no game-specific tuning'),indent=2)+'\n',encoding='utf-8')
    rows=[];started=time.monotonic()
    with (root/'trajectories.jsonl').open('w',encoding='utf-8') as f,ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(evaluate,j,arms):j for j in jobs}
        for i,fu in enumerate(as_completed(futures)):
            for row in fu.result():rows.append(row);f.write(json.dumps(row,separators=(',',':'))+'\n')
            f.flush();print(json.dumps(dict(done=i+1,total=len(jobs),job=futures[fu],seconds=round(time.monotonic()-started,1))),flush=True)
    (root/'summary.json').write_text(json.dumps(summary(rows),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
