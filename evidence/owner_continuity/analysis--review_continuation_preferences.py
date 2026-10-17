"""Compare two realized continuations with the SAME root person's full selector.

This evaluator reads already completed replay paths. It is hindsight, not the
person's information at decision time or proof that a sacrifice was intentional.
The saved replay is valid only for the benchmark's action-independent responses.
"""
import argparse, copy, hashlib, json, math, sys
from collections import Counter
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.deliberation import select
from reflex.intertemporal import Branch
from reflex.laboratory import PROFILES
from reflex.purpose_plan import goal_forecast
from tools.compare_owned_studies import compare
from tools.continuing_games import ContinuingGame


def preference(g, owner, context, root, paths, aliases):
    c=copy.deepcopy(context);c['actions']=[a for a in c['actions'] if a['id']==root]
    records={}
    class Capture:
        def decide(self,b,*args,**kwargs):
            d=owner.policy.decide(b,*args,**kwargs)
            for i,k in enumerate(b.ids[0]):
                records[k]=dict(tier_eligible=bool(d.eligible[0,i]),score=float(d.scores[0,i]),
                    strongest_value=float(d.features[0,i,2]))
            return d
    branches={aliases[label]:[Branch(1.,tuple(p['flows']),(1.,)*len(p['flows']))] for label,p in paths.items()}
    audit={aliases[label]:[dict(actions=p['actions'],terminal=g.terminal(p['endpoint']),
        assessment=g.goal(p['endpoint']).record())] for label,p in paths.items()}
    f=goal_forecast(c,branches,audit,horizon=len(next(iter(paths.values()))['flows']),
        unit='owner-decisions',target=g.target,policy=Capture(),plan_roots={v:root for v in aliases.values()},
        settlement_weight='absolute',objective_origin=g.goal(g.world(c)),opportunity_regret=True)
    decisions,selection=select((c,),f,owner.policy)
    assert decisions is not None
    label_by_alias={v:k for k,v in aliases.items()}
    label_by_plan={k:label_by_alias[v['proposal']] for k,v in f.audit['plans'].items()}
    rows={label_by_plan[k]:dict(v,purpose=g.goal(paths[label_by_plan[k]]['endpoint']).value,
        purpose_eligible=k in f.purpose and f.purpose[k]>=max(f.purpose.values())-f.max_regret-1e-12)
        for k,v in records.items()}
    return dict(selected=label_by_plan[selection['selected_plan']],paths=rows)


def review(studies,replay):
    validated=compare(studies,[])
    rows={}
    rule=hashlib.sha256((R/'tools/continuing_games.py').read_bytes()).hexdigest()
    for study in map(Path,studies):
        manifest=json.loads((study/'source_snapshot.json').read_text())
        assert manifest['tools/continuing_games.py']==rule
        for name,expected in manifest.items():
            if name.startswith('reflex/'):
                assert hashlib.sha256((R/name).read_bytes()).hexdigest()==expected,('frozen core differs',name)
        for row in map(json.loads,(study/'trajectories.jsonl').read_text().splitlines()):
            key=tuple(row[k] for k in ('game','profile','seed','regime','arm','interval'))
            assert key not in rows;rows[key]=row
    replay_path=Path(replay);prior=json.loads(replay_path.read_text());out=[]
    for record in prior['runs']:
        key=tuple(record[k] for k in ('game','profile','seed','regime','arm','interval'))
        r=rows[key];g=ContinuingGame(r['game'],next(p for p in PROFILES if p['id']==r['profile']),r['turns'],r['seed']%2)
        cases=[]
        for case in record['cases']:
            tick=case['tick'];t=r['trace'][tick];n=len(case['actual_actions'])
            segment=r['trace'][tick:tick+n]
            assert case['actual_actions']==[x['action'] for x in segment]
            c=g.context(t['before']);segmented=t['checkpoint']['predictor']['memory']['format']=='segmented-events-v1'
            owner=DecisionLoop.from_record(c,t['checkpoint'],predictor=g.actor(c,learning='segmented' if segmented else 'windows').predictor)
            saved=digest(owner.record());assert saved==t['before_owner']
            effective=owner.preview(g.request(c))[0];paths={}
            for label,field in [('actual','actual_actions'),('reference','reference_actions')]:
                w=copy.deepcopy(t['before']);actions=case[field];flows=[]
                assert len(actions)==n and actions[0]==t['action']
                for action,observed in zip(actions,segment):
                    assert action in g.keys(w)
                    w,flow=g.transition(w,action,observed['response']);flows.append(flow)
                expected=case['executed'] if label=='actual' else case['reflex_replay']
                assert math.isclose(g.goal(w).value,expected,rel_tol=0,abs_tol=1e-12)
                paths[label]=dict(actions=actions,flows=flows,endpoint=w)
            a=preference(g,owner,effective,t['action'],paths,dict(actual='a',reference='b'))
            b=preference(g,owner,effective,t['action'],paths,dict(actual='b',reference='a'))
            assert a['paths']==b['paths'],'labels changed valuation'
            tied=a['selected']!=b['selected']
            if tied:
                assert a['paths']['actual']['tier_eligible'] and a['paths']['reference']['tier_eligible']
                assert math.isclose(a['paths']['actual']['score'],a['paths']['reference']['score'],rel_tol=0,abs_tol=1e-12)
            identical=preference(g,owner,effective,t['action'],dict(actual=paths['actual'],reference=paths['actual']),dict(actual='a',reference='b'))
            assert identical['paths']['actual']==identical['paths']['reference']
            assert identical['paths']['actual']['tier_eligible'] and identical['paths']['actual']['purpose_eligible']
            assert digest(owner.record())==saved
            cases.append(dict(tick=tick,purpose_delta=case['control_contribution'],
                hindsight_selected='tie' if tied else a['selected'],paths=a['paths']))
        out.append(dict(**{k:r[k] for k in ('game','profile','seed','regime','arm','interval')},cases=cases,
            selected_counts=dict(Counter(c['hindsight_selected'] for c in cases)),
            purpose_losing_windows=[dict(tick=c['tick'],delta=c['purpose_delta'],selected=c['hindsight_selected'])
                                    for c in cases if c['purpose_delta'] < -1e-12]))
    return dict(sources=validated['sources'],replay=str(replay_path),
        replay_sha256=hashlib.sha256(replay_path.read_bytes()).hexdigest(),rule_sha256=rule,runs=out,
        checks=['saved root owner identity','same realized response sequence','reproduced endpoint values',
                'proposal label permutation','identical paths tie','no mutation of factual owner'],
        limitations=[__doc__,'Single realized paths do not estimate their ex-ante risk or expected utility.',
            'Both paths share the actual root. This does not compare alternative initial actions.',
            'The decision-time person evaluates the complete paths. Later priorities may legitimately differ.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--studies',nargs='+',required=True)
    p.add_argument('--replay',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=review(a.studies,a.replay);Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(r,cases=len(r['cases'])) for r in result['runs']],indent=2))
