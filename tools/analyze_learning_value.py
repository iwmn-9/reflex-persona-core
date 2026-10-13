"""Account every registered policy; independently replay sealed world outcomes."""
import argparse,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from reflex.laboratory import profiles
from tools.learning_value_games import LearningGame


def summarize(path,*,replay=True):
    path=Path(path);pre=json.loads((path/'preregister.json').read_text());evaluation=json.loads((path/'evaluation.json').read_text())
    rows=[json.loads(s) for s in (path/'trajectories.jsonl').read_text().splitlines()]
    assert not evaluation['errors'] and len(rows)==len(pre['jobs'])==evaluation['complete']
    assert set(r['index'] for r in rows)==set(range(len(rows)))
    decisions=0;known_probes=0;probes=0;persona={};max_gap=0.
    for r in rows:
        case,profile,arm,seeds=pre['jobs'][r['index']]
        assert (r['case'],r['profile'],r['arm'])==(case,profile['id'],arm)
        assert [e['seed'] for e in r['executions']]==seeds
        assert -1.-1e-10<=r['value']<=r['oracle']+1e-10<=1.+2e-10
        persona.setdefault((digest(case),arm),set()).add(r['initial'])
        if arm=='adaptive':max_gap=max(max_gap,max(abs(t['gap']) for t in r['trace']))
        if not replay:continue
        g=LearningGame(case,profile)
        for e in r['executions']:
            s=g.start();h=g.names[0] if random.Random(e['seed']).random()<s.weights[0] else g.names[1]
            for row in e['steps']:
                assert digest(row['before'])==digest(g.kernel.record(s))
                key=row['action'];assert key in g.keys(s)
                if key in ('survey','probe','feint'):
                    probes+=1;known_probes+=int(max(s.weights)>=1.-1e-10 and case.get('flip',0.)==0.)
                outlaws=g.model(h,s.public,key);u=random.Random(int(digest(['learning-value-actual',e['seed'],s.tick])[:16],16)).random();mass=0.
                for out in outlaws:
                    mass+=out.probability
                    if u<mass:break
                assert digest(out.flow)==digest(row['flow']) and digest(out.public)==digest(row['after']['public'])
                s=g.kernel.reveal(s,key,out.public,out.flow);h=out.next_hypothesis
                assert digest(g.kernel.record(s))==digest(row['after']);decisions+=1
            assert g.terminal(s) and g.goal(s).record()==e['goal']
    groups=[]
    for game in ['all']+sorted({r['case']['game'] for r in rows}):
        for candidate in sorted({r.get('candidate','adaptive') for r in evaluation['pairs']}):
            for base in ('myopic','frozen'):
                ps=[p for p in evaluation['pairs'] if (game=='all' or p['case']['game']==game) and p['baseline']==base and p.get('candidate','adaptive')==candidate]
                groups.append(dict(game=game,candidate=candidate,baseline=base,pairs=len(ps),
                    better=sum(p['delta']>1e-10 for p in ps),worse=sum(p['delta']< -1e-10 for p in ps),same=sum(abs(p['delta'])<=1e-10 for p in ps),
                    average_delta=sum(p['delta'] for p in ps)/len(ps) if ps else None))
    return dict(study=path.name,policies=len(rows),executions=sum(len(r['executions']) for r in rows),replayed_decisions=decisions,
        decision_states=sum(r['states'] for r in rows),max_adaptive_prediction_gap=max_gap,groups=groups,
        information_actions=probes,fully_known_information_actions=known_probes,
        diverse_initial_conditions=sum(len(v)>1 for v in persona.values()),persona_condition_groups=len(persona),
        information_measure='observed action counts only; not an intrinsic information reward or proof of efficient exploration',
        rows_digest=digest(rows),registration_digest=digest(pre))


def main():
    p=argparse.ArgumentParser();p.add_argument('--studies',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=[summarize(path) for path in a.studies];out=Path(a.output);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(study=r['study'],policies=r['policies'],replayed=r['replayed_decisions'],groups=[g for g in r['groups'] if g['game']=='all']) for r in result]),flush=True)


if __name__=='__main__':main()
