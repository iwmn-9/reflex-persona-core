"""Evaluator-only Bellman/martingale accounting of complete auction returns.

For the true finite-horizon purpose value V*, write each chosen-action gap as
V*(s)-E[V*(s')|s,a], including accumulated score in V*. The realized residual is
V*(s')-E[V*(s')|s,a]. Thus final score = initial V* - sum(gaps) + sum(residuals).
Residuals have zero conditional mean under the true public event law, even for
history-dependent policies. Initial V* minus visited gaps is consequently an
unbiased purpose-return estimator across fresh world randomness for a fixed
policy. A single path or a small selected seed set does not prove superiority.

V* knows future regime changes and ignores personality; it is never passed to
the NPC. Its gap includes legitimate preference costs and unavailable knowledge.
This is a post-hoc diagnostic, not a replacement preregistered acceptance metric.
"""
import argparse, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from reflex.laboratory import PROFILES
from tools.compare_owned_studies import compare
from tools.continuing_games import ContinuingGame
from tools.diagnose_auction_continuity import reference
from tools.run_owned_value import law


def analyze(row):
    assert row['game']=='auction' and len(row['trace'])==row['turns']
    turns=row['turns'];g=ContinuingGame('auction',next(p for p in PROFILES if p['id']==row['profile']),turns,row['seed']%2)
    q=reference(turns,row['seed']%2,lambda t:tuple(law(t,turns,row['regime']).values()))
    def potential(w):
        future=.25*w['cash']/(8*turns) if w['tick']==turns else max(q(w['tick'],int(w['cash'])).values())
        return (w['score']-w['rival'])/(8*turns)+future
    initial=potential(row['trace'][0]['before']);steps=[]
    for t in row['trace']:
        before=t['before'];after=t['after'];key=t['action'];p=law(t['tick'],turns,row['regime'])
        assert p==t['truth'] and g.transition(before,key,t['response'])[0]==after
        values={z:potential(g.transition(before,key,z)[0]) for z in p}
        expected=math.fsum(p[z]*values[z] for z in p)
        current=potential(before);actual=potential(after)
        gap=current-expected;noise=actual-expected
        direct=max(q(t['tick'],int(before['cash'])).values())-q(t['tick'],int(before['cash']))[key]
        assert math.isclose(gap,direct,rel_tol=0,abs_tol=1e-12) and gap>=-1e-12
        assert abs(math.fsum(p[z]*(values[z]-expected) for z in p))<1e-12
        steps.append(dict(tick=t['tick'],purpose_opportunity_loss=gap,innovation=noise))
    loss=math.fsum(s['purpose_opportunity_loss'] for s in steps)
    noise=math.fsum(s['innovation'] for s in steps)
    final=potential(row['trace'][-1]['after'])
    assert math.isclose(final,row['score'],rel_tol=0,abs_tol=1e-12)
    error=abs(final-(initial-loss+noise));assert error<1e-12
    return dict(**{k:row[k] for k in ('game','profile','seed','regime','turns','tail_steps','offset')},
        arm=row['arm']+'@'+str(row['interval']),score=final,oracle_initial=initial,
        purpose_opportunity_loss=loss,innovation=noise,oracle_adjusted_return=initial-loss,
        accounting_error=error,steps=steps)


def main():
    p=argparse.ArgumentParser();p.add_argument('--studies',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    validated=compare(a.studies,[]);runs=[];groups=defaultdict(dict)
    rule_sha=hashlib.sha256((R/'tools/continuing_games.py').read_bytes()).hexdigest()
    for study in map(Path,a.studies):
        assert json.loads((study/'source_snapshot.json').read_text())['tools/continuing_games.py']==rule_sha
        for row in map(json.loads,(study/'trajectories.jsonl').read_text().splitlines()):
            if row['game']!='auction':continue
            r=analyze(row);key=tuple(r[k] for k in ('profile','seed','regime','turns','tail_steps','offset'))
            assert r['arm'] not in groups[key];groups[key][r['arm']]=r;runs.append(r)
    pairs=[]
    for key,arms in sorted(groups.items()):
        if 'receding_recovery@1' not in arms:continue
        new=arms['receding_recovery@1']
        for baseline,old in sorted(arms.items()):
            if baseline=='receding_recovery@1':continue
            assert old['oracle_initial']==new['oracle_initial']
            delta=new['score']-old['score'];estimated=new['oracle_adjusted_return']-old['oracle_adjusted_return']
            noise=new['innovation']-old['innovation']
            assert abs(delta-estimated-noise)<1e-12
            pairs.append(dict(zip(('profile','seed','regime','turns','tail_steps','offset'),key),baseline=baseline,
                score_delta=delta,oracle_adjusted_return_delta=estimated,innovation_delta=noise))
    result=dict(semantics=__doc__,sources=validated['sources'],rule_sha256=rule_sha,runs=runs,pairs=pairs,
        limitations=['Not a personality loss decomposition or a measure of accessible-information optimality.',
            'Only the declared action-independent auction law and known finite scoring horizon are evaluated.',
            'This does not validate a different game, unknown transition rules, or the NPC world model.',
            'Sampling uncertainty remains; related personae and comparator arms do not create independent world seeds.'])
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(runs=len(runs),pairs=len(pairs),max_accounting_error=max(r['accounting_error'] for r in runs))))


if __name__=='__main__':main()
