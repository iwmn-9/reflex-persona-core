"""Summarize complete matched long executions; no selection on their outcome."""
import argparse, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    studies = sorted(p.parent for p in root.glob('*/registration.json'))
    assert len(studies) == 32, '24 comparison shards and 8 stress shards required'
    rows, checks, sources = [], Counter(), []
    for study in studies:
        audit = json.loads((study/'audit.json').read_text())
        assert audit['complete'] and audit['source_freeze_verified']
        assert audit['trajectory_sha256'] == sha(study/'trajectories.jsonl')
        manifest = json.loads((study/'source_snapshot.json').read_text())
        assert all(sha(study/'_source'/name) == digest for name, digest in manifest.items())
        checks.update(audit['checks'])
        part = [json.loads(line) for line in (study/'trajectories.jsonl').read_text().splitlines()]
        assert len(part) == audit['completed_runs'] == audit['planned_runs']
        rows.extend(part)
        sources.append(dict(study=study.name, trajectory_sha256=audit['trajectory_sha256'],
                            source_manifest_sha256=sha(study/'source_snapshot.json')))
    grouped = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ('game', 'profile', 'seed', 'regime', 'turns'))
        assert row['arm'] not in grouped[key]
        grouped[key][row['arm']] = row
    expected = {(g, p, seed, regime, 128)
        for g in ('auction','resources','combat','gambling')
        for p in ('growth','steady','care','ego') for seed in (401,547,548)
        for regime in ('outside','shift-return')}
    expected |= {(g,p,614,'irregular',256)
        for g in ('auction','resources','combat','gambling') for p in ('growth','steady')}
    assert set(grouped) == expected
    pairs, descriptions = [], []
    for key, arms in sorted(grouped.items()):
        assert set(arms) == {'endpoint','reflex','endpoint_posterior','endpoint_persistent'}
        baseline = arms['endpoint']
        for row in arms.values():
            assert all(row[k] == baseline[k] for k in ('interval','tail_steps','offset'))
            overlap = min(row['executed'], baseline['executed'])
            assert [t['response'] for t in row['trace'][:overlap]] == [t['response'] for t in baseline['trace'][:overlap]]
            ts = row['trace']; n = len(ts)
            switches = [i for i in range(1,n) if ts[i]['truth'] != ts[i-1]['truth']]
            phases = []
            for start, stop in zip([0,*switches], [*switches,n]):
                if stop-start < 16: continue
                def error(begin,end):
                    return sum(t['brier'] for t in ts[begin:end])/(end-begin)
                def gain(begin,end):
                    before = ts[begin-1]['goal'] if begin else None
                    # Initial game goals differ; use observed boundaries for
                    # post-change gains and leave the initial gain unspecified.
                    return None if before is None else ts[end-1]['goal']-before
                phases.append(dict(start=start, stop=stop,
                    first_error=error(start,start+16),last_error=error(stop-16,stop),
                    first_gain=gain(start,start+16),last_gain=gain(stop-16,stop)))
            descriptions.append({**{k:row[k] for k in ('game','profile','seed','regime','turns','arm','score','executed')},
                'brier':sum(t['brier'] for t in ts)/n,
                'actions':dict(Counter(t['action'] for t in ts)), 'phases':phases})
        for name in ('reflex','endpoint_posterior','endpoint_persistent'):
            row = arms[name]
            for reference in ('endpoint','reflex'):
                if name == reference: continue
                delta = row['score']-arms[reference]['score']
                pairs.append(dict(zip(('game','profile','seed','regime','turns'),key),candidate=name,
                    baseline=reference,delta=delta,better=int(delta>1e-12),same=int(abs(delta)<=1e-12),worse=int(delta< -1e-12)))
    summaries = []
    for candidate,baseline,turns in sorted({(p['candidate'],p['baseline'],p['turns']) for p in pairs}):
        ps = [p for p in pairs if (p['candidate'],p['baseline'],p['turns']) == (candidate,baseline,turns)]
        for field in ('game','profile','regime','seed'):
            for value in sorted({p[field] for p in ps}):
                group = [p for p in ps if p[field] == value]
                summaries.append(dict(candidate=candidate,baseline=baseline,turns=turns,group_by=field,group=value,
                    pairs=len(group), **{k:sum(p[k] for p in group) for k in ('better','same','worse')},
                    mean_delta=sum(p['delta'] for p in group)/len(group),worst=min(p['delta'] for p in group)))
    initial_choices=defaultdict(dict)
    for row in rows:
        if row['arm']=='endpoint_posterior' and row['turns']==128 and row['regime']=='outside':
            initial_choices[(row['game'],row['seed'])][row['profile']]=row['trace'][0]['action']
    result = dict(runs=len(rows),conditions=len(grouped),executed=sum(r['executed'] for r in rows),
        complete=True,checks=dict(checks),sources=sources,pairs=pairs,summaries=summaries,runs_detail=descriptions,
        same_initial_world_choices=[dict(game=k[0],seed=k[1],actions=v) for k,v in sorted(initial_choices.items())],
        caveats=['Four synthetic rule families, not human play or all games.',
                 'Conditions share seeds; counts are not independent statistical trials or win rates.',
                 'Cross-game scores have different meanings; pooled averages do not establish general strength.',
                 'Posterior was selected before its first confirmation; persistent is an exploratory contrast.',
                 'Endpoint forecast error includes subsequent factual replanning; it is not a pure model error.',
                 'Brier fields are squared distances to evaluator-only true probabilities, not realized categorical Brier scores.',
                 'Baseline and candidates run on the same Linux platform in every pair.'])
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args()
    result=summarize(Path(args.root))
    print(json.dumps({k:result[k] for k in ('runs','conditions','executed','checks')},indent=2))
    print(json.dumps([r for r in result['summaries'] if r['baseline']=='endpoint' and r['group_by']=='game'],indent=2))
