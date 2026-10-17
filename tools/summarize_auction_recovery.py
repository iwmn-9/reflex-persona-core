"""Summarize complete matched long executions; no selection on their outcome."""
import argparse, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ablation(root, reference):
    audit=json.loads((root/'audit.json').read_text())
    assert audit['complete'] and audit['completed_runs']==32 and audit['source_freeze_verified']
    assert sha(root/'trajectories.jsonl')==audit['trajectory_sha256']
    source=json.loads((root/'source_snapshot.json').read_text())
    assert all(sha(root/'_source'/name)==value for name,value in source.items())
    reference_evaluation=json.loads((reference/'evaluation.json').read_text())
    assert reference_evaluation['complete'] and reference_evaluation['runs']==416
    baselines={}
    for study in sorted(p.parent for p in reference.glob('*/registration.json')):
        check=json.loads((study/'audit.json').read_text())
        assert check['complete'] and sha(study/'trajectories.jsonl')==check['trajectory_sha256']
        for row in map(json.loads,(study/'trajectories.jsonl').read_text().splitlines()):
            if row['seed']==401 and row['turns']==128 and row['arm'] in ('endpoint','endpoint_posterior'):
                key=tuple(row[k] for k in ('game','profile','regime','arm'))
                assert key not in baselines;baselines[key]=row
    rows=list(map(json.loads,(root/'trajectories.jsonl').read_text().splitlines()))
    keys={(row['game'],row['profile'],row['regime']) for row in rows}
    assert len(rows)==len(keys)==32
    assert keys=={(g,p,r) for g in ('auction','resources','combat','gambling') for p in ('growth','steady','care','ego') for r in ('outside','shift-return')}
    pairs=[]
    for row in rows:
        assert row['arm']=='endpoint_unpruned' and row['seed']==401 and row['turns']==128
        for arm in ('endpoint','endpoint_posterior'):
            base=baselines[row['game'],row['profile'],row['regime'],arm]
            assert all(row[k]==base[k] for k in ('interval','tail_steps','offset'))
            n=min(row['executed'],base['executed'])
            assert [t['response'] for t in row['trace'][:n]]==[t['response'] for t in base['trace'][:n]]
            delta=row['score']-base['score']
            pairs.append(dict(game=row['game'],profile=row['profile'],regime=row['regime'],baseline=arm,
                delta=delta,better=int(delta>1e-12),same=int(abs(delta)<=1e-12),worse=int(delta< -1e-12)))
    groups=[]
    for arm in ('endpoint','endpoint_posterior'):
        for game in ('auction','resources','combat','gambling'):
            ps=[p for p in pairs if p['baseline']==arm and p['game']==game]
            groups.append(dict(baseline=arm,game=game,pairs=len(ps),
                **{k:sum(p[k] for p in ps) for k in ('better','same','worse')},
                mean_delta=sum(p['delta'] for p in ps)/len(ps),worst=min(p['delta'] for p in ps)))
    result=dict(complete=True,runs=len(rows),conditions=len(rows),executed=sum(r['executed'] for r in rows),
        checks=audit['checks'],pairs=pairs,summaries=groups,
        reference_evaluation_sha256=sha(reference/'evaluation.json'),
        caveat='Post-hoc development on previously examined seed 401, not independent confirmation. The ablation runs on Windows and references matched Linux study rows; separate platform replay checks are descriptive, not a proof of universal floating-point equivalence.')
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result


def replication(root):
    import numpy as np
    studies=sorted(p.parent for p in root.glob('*/registration.json'))
    assert len(studies)==16
    rows=[];checks=Counter();sources=[]
    for study in studies:
        audit=json.loads((study/'audit.json').read_text())
        assert audit['complete'] and audit['source_freeze_verified'] and audit['completed_runs']==16
        assert audit['trajectory_sha256']==sha(study/'trajectories.jsonl')
        manifest=json.loads((study/'source_snapshot.json').read_text())
        assert all(sha(study/'_source'/name)==digest for name,digest in manifest.items())
        rows.extend(json.loads(line) for line in (study/'trajectories.jsonl').read_text().splitlines())
        checks.update(audit['checks']);sources.append(dict(study=study.name,trajectory_sha256=audit['trajectory_sha256']))
    grouped=defaultdict(dict)
    for row in rows:
        assert row['game']=='auction' and row['turns']==128 and row['interval']==7 and row['tail_steps']==15 and row['offset']==0
        key=(row['seed'],row['profile'],row['regime'])
        assert row['arm'] not in grouped[key];grouped[key][row['arm']]=row
    assert set(grouped)=={(s,p,r) for s in range(21001,21017) for p in ('growth','steady','care','ego') for r in ('outside','shift-return')}
    pairs=[]
    for key,arms in sorted(grouped.items()):
        assert set(arms)=={'endpoint','endpoint_posterior'}
        base,new=arms['endpoint'],arms['endpoint_posterior']
        assert [t['response'] for t in base['trace']]==[t['response'] for t in new['trace']]
        d=new['score']-base['score']
        pairs.append(dict(seed=key[0],profile=key[1],regime=key[2],delta=d,
            better=int(d>1e-12),same=int(abs(d)<=1e-12),worse=int(d< -1e-12)))
    summaries=[]
    groups=[('all','all',pairs)]
    for field in ('profile','regime','seed'):
        groups += [(field,value,[p for p in pairs if p[field]==value]) for value in sorted({p[field] for p in pairs})]
    for field,value,ps in groups:
        means=np.array([np.mean([p['delta'] for p in ps if p['seed']==seed]) for seed in sorted({p['seed'] for p in ps})])
        # Keep all personas/regimes sharing a world seed in the same block.
        interval=None
        if len(means)>1:
            rng=np.random.default_rng(73019)
            boot=means[rng.integers(0,len(means),size=(20000,len(means)))].mean(axis=1)
            interval=np.quantile(boot,[.025,.975]).tolist()
        summaries.append(dict(group_by=field,group=value,pairs=len(ps),world_seeds=len(means),
            **{k:sum(p[k] for p in ps) for k in ('better','same','worse')},
            mean_delta=float(np.mean([p['delta'] for p in ps])),worst=min(p['delta'] for p in ps),
            seed_cluster_percentile_95=interval,positive_seed_means=int(sum(means>1e-12)),negative_seed_means=int(sum(means< -1e-12))))
    result=dict(complete=True,runs=len(rows),conditions=len(pairs),executed=sum(r['executed'] for r in rows),
        checks=dict(checks),sources=sources,pairs=pairs,summaries=summaries,
        caveat='Descriptive percentile bootstrap with world seed as the cluster (16 seeds), conditional on these two synthetic response regimes and auction variants. Not a win-rate or cross-game generalization guarantee; no candidate or persona coefficients were fitted on these outcomes.')
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result


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
        registration=json.loads((study/'registration.json').read_text())
        planned=len(registration['jobs'])*len(registration['arms'])
        assert len(part)==audit['completed_runs']==planned
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
    motion=[]
    for row in rows:
        # One representative per actual response sequence: avoid counting the
        # same world stream again for every persona and game.
        if row['game']!='auction' or row['profile']!='growth' or row['turns']!=128 or row['arm'] not in ('endpoint','endpoint_posterior'):continue
        ts=row['trace'];changes=[i for i in range(1,len(ts)) if ts[i]['truth']!=ts[i-1]['truth']]
        adapting={i for start in changes for i in range(start,min(start+16,len(ts)))}
        later=set(range(16,len(ts)))-adapting
        def average(values):return sum(values)/len(values) if values else None
        motion.append(dict(seed=row['seed'],regime=row['regime'],arm=row['arm'],
            mse=average([t['brier'] for t in ts]),
            postchange_first16_mse=average([ts[i]['brier'] for i in sorted(adapting)]),
            other_after_initial16_mse=average([ts[i]['brier'] for i in sorted(later)]),
            other_after_initial16_movement=average([sum(abs(ts[i]['law'][k]-ts[i-1]['law'][k]) for k in ts[i]['law'])/2 for i in sorted(later)])))
    result = dict(runs=len(rows),conditions=len(grouped),executed=sum(r['executed'] for r in rows),
        complete=True,checks=dict(checks),sources=sources,pairs=pairs,summaries=summaries,runs_detail=descriptions,prediction_motion=motion,
        same_initial_world_choices=[dict(game=k[0],seed=k[1],actions=v) for k,v in sorted(initial_choices.items())],
        caveats=['Four synthetic rule families, not human play or all games.',
                 'Conditions share seeds; counts are not independent statistical trials or win rates.',
                 'Cross-game scores have different meanings; pooled averages do not establish general strength.',
                 'Posterior was selected before its first confirmation; persistent is an exploratory contrast.',
                 'Endpoint forecast error includes subsequent factual replanning; it is not a pure model error.',
                 'Brier fields are squared distances to evaluator-only true probabilities, not realized categorical Brier scores.',
                 'Historical audit planned_runs accidentally contains the last run forecast rows. Registered counts are recomputed independently here; original audits are retained. The audit exporter is corrected in final source.',
                 'Baseline and candidates run on the same Linux platform in every pair.'])
    (root/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root')
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--replication',action='store_true');mode.add_argument('--ablation',action='store_true')
    parser.add_argument('--reference');args=parser.parse_args()
    if args.ablation and not args.reference:parser.error('ablation requires the complete primary reference')
    result=ablation(Path(args.root),Path(args.reference)) if args.ablation else (replication if args.replication else summarize)(Path(args.root))
    print(json.dumps({k:result[k] for k in ('runs','conditions','executed','checks')},indent=2))
    print(json.dumps(result['summaries'] if args.ablation else [r for r in result['summaries'] if r['group_by'] in ('all','profile')] if args.replication else
        [r for r in result['summaries'] if r['baseline']=='endpoint' and r['group_by']=='game'],indent=2))
