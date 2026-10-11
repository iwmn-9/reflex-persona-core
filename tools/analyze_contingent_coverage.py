"""Post-study audit of whole independent game cohorts; never select winning rows.

Raw traces/frozen trees stay private. Endpoint deltas are paired only within
the same game, case, owner and actual seed. Different goal scales are not pooled.
"""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex.core import digest
from reflex.contingent_games import BRIDGES

EXCLUDED={'laboratory','team_combat','mahjong_closed'}
COHORTS={
    'development-v2':('development',set()),
    'quoridor-development-v2':('development',set()),
    'quoridor-confirmation-v1':('confirmation',set()),
    'confirmation-v1':('confirmation',EXCLUDED),
    'social-development-v2':('development',set()),
    'endpoints-confirmation-v2':('confirmation',set()),
    'social-confirmation-v2':('confirmation',set()),
    'mahjong-development-v2':('development',set()),
    'mahjong-confirmation-v3':('confirmation',set()),
    'mechanics-development-v1':('development',set()),
    'mechanics-confirmation-v1':('confirmation',set()),
    'tiny-development-v1':('development',set()),
    'tiny-confirmation-v1':('confirmation',set()),
    'graph-development-v1':('development',set()),
    'graph-confirmation-v1':('confirmation',set()),
}
SUPPLEMENTAL=('development-v1','team-endpoints-development-v2','laboratory-development-v2',
    'social-development-v1','duel-validation-v1','mahjong-confirmation-v2')

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def analyze(root,out):
    out.mkdir(parents=True,exist_ok=True)
    registration=dict(format='conditional-analysis-v1',timing='post-study descriptive analysis; not a prospective statistical test',
        cohorts={k:dict(stage=v[0],exclude_whole_games=sorted(v[1])) for k,v in COHORTS.items()},
        supplemental=list(SUPPLEMENTAL),
        replacement='whole laboratory/team-combat/mahjong cohorts replaced; social-v1 replaced in full',
        comparisons='same case/profile/actual seed; goal-reflex vs conditional; no pooled goal mean or significance',
        required_games=sorted(set(BRIDGES)|{'observed_resources','observed_delivery','observed_auction','observed_combat','mahjong_closed'}),
        caveats=['deterministic repeated seeds are repeated executions, not independent environments',
                'not matched compute, not a comparison with legacy fixed-script search',
                'mechanics coverage does not mean all prior psychological and team-agreement pipelines connected'])
    path=out/'analysis_registration.json'
    if path.exists():assert read(path)==registration
    else:path.write_text(json.dumps(registration,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    accepted={};provenance=[];source_differences={};model_nodes=Counter();execution_checks=Counter();plan_continuity={}
    final=Path(__file__).resolve().parents[1]
    for study in (*COHORTS,*SUPPLEMENTAL):
        folder=root/study;pre=read(folder/'preregister.json');evaluation=read(folder/'evaluation.json')
        assert digest(pre)==evaluation['preregister_digest']
        assert pre['source_hashes']==evaluation['source_hashes']
        assert evaluation['episodes']+len(evaluation['errors'])==evaluation['planned']==len(pre['jobs'])
        for serial,resumption in enumerate(evaluation.get('resumptions',[]),1):
            prior=read(folder/f'evaluation_before_resume-{serial}.json')
            assert sha(folder/f'evaluation_before_resume-{serial}.json')==resumption['prior_evaluation_sha256']
            assert read(folder/f'resume_registration-{serial}.json')==resumption
            assert resumption['source_hashes']==pre['source_hashes']
            assert set(resumption['indices'])=={e['index'] for e in prior['errors']}
            prefix=hashlib.sha256()
            with (folder/'trajectories.jsonl').open('rb') as stream:
                for _ in range(prior['episodes']):prefix.update(next(stream))
            assert prefix.hexdigest()==resumption['prior_raw_sha256']
        normalized={(name if '/' in name else 'reflex/'+name):expected for name,expected in pre['source_hashes'].items()}
        for name,expected in normalized.items():assert sha(folder/'source'/name)==expected
        source_differences[study]=[name for name,expected in normalized.items() if not (final/name).is_file() or sha(final/name)!=expected]
        stage,excluded=COHORTS.get(study,('supplemental',set()))
        runs=[];all_runs=[];seen=set();trace_rows=0
        with (folder/'trajectories.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                r=json.loads(line);index=r['job_index'];assert index not in seen;seen.add(index)
                case,profile,seed,method,*_=pre['jobs'][index]
                assert (r['case'],r['profile'],r['seed'],r['method'])==(case,profile['id'],seed,method)
                assert r['steps']==len(r['trace']);trace_rows+=len(r['trace'])
                row={k:v for k,v in r.items() if k!='trace'};all_runs.append(row)
                if study not in COHORTS or case['game'] in excluded:continue
                assert not any(e['task'][0]['game']==case['game'] for e in evaluation['errors'])
                key=(stage,digest(case),profile['id'],seed,method)
                assert key not in accepted;accepted[key]=row;runs.append(row)
                generic=case['game'] in BRIDGES
                if generic:assert r['action_replay_verified'];execution_checks['replayed_generic_episodes']+=1
                execution_checks['accepted_episodes']+=1;execution_checks['accepted_engine_steps']+=r['steps']
                execution_checks['accepted_owner_steps']+=r.get('owner_decisions',r['steps'])
                previous_program=None
                for t in r['trace']:
                    if generic:
                        assert t['persona_hash']==r['trace'][0]['persona_hash'];execution_checks['fixed_owner_steps']+=1
                    a=t.get('conditional')
                    if a:
                        assert 0<=len(a['program']['entries'])<=128
                        assert a['legal_roots']==len(a['extra_slots'])
                        execution_checks['conditional_audited_steps']+=1
                        execution_checks['retained_node_count']+=len(a['program']['entries'])
                        if generic:
                            continuity=plan_continuity.setdefault(case['game'],Counter())
                            current_nodes={b['public_nodes'][0]['node'] for b in t['branches']}
                            assert len(current_nodes)==1
                            if previous_program:
                                continuity['nonempty_previous_program_steps']+=1
                                wanted=previous_program.get(next(iter(current_nodes)))
                                if wanted is not None:
                                    continuity['previous_program_matching_observation']+=1
                                    continuity['followed' if wanted==t['root'] else 'revised_after_new_search']+=1
                                else:continuity['previous_program_unmatched_observation']+=1
                            previous_program=dict(a['program']['entries'])
                    for branch in t.get('branches',[]):
                        for node in branch.get('public_nodes',[])[1:]:
                            assert node['action'] in node['allowed']
                            model_nodes['conditional' if node['supplied'] else 'reflex_fallback']+=1
        assert len(seen)==evaluation['episodes'] and sorted(all_runs,key=lambda r:r['job_index'])==evaluation['runs']
        # Complete aggregates retain all errors, including excluded cohorts.
        target=out/study;target.mkdir(exist_ok=True)
        for name in ('preregister.json','evaluation.json'):(target/name).write_bytes((folder/name).read_bytes())
        provenance.append(dict(study=study,stage=stage,accepted_episodes=len(runs),completed_episodes=evaluation['episodes'],
            planned=evaluation['planned'],errors=len(evaluation['errors']),trace_rows=trace_rows,
            excluded_whole_games=sorted(excluded),supplemental=study in SUPPLEMENTAL,
            raw_sha256=sha(folder/'trajectories.jsonl'),source_hashes=evaluation['source_hashes'],
            evidence_sha256={n:sha(folder/n) for n in ('preregister.json','evaluation.json')}))
        provenance[-1]['frozen_tools_sha256']={p.name:sha(p) for p in sorted((folder/'source/tools').glob('*.py'))}
    comparisons=[]
    for stage,case_hash,profile,seed,method in sorted(accepted):
        if method!='goal-reflex':continue
        a=accepted[(stage,case_hash,profile,seed,method)];b=accepted[(stage,case_hash,profile,seed,'conditional')]
        if a['case']['game']=='mahjong_closed':av=a['point_delta'];bv=b['point_delta'];unit='actual points; own-hand forecast proxy'
        else:av=a['goal']['value'];bv=b['goal']['value'];unit=a['goal']['target']
        comparisons.append(dict(stage=stage,case=a['case'],profile=profile,seed=seed,
            baseline=av,conditional=bv,delta=bv-av,unit=unit,
            baseline_status=a.get('goal',{}).get('status'),conditional_status=b.get('goal',{}).get('status'),
            baseline_steps=a['steps'],conditional_steps=b['steps'],
            actions_changed=a.get('actions')!=b.get('actions') if 'actions' in a and 'actions' in b else None))
    assert len(comparisons)*2==len(accepted)
    coverage=[];totals=Counter()
    for game in registration['required_games']:
        group=[r for r in comparisons if r['case']['game']==game]
        assert {r['profile'] for r in group if r['stage']=='development'}=={'growth','steady','care','ego'},game
        by_profile=[]
        for profile in ('growth','steady','care','ego'):
            rows=[r for r in group if r['profile']==profile]
            assert Counter(r['stage'] for r in rows)=={'development':1,'confirmation':2},(game,profile)
            counts=Counter('better' if r['delta']>1e-12 else 'worse' if r['delta']<-1e-12 else 'same' for r in rows)
            by_profile.append(dict(profile=profile,pairs=len(rows),**{k:counts[k] for k in ('better','same','worse')}))
            totals.update(counts)
        coverage.append(dict(game=game,pairs=len(group),development_cases=len({digest(r['case']) for r in group if r['stage']=='development'}),
            confirmation_cases=len({digest(r['case']) for r in group if r['stage']=='confirmation'}),profiles=by_profile))
    result=dict(format='conditional-coverage-analysis-v1',protocols=len(coverage),primary_episodes=len(accepted),pairs=len(comparisons),
        counts=dict(totals),coverage=coverage,comparisons=comparisons,execution_checks=dict(execution_checks),model_nodes=dict(model_nodes),
        source_differences=source_differences,
        execution_steps_semantics='sum of recorded trace transitions: owner-decision intervals in generic bridges and driver transitions in other protocols; not every other-player engine action',
        plan_continuity={k:dict(v) for k,v in plan_continuity.items()},
        plan_continuity_semantics='actual next observed node compared with prior selected finite program before fresh search; only generic bridge conditional episodes; unmatched does not mean actual action used reflex',
        default_policy_changed=False,cloud_runtime_used=False)
    (out/'coverage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps(dict(studies=provenance,analyzer_sha256=sha(Path(__file__))),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('protocols','primary_episodes','pairs','counts','execution_checks','model_nodes')},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    analyze(Path(args.root).resolve(),Path(args.output).resolve())
