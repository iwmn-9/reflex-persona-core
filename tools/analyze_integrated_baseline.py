"""Audit complete frozen cohorts and publish comparable, genre-specific results."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reflex.core import digest
from tools.run_integrated_baseline import GAMES

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def tally(xs):return dict(better=sum(x>1e-12 for x in xs),worse=sum(x<-1e-12 for x in xs),same=sum(abs(x)<=1e-12 for x in xs),mean_delta=sum(xs)/len(xs) if xs else None)

def validate(root):
    pre=read(root/'preregister.json');records={};receipts={}
    for cohort,jobs in pre['jobs'].items():
        result=read(root/(cohort+'-result.json'));assert not result['errors'] and result['episodes']==result['planned']==len(jobs)
        fixed=pre['source_hashes']['legacy' if cohort=='legacy' else 'current'];assert result['source_hashes']==fixed and result['preregister_sha256']==digest(pre)
        frozen=root/('legacy' if cohort=='legacy' else 'current')
        assert all(sha(frozen/name)==value for name,value in fixed.items())
        rows=[json.loads(s) for s in (root/(cohort+'.jsonl')).read_text(encoding='utf-8').splitlines()];assert len(rows)==len(jobs)
        assert set(r['index'] for r in rows)==set(range(len(jobs)))
        by_index={r['index']:r for r in result['runs']}
        for r in rows:
            assert r['job']==jobs[r['index']] and r['action_replay_verified'] is True
            assert {k:v for k,v in r.items() if k!='trace'}==by_index[r['index']]
        records[cohort]=rows;receipts[cohort]=dict(episodes=len(rows),trajectories_sha256=sha(root/(cohort+'.jsonl')),result_sha256=sha(root/(cohort+'-result.json')))
    return pre,records,receipts

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--contracts',required=True);p.add_argument('--old-contracts',required=True);p.add_argument('--parity',required=True);p.add_argument('--projection',required=True);p.add_argument('--horizons',required=True);p.add_argument('--banks',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    base=Path(a.root).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    pre,records,receipts=validate(base);evaluation=read(base/'evaluation.json');assert evaluation['game_episodes']==768 and evaluation['life_worlds']==48 and evaluation['life_actor_episodes']==288
    pair=evaluation['paired'];assert len(pair)==256
    persona=[]
    for game in GAMES:
        for profile in sorted({r['profile'] for r in pair}):
            ps=[r for r in pair if r['game']==game and r['profile']==profile]
            persona.append(dict(game=game,profile=profile,pairs=len(ps),returns=tally([r['return_delta'] for r in ps]),owned=tally([r['owned_delta'] for r in ps])))
    grouped={};life_metrics=[]
    for r in records['life']:
        # Count mutual pairs once, distinguish participation from whole worlds.
        rows=[row for t in r['trace'] for row in t['rows']]
        counts=dict(mutual_cooperation=sum(row['matched'] and row['kind']==row['other_kind']=='cooperate' for row in rows)//2,
            mutual_demands=sum(row['matched'] and row['kind']==row['other_kind']=='demand' for row in rows)//2,
            unilateral_cooperation=sum(row['kind']=='cooperate' and not (row['matched'] and row['other_kind']=='cooperate') for row in rows))
        life_metrics.append(dict(domain=r['domain'],layout=r['layout'],seed=r['seed'],arm=r['arm'],**counts))
        for npc in r['actors']:
            key=(r['domain'],r['layout'],r['seed'],npc['npc']);grouped.setdefault(key,{})[r['arm']]=dict(npc,chosen=[t['chosen'][npc['npc']] for t in r['trace']])
    life_pairs=[]
    for key,group in grouped.items():
        assert set(group)=={'reflex','returns','owned'}
        rr,rt,ro=(group[k] for k in ('reflex','returns','owned'))
        life_pairs.append(dict(domain=key[0],layout=key[1],seed=key[2],npc=key[3],profile=rr['profile'],
            reflex=rr['goal']['value'],returns=rt['goal']['value'],owned=ro['goal']['value'],
            returns_delta=rt['goal']['value']-rr['goal']['value'],owned_delta=ro['goal']['value']-rt['goal']['value'],
            returns_actions_changed=rr['chosen']!=rt['chosen'],owned_actions_changed=rt['chosen']!=ro['chosen']))
    assert len(life_pairs)==96
    life_summary=[]
    for domain in ('exchange','watch'):
        for layout in ('balanced4','balanced8'):
            ps=[r for r in life_pairs if r['domain']==domain and r['layout']==layout]
            life_summary.append(dict(domain=domain,layout=layout,npc_pairs=len(ps),returns=tally([r['returns_delta'] for r in ps]),owned=tally([r['owned_delta'] for r in ps])))
    contracts=Path(a.contracts).resolve();diagnostic=read(contracts/'evaluation.json');assert diagnostic['schema_starts']==120 and diagnostic['schema_connections']==30 and diagnostic['route_episodes']==48
    previous={digest([r['profile'],r['seed'],r['route']]):r for r in (json.loads(s) for s in (Path(a.old_contracts)/'routes.jsonl').read_text(encoding='utf-8').splitlines())}
    # Re-evaluate the historical monitor observations with the current monitor.
    # The post-freeze late-deadline guard must not alter valid observed windows.
    from reflex.execution_window import Directive,ExecutionWindow
    from reflex.resource_world import ROUTES,world_from_record,achieved,winners,terminal
    from reflex.route_experiment import routed_context
    from reflex.laboratory import profiles
    monitor_replays=0
    for text in (contracts/'routes.jsonl').read_text(encoding='utf-8').splitlines():
        run=json.loads(text);profile=next(p for p in profiles() if p['id']==run['profile']);first=next(t for t in run['trace'] if t['own'])
        old=previous[digest([run['profile'],run['seed'],run['route']])]
        assert all(run[k]==old[k] for k in ('actions','won','achieved','actual_route_satisfied'))
        assert [{k:v for k,v in t.items() if k not in ('execution','global_tick')} for t in run['trace']]==[{k:v for k,v in t.items() if k!='execution'} for t in old['trace']]
        c,_=routed_context(world_from_record(first['before']),profile,42,0,None,run['route']);scope=c['scope']
        window=ExecutionWindow(scope,tuple(Directive(**d) for d in run['execution']['directives']))
        for t in run['trace']:
            before=world_from_record(t['before']);after=world_from_record(t['after']);tick=t['global_tick']
            if t['own']:window.issue(scope,run['route'],tick,opportunity=True)
            if t['execution'] is None:continue
            met=run['route'] in achieved(after.empires[run['seat']])
            opportunity=t['own'] or after.empires[run['seat']]!=before.empires[run['seat']] or met or terminal(after)
            audit=window.observe(scope,tick,opportunity=opportunity,met=met,failed=terminal(after) and not met)
            assert audit==t['execution']
        assert window.record()==run['execution'];monitor_replays+=1
    parity=read(Path(a.parity)/'result.json');assert parity['records']==1200 and not parity['mismatches']
    differences={}
    for name,old_sha in pre['source_hashes']['current'].items():
        now=ROOT/name
        if now.exists() and sha(now)!=old_sha:differences[name]=dict(frozen_sha256=old_sha,current_sha256=sha(now))
    core_differences={k for k in differences if k.startswith('reflex/')}
    assert core_differences=={'reflex/flow_rollout.py','reflex/loop_predictor.py','reflex/scaling.py','reflex/execution_window.py','reflex/purpose_plan.py','reflex/actor_model.py','reflex/decision_loop.py'}
    def without_docs(path):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if hasattr(node,'body') and isinstance(node.body,list) and node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
        return ast.dump(tree,include_attributes=False)
    assert without_docs(base/'current/reflex/flow_rollout.py')==without_docs(ROOT/'reflex/flow_rollout.py')
    old_goal=ast.parse((base/'current/reflex/purpose_plan.py').read_text(encoding='utf-8'))
    new_goal=ast.parse((ROOT/'reflex/purpose_plan.py').read_text(encoding='utf-8'))
    new_goal.body=[n for n in new_goal.body if not (isinstance(n,ast.ImportFrom) and n.module=='planning' and [x.name for x in n.names]==['vector'])]
    messages={'unknown truncated future cannot be padded as a terminal','no reward or preference flow after modeled settlement'}
    removed=0
    for node in ast.walk(new_goal):
        if hasattr(node,'body') and isinstance(node.body,list):
            def new_guard(n):
                return isinstance(n,ast.If) and len(n.body)==1 and isinstance(n.body[0],ast.Raise) and isinstance(n.body[0].exc,ast.Call) and n.body[0].exc.args and isinstance(n.body[0].exc.args[0],ast.Constant) and n.body[0].exc.args[0].value in messages
            filtered=[n for n in node.body if not new_guard(n)];removed+=len(node.body)-len(filtered);node.body=filtered
    assert removed==2 and ast.dump(old_goal,include_attributes=False)==ast.dump(new_goal,include_attributes=False)
    old_loop=ast.parse((base/'current/reflex/decision_loop.py').read_text(encoding='utf-8'))
    new_loop=ast.parse((ROOT/'reflex/decision_loop.py').read_text(encoding='utf-8'))
    method=next(n for cls in new_loop.body if isinstance(cls,ast.ClassDef) and cls.name=='DecisionLoop' for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='decide_batch')
    assert [n.arg for n in method.args.kwonlyargs]==['_counterfactual_roots'];method.args.kwonlyargs=[];method.args.kw_defaults=[];removed=0
    for node in ast.walk(method):
        if hasattr(node,'body') and isinstance(node.body,list):
            def extra(n):
                return isinstance(n,ast.If) and (any(isinstance(x,ast.Name) and x.id=='_counterfactual_roots' for x in ast.walk(n.test)) or any(isinstance(x,ast.Constant) and x.value=='counterfactual' for x in ast.walk(n.test)))
            filtered=[n for n in node.body if not extra(n)];removed+=len(node.body)-len(filtered);node.body=filtered
    assert removed==3 and ast.dump(old_loop,include_attributes=False)==ast.dump(new_loop,include_attributes=False)
    for cohort in ('legacy','current','life'):
        for run in records[cohort]:
            for row in run['trace']:
                traces=row.get('modeled',[])
                if isinstance(traces,dict):traces=[t for ts in traces.values() for t in ts]
                for trace in traces:
                    assert len(trace['actions'])==2 or trace['terminal']
    projection=read(Path(a.projection)/'result.json');assert projection['checks']==2528 and projection['connections']==30
    hr=Path(a.horizons);horizons=read(hr/'evaluation.json');hp=read(hr/'preregister.json')
    assert horizons['episodes']==horizons['planned']==288 and len(horizons['paired'])==96 and not horizons['errors']
    assert horizons['source_hashes']==hp['source_hashes'] and horizons['preregister_sha256']==digest(hp)
    assert all(sha(hr/'source'/name)==value for name,value in hp['source_hashes'].items())
    hrows=[json.loads(s) for s in (hr/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()];assert len(hrows)==288 and {r['index'] for r in hrows}==set(range(288))
    main={digest([r['case'],r['profile'],r['seed'],r['job']['stage']]):r for r in records['current'] if r['arm']=='returns'};h2_replays=0
    for r in hrows:
        assert r['action_replay_verified']
        if r['horizon']==2:
            old=main[digest([r['case'],r['profile'],r['seed'],r['stage']])]
            assert r['actions']==old['actions'] and r['goal']==old['goal'] and r['trace']==old['trace'];h2_replays+=1
    assert h2_replays==96
    horizon_summary=[]
    for game in ('hagetaka','auction_contest','gambling'):
        ps=[r for r in horizons['paired'] if r['game']==game];rs=[r for r in horizons['runs'] if r['case']['game']==game]
        horizon_summary.append(dict(game=game,pairs=len(ps),h4=tally([r['delta4'] for r in ps]),h8=tally([r['delta8'] for r in ps]),
            settlements={str(h):dict(terminal=sum(r['selected_terminal_branches'] for r in rs if r['horizon']==h),branches=sum(r['selected_model_branches'] for r in rs if r['horizon']==h)) for h in (2,4,8)}))
    br=Path(a.banks);banks=read(br/'evaluation.json');bp=read(br/'preregister.json')
    assert banks['planned']==banks['episodes']==192 and len(banks['paired'])==96 and not banks['errors']
    assert banks['source_hashes']==bp['source_hashes'] and banks['preregister_sha256']==digest(bp)
    assert all(sha(br/'source'/name)==value for name,value in bp['source_hashes'].items())
    assert all(sha(ROOT/name)==value for name,value in bp['source_hashes'].items() if name.startswith('reflex/'))
    brows=[json.loads(s) for s in (br/'trajectories.jsonl').read_text(encoding='utf-8').splitlines()];assert len(brows)==192 and {r['index'] for r in brows}==set(range(192))
    h8={digest([r['case'],r['profile'],r['seed'],r['stage']]):r for r in hrows if r['horizon']==8};bank2_replays=0
    for r in brows:
        assert r['action_replay_verified']
        if r['bank']==2:
            old=h8[digest([r['case'],r['profile'],r['seed'],r['stage']])]
            assert r['actions']==old['actions'] and r['goal']==old['goal'] and r['trace']==old['trace'];bank2_replays+=1
    assert bank2_replays==96
    bank_summary=[]
    for game in ('hagetaka','auction_contest','gambling'):
        ps=[r for r in banks['paired'] if r['game']==game]
        expanded=[r for r in banks['runs'] if r['case']['game']==game and r['bank']==8]
        combined=[r['goal']['value']-main[digest([r['case'],r['profile'],r['seed'],r['stage']])]['goal']['value'] for r in expanded]
        bank_summary.append(dict(game=game,pairs=len(ps),expanded_bank_vs_h8_bank2=tally([r['delta'] for r in ps]),h8_bank8_vs_h2_bank2=tally(combined)))
    provenance=dict(format='integrated-baseline-provenance-v1',receipts=receipts,differences=differences,analyzer_sha256=sha(Path(__file__)),
        core_difference_meaning={'reflex/flow_rollout.py':'documentation only, executable AST equality verified',
            'reflex/scaling.py':'social/purpose forwarding in TiledPolicy; these cohorts use ordinary Policy',
            'reflex/loop_predictor.py':'optional belief_options forwarding and explicit retention checkpoint; these cohorts configure no predictor',
            'reflex/execution_window.py':'reject backward/expired issue and late success; no main game uses this monitor, 48 diagnostic streams replay identically',
            'reflex/purpose_plan.py':'only two invalid-tail guards added, AST remainder equality verified; rollout pads terminal states with zero effects, never unknown cutoffs',
            'reflex/decision_loop.py':'detached counterfactual root override only; AST equality after removing three guarded additions and private parameter verified',
            'reflex/actor_model.py':'forced-root projection now supports active progress guards; 2528 old/new resolved checkpoints match across 30 public start connections, including social and immediate learning'},
        monitor_replays=monitor_replays,contract_trajectories_sha256=sha(contracts/'routes.jsonl'),contract_source_hashes=diagnostic['source_hashes'],
        monitor_correction='all 48 original physical trajectories/actions/endpoints identical; monitor now observes passive and terminal public events; v1 retained privately and excluded from valid totals',
        horizon_h2_replays=h2_replays,bank2_horizon8_replays=bank2_replays,bank_trajectories_sha256=sha(br/'trajectories.jsonl'),bank_source_hashes=bp['source_hashes'],
        horizon_trajectories_sha256=sha(hr/'trajectories.jsonl'),horizon_source_hashes=hp['source_hashes'],raw_data_public=False,cloud_runtime_used=False)
    diversity=[]
    for game in GAMES:
        for arm in ('legacy','returns','owned'):
            rs=[r for cohort in ('legacy','current') for r in records[cohort] if r['case']['game']==game and r['arm']==arm]
            diversity.append(dict(game=game,arm=arm,episodes=len(rs),distinct_case_action_streams=len({digest([r['case'],r['actions']]) for r in rs}),
                terminal_statuses={s:sum(r['goal']['status']==s for r in rs) for s in sorted({r['goal']['status'] for r in rs})}))
    summary=dict(format=pre['format'],game_episodes=768,game_pairs=256,life_worlds=48,life_actor_episodes=288,life_npc_pairs=96,
        summary=evaluation['summary'],persona=persona,life_summary=life_summary,life_metrics=life_metrics,
        diagnostic=dict(schema_starts=120,schema_connections=30,route_episodes=48,causal_states=diagnostic['causal_states'],causal_changed=diagnostic['causal_changed'],
            route_wins=sum(r['won'] for r in diagnostic['routes']),route_criterion_met=sum(r['actual_route_satisfied'] for r in diagnostic['routes']),
            execution_successes=sum(sum(c['success'] for c in r['execution']['counts'].values()) for r in diagnostic['routes'])),
        learning=[dict(game=g,entries=sum(r['learned_entries'] for r in records['current'] if r['case']['game']==g and r['arm']=='owned')) for g in GAMES],
        diversity=diversity,default_parity=parity,projection_checks=projection['checks'],horizon_episodes=288,horizon_summary=horizon_summary,bank_episodes=192,bank_summary=bank_summary,
        limitations=pre['limitations']+['repeated deterministic action streams are reported; episodes are not independent intelligence demonstrations','horizon follow-up selected after an observed loss; exploratory, not independent confirmation'])
    write(out/'summary.json',summary);write(out/'paired.json',dict(games=pair,life=life_pairs));write(out/'provenance.json',provenance)
    write(out/'preregister.json',pre);write(out/'contracts.json',diagnostic);write(out/'default_parity.json',parity)
    write(out/'projection_parity.json',projection);write(out/'horizons.json',horizons);write(out/'horizon_preregister.json',hp)
    write(out/'banks.json',banks);write(out/'bank_preregister.json',bp)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('persona','life_metrics','default_parity')},ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
