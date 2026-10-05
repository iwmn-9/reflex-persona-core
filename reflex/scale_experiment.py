"""Isolated CPU throughput and actual independent headless encounter measurements.

Run with CLI npc-scale. Timings never serve as CI pass/fail requirements.
"""
from dataclasses import fields
from pathlib import Path
import copy
import ctypes
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import numpy as np
from .core import Decisions, Policy, digest
from .examples import families
from .runtime import Population
from .scaling import TiledPolicy, array_bytes
from .teachers import write_json


def contexts(count, candidates=12, outcomes=2):
    """Heterogeneous authored numeric workload; not simulated gameplay/learning."""
    base = copy.deepcopy(families()[0]['context'])
    base['actions'] = []
    for j in range(candidates):
        base['actions'].append(dict(id=f'option-{j:02d}', target=None, legal=True,
            confidence=.65 + .02 * (j % 8), familiarity=.1 * (j % 7),
            known_failure=False, switch_cost=.01 * (j % 4), outcomes=[dict(
                p=1 / outcomes, objective=(j-h-candidates/2) / candidates,
                needs={'physiology':.1 * ((j+h)%5-2)}, values={'security':.1 * ((j+h)%3-1)},
                style={}, cost=.03 * (j % 3)) for h in range(outcomes)]))
    result = []
    for i in range(count):
        c = copy.deepcopy(base)
        c['scope']['npc'] = f'actor-{i}'
        c['personality'] = {key: ((i * 7 + j * 3) % 17) / 16 for j,key in enumerate(c['personality'])}
        c['values']['security'] = (i % 11) / 10
        for j, item in enumerate(c['needs'].values()):
            if item['supported'] and item['enabled']: item['deficit'] = ((i + j) % 13) / 12
        for a in c['actions']:
            a['outcomes'][0]['objective'] += .005 * (i % 7)
        result.append(c)
    return result


def peak_rss_bytes():
    """Whole-worker lifetime peak, including input construction; no subtraction."""
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)] + [
                (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        api = ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
        api.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        api.restype = wintypes.BOOL
        value = Counters(); value.cb = ctypes.sizeof(value)
        if not api(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return value.PeakWorkingSetSize
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == 'darwin' else peak * 1024)


def numeric_worker(count, candidates, outcomes, tile, repeats=15):
    source = contexts(count,candidates,outcomes)
    policy = Policy() if tile == 0 else TiledPolicy(tile)
    t = time.perf_counter(); pop = Population(source,policy)
    setup_ms = (time.perf_counter()-t)*1000
    # One unchanged reference verification before measuring mutable numeric steps.
    expected = Policy().decide(pop.batch); actual = policy.decide(pop.batch)
    for f in fields(Decisions): np.testing.assert_array_equal(getattr(expected,f.name),getattr(actual,f.name))
    del source, expected, actual; gc.collect()
    base_needs = pop.batch.needs.copy(); base_effects = pop.batch.effects.copy()
    base_conf = pop.batch.confidence.copy(); times = []; update_ms = []
    action_hashes = []
    for tick in range(repeats+2):
        t = time.perf_counter()
        needs = np.where(pop.batch.enabled,np.clip(base_needs + .015 * (tick%5-2),0,1),0)
        effects = base_effects.copy()
        effects[:,:,:,0] = np.clip(effects[:,:,:,0]+.007 * (tick%3-1),-1,1)
        confidence = np.clip(base_conf + .01*(tick%3-1),0,1)
        generated = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        decision = pop.step(needs=needs,effects=effects,confidence=confidence)
        elapsed = (time.perf_counter()-t)*1000
        action_hashes.append(digest(decision.action.tolist()))
        if tick >= 2: times.append(elapsed); update_ms.append(generated)
    p50 = statistics.median(times); p95 = float(np.percentile(times,95))
    return dict(npcs=count,candidates=candidates,outcomes=outcomes,tile=tile,repeats=repeats,
        compile_once_ms=setup_ms,step_p50_ms=p50,step_p95_ms=p95,
        update_array_generation_p50_ms=statistics.median(update_ms),
        decisions_per_second=count*1000/p50,batch_array_bytes=array_bytes(pop.batch),
        process_peak_rss_bytes=peak_rss_bytes(),action_hashes=action_hashes,
        scope='heterogeneous authored input; step includes validated dynamic needs/effects/confidence updates and persistent NPC state; excludes game, reading and IO')


def combat_run(encounters,tile,ticks=6,geometry_cache=True):
    """All six actors use the persona policy, no neutral/comparator free work."""
    if not geometry_cache:
        from unittest.mock import patch
        from .combat import terrain_zone_distance
        # Exact old BFS, for reproducible comparison; patch is confined to this
        # isolated benchmark invocation and restored before returning.
        with patch('reflex.combat.terrain_zone_distance',terrain_zone_distance.__wrapped__):
            result = combat_run(encounters,tile,ticks,True)
        result['geometry_cache'] = False
        return result
    from .combat import Battle,alive,terminal,route_context,make_context,resolve,battle_record,MAPS,GOALS
    from .laboratory import profiles
    from .routes import RouteState,choose_route
    worlds = [Battle.start(tuple(MAPS)[i%3],GOALS[i%4]) for i in range(encounters)]
    roster = profiles(); states = {}; memories = {}; samples = []; hashes = []; decisions_total = 0
    policy = Policy() if tile == 0 else TiledPolicy(tile)
    for tick in range(ticks):
        t = time.perf_counter(); cs = []; owners = []
        for e,w in enumerate(worlds):
            if terminal(w): continue
            for actor in alive(w,0)+alive(w,1):
                key = (e,actor); profile = roster[(e+actor)%len(roster)]
                state,_ = choose_route(route_context(w,actor,profile,e),states.get(key,RouteState()),uncertainty=.35)
                states[key] = state
                c = make_context(w,actor,profile,e,state.chosen,memories.get(key))
                c['scope']['episode'] = f'encounter-{e}'
                cs.append(c); owners.append(key)
        adapter_ms = (time.perf_counter()-t)*1000
        if not cs: break
        from .core import compile_batch
        t = time.perf_counter(); batch = compile_batch(cs)
        compile_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter(); records = policy.decide(batch).records(batch)
        score_ms = (time.perf_counter()-t)*1000
        choices = [{} for _ in worlds]
        for key,d in zip(owners,records):
            e,actor = key; choices[e][actor] = d['action_id']; memories[key] = d['next_state']
        t = time.perf_counter()
        for e,w in enumerate(worlds):
            if choices[e]: worlds[e],_ = resolve(w,choices[e],int(digest(['scale-world',e])[:16],16))
        world_ms = (time.perf_counter()-t)*1000
        hashes.append(digest([battle_record(w) for w in worlds]))
        decisions_total += len(owners)
        samples.append(dict(tick=tick,npcs=len(owners),adapter_ms=adapter_ms,compile_ms=compile_ms,
            score_and_records_ms=score_ms,world_ms=world_ms,
            total_ms=adapter_ms+compile_ms+score_ms+world_ms))
    warm = samples[1:] or samples
    return dict(encounters=encounters,initial_npcs=encounters*6,tile=tile,geometry_cache=True,decisions=decisions_total,
        samples=samples,state_hashes=hashes,
        warm_p50_ms=statistics.median(s['total_ms'] for s in warm),
        warm_p95_ms=float(np.percentile([s['total_ms'] for s in warm],95)),
        decisions_per_second=sum(s['npcs'] for s in warm)*1000/sum(s['total_ms'] for s in warm),
        scope='independent 3v3 encounters, all actors persona-controlled; includes routes, game effects, JSON validation, score, records and world; excludes rendering, physics, opponent learning and IO')


def experiment(output,progress=None):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    rows = []
    for candidates,outcomes in ((12,2),(16,4)):
        for n in (100,1000,10000):
            for tile in (0,256,1024):
                command = [sys.executable,'-m','reflex.scale_experiment','worker',str(n),str(candidates),str(outcomes),str(tile)]
                worker = subprocess.run(command,check=True,capture_output=True,text=True,encoding='utf-8',timeout=180,env=env)
                rows.append(json.loads(worker.stdout))
                if progress: progress(f'npc-scale: {n} NPC / {candidates}x{outcomes} / tile {tile}: {rows[-1]["step_p50_ms"]:.2f} ms')
    for candidates,outcomes in ((12,2),(16,4)):
        for n in (100,1000,10000):
            group = [r for r in rows if (r['npcs'],r['candidates'],r['outcomes'])==(n,candidates,outcomes)]
            assert all(r['action_hashes']==group[0]['action_hashes'] for r in group)
    combat = []
    for encounters in (10,100,300):
        for tile,cached in ((0,False),(0,True),(256,True)):
            command = [sys.executable,'-m','reflex.scale_experiment','combat',str(encounters),str(tile),'6',str(int(cached))]
            worker = subprocess.run(command,check=True,capture_output=True,text=True,encoding='utf-8',timeout=180,env=env)
            combat.append(json.loads(worker.stdout))
            if progress: progress(f'npc-scale combat: {encounters*6} initial NPC / tile {tile}: {combat[-1]["warm_p50_ms"]:.2f} ms')
        assert combat[-1]['state_hashes']==combat[-2]['state_hashes']==combat[-3]['state_hashes']
    result = dict(format='npc-scale-v1',python=platform.python_version(),numpy=np.__version__,
        platform=platform.system(),processor=platform.processor(),rows=rows,combat=combat,
        limitations=['one run per variant, 15 timed numeric steps; p95 is descriptive, not SLA',
            'sequential NumPy tiling, no multi-core/thread/GPU acceleration',
            'numeric workload is authored; actual combat is many independent 3v3 encounters, not a crowded shared arena',
            'persistent Population requires stable action IDs/capacity; combat still rebuilds changing JSON contexts',
            'RSS is whole child-process lifetime peak including setup/equality check, not scorer-only allocation',
            'tiling preserves decisions/state; input/output and numeric update copies remain proportional to population'])
    write_report(output,result)
    return result


def write_report(output,result):
    output = Path(output); rows = result['rows']; combat = result['combat']
    write_json(output/'evaluation.json',result)
    lines = ['# 大量NPC性能','',f'Python {result["python"]}, NumPy {result["numpy"]}, {result["platform"]} / {result["processor"]}.',
        '各条件は独立プロセス。NumPyのスレッド数はこの実験では1に固定。初期2tickを除く15tick、欲求/結果/確信度を毎回変更。数値stepは入力検証・コピー・個別記憶更新込み。入力配列生成は別掲。tile以上の人数でなければ同じ通常方式になる。各条件1実行なので小差を改善とは扱わない。',
        '', '|NPC|候補×結果|tile (0=通常)|step p50 ms|p95 ms|配列生成 p50 ms|判断/秒|入力配列 MiB|プロセスpeak MiB|',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'|{r["npcs"]}|{r["candidates"]}×{r["outcomes"]}|{r["tile"]}|{r["step_p50_ms"]:.2f}|{r["step_p95_ms"]:.2f}|{r["update_array_generation_p50_ms"]:.2f}|{r["decisions_per_second"]:.0f}|{r["batch_array_bytes"]/2**20:.2f}|{r["process_peak_rss_bytes"]/2**20:.2f}|')
    lines += ['', '全17tickで通常/256/1024の選択hash一致。Policy内部の式・人格・乱数・候補数・判断頻度は変更していない。TiledPolicyはPopulationへ任意指定する。通常Policyは既定のまま。',
        '', '## ゲーム状況評価を含む実測','',
        '各戦闘内は相互干渉する6人だが、戦闘間は独立。全員を人格Policyで動かし、方針選択・候補効果計算・JSON変換・採点・記憶・世界更新まで6tick実行する。以下は初期tickを除く5tickの中央値。時間は計測区間の合計で、選択の辞書振り分けなど区間外の小処理、画面/物理/相手学習は含めない。',
        '', '|初期NPC|独立戦闘|地形cache|tile|全処理 p50 ms|p95 ms|実判断/秒|状況変換 p50 ms|compile p50 ms|採点＋records p50 ms|世界 p50 ms|',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in combat:
        med = lambda key: statistics.median(s[key] for s in r['samples'][1:])
        lines.append(f'|{r["initial_npcs"]}|{r["encounters"]}|{r["geometry_cache"]}|{r["tile"]}|{r["warm_p50_ms"]:.2f}|{r["warm_p95_ms"]:.2f}|{r["decisions_per_second"]:.0f}|{med("adapter_ms"):.2f}|{med("compile_ms"):.2f}|{med("score_and_records_ms"):.2f}|{med("world_ms"):.2f}|')
    lines += ['', '通常/分割・地形cache有無で全tickの実盤面hash一致。共通採点の作業領域を抑える任意経路と、大量処理の費用内訳を追加。戦闘の静的地形から拠点への距離を512項目上限で共有し、占有/人格/体力/目標進捗は保存しない。判断内容は削っていない。追加費用は地形cacheと分割呼出し/出力結合で、小分けは小人数に不利な場合がある。',
        '現在の戦闘アダプターを大量実時間NPCにそのまま使える保証はない。候補効果/方針評価の再計算が重い場合は、ゲーム側の数値接続・共有地形計算・変化イベント更新が次の改善対象。頻度を落とす、候補を削る等の品質を変える処置はこの実験では行っていない。',
        '', '## 限界',''] + ['- '+x for x in result['limitations']] + ['']
    if result.get('hardware_verified'): lines.insert(3,result['hardware_verified'])
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


if __name__ == '__main__':
    # Limit BLAS threads via parent environment too, before NumPy is imported.
    mode = sys.argv[1]
    if mode == 'worker': result = numeric_worker(*map(int,sys.argv[2:]))
    elif mode == 'combat': result = combat_run(*map(int,sys.argv[2:]))
    else: raise ValueError('worker or combat expected')
    print(json.dumps(result,allow_nan=False))
