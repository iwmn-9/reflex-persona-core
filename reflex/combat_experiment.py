"""Actual concurrent battles, shared Policy batches and generic RouteState transfer."""
from pathlib import Path
from collections import Counter
from dataclasses import replace
import copy
import json
import time
import numpy as np
from .core import Policy,compile_batch,digest
from .routes import RouteState,choose_route
from .laboratory import profiles
from .cross_games import core_hashes
from .teachers import write_json
from .combat import (Battle,Unit,GOALS,MAPS,HP,MAGAZINE,alive,legal,terminal,victory,
    resolve,preview,route_context,make_context,reference,battle_record,battle_from_record)


def run(profile,seed,map_name,goal,record=True):
    w=Battle.start(map_name,goal); focal=seed%2; routes={i:RouteState() for i in range(6)}
    memories={}; trace=[]; counts=Counter(); route_counts=Counter(); elapsed=[]; collisions=0
    world_seed=int(digest(['combat-world',seed])[:16],16)
    while not terminal(w):
        before=w; contexts=[]; actors=[]; proposals={}; t=time.perf_counter()
        for actor in alive(w,focal):
            rc=route_context(w,actor,profile,seed)
            routes[actor],proposal=choose_route(rc,routes[actor],uncertainty=.35)
            proposals[actor]=proposal; route_counts[proposal['chosen']]+=1
            actors.append(actor); contexts.append(make_context(w,actor,profile,seed,proposal['chosen'],memories.get(actor)))
        batch=compile_batch(contexts); decisions=Policy().decide(batch).records(batch)
        elapsed.append((time.perf_counter()-t)*1000)
        choices={i:d['action_id'] for i,d in zip(actors,decisions)}
        for actor,d in zip(actors,decisions):
            memories[actor]=d['next_state']; counts[d['action_id'].split(':')[0]]+=1
        # The comparator sees the SAME pre-state; it receives no selected focal
        # actions. Neither set of intents is revealed until both are committed.
        for actor in alive(w,1-focal): choices[actor]=reference(w,actor,seed)
        w,audit=resolve(w,choices,world_seed); collisions+=audit['collisions']
        trace.append(dict(before=battle_record(before),actors=actors,contexts=contexts,decisions=decisions,
            proposals=proposals,choices=choices,audit=audit,after=battle_record(w)) if record else None)
    return dict(profile=profile['id'],seed=seed,map=map_name,goal=goal,team=focal,
        won=focal in victory(w),lost=1-focal in victory(w),winner=list(victory(w)),ticks=w.tick,
        alive=len(alive(w,focal)),health=sum(w.units[i].hp for i in alive(w,focal)),
        team_actions=dict(counts),route_choices=dict(route_counts),route_switches=sum(s.switches for s in routes.values()),
        movement_conflicts=collisions,team_decision_p50_ms=float(np.median(elapsed)),
        team_decision_p95_ms=float(np.percentile(elapsed,95)),final=battle_record(w),world_seed=world_seed,trace=trace)


def replay(r):
    checked=0; route_states={i:RouteState() for i in range(6)}
    for row in r['trace']:
        before=battle_from_record(row['before']); choices={int(k):v for k,v in row['choices'].items()}
        after,audit=resolve(before,choices,r['world_seed'])
        assert json.loads(json.dumps(battle_record(after)))==json.loads(json.dumps(row['after']))
        assert audit==row['audit']
        batch=compile_batch(row['contexts']); decisions=Policy().decide(batch).records(batch)
        assert decisions==row['decisions']
        for actor,c,d in zip(row['actors'],row['contexts'],decisions):
            assert d==Policy().choose(c); assert d['action_id']==choices[actor]
            assert c['facts']['public_tick']==str(before.tick)
            assert 'world_seed' not in c and 'choices' not in c['facts']
            proposal=row['proposals'].get(actor,row['proposals'].get(str(actor)))
            route_states[actor],candidate=choose_route(proposal['context'],route_states[actor],uncertainty=.35)
            assert candidate==proposal
            checked+=1
    return checked


def diagnostic_probe():
    # Fixed diagnostic scene with wounded ally, loaded foes, LOS, cover, objective.
    base=Battle.start('cover','both')
    roster=list(base.units)
    roster[0]=replace(roster[0],x=3,y=2)
    roster[1]=replace(roster[1],x=3,y=1,hp=3,ammo=0)
    roster[3]=replace(roster[3],x=5,y=2,hp=3)
    w=replace(base,units=tuple(roster))
    aid=list(base.units); aid[1]=replace(aid[1],x=1,y=1,hp=3)
    scenes={'threatened_ally':w,'safe_ally':replace(base,units=tuple(aid)),
            'loaded_fight':replace(w,units=tuple(replace(u,hp=9) for u in w.units))}
    rows=[]
    for scene,w in scenes.items():
        for actor in (0,1,2):
            contexts=[]; ps=profiles()
            for profile in ps:
                state,route=choose_route(route_context(w,actor,profile,19))
                c=make_context(w,actor,profile,19,state.chosen)
                contexts.append(c); d=Policy().choose(c)
                rows.append(dict(scene=scene,actor=actor,profile=profile['id'],route=state.chosen,action=d['action_id'],context=c))
            batch=compile_batch(contexts)
            assert Policy().decide(batch).records(batch)==[Policy().choose(c) for c in contexts]
    differences=sum(len({r['action'] for r in rows if r['actor']==i and r['scene']==s})>1 for s in scenes for i in (0,1,2))
    return dict(rows=rows,scenes={s:battle_record(w) for s,w in scenes.items()},persona_different_actors=differences,
        actor_scenes=9,batch_single_identity=len(rows),scope='fixed development diagnostics, not a strength benchmark')


def experiment(output,seeds=2,progress=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); before=core_hashes(); runs=[]; checked=0
    roster=profiles()+[dict(id='neutral',traits=(.5,)*5,values={})]
    with (output/'trajectories.jsonl').open('w',encoding='utf-8') as stream:
        for map_name in MAPS:
            for goal in GOALS:
                for profile in roster:
                    for seed in range(seeds):
                        r=run(profile,seed,map_name,goal); checked+=replay(r)
                        stream.write(json.dumps(r,ensure_ascii=False)+'\n')
                        runs.append({k:v for k,v in r.items() if k!='trace'})
                if progress: progress(f'combat/{map_name}/{goal}: {len(runs)} battles, {checked} batch/single decisions replayed')
    summary=[]
    for goal in GOALS:
        for profile in roster:
            group=[r for r in runs if r['goal']==goal and r['profile']==profile['id']]; actions=Counter(); routes=Counter()
            for r in group: actions.update(r['team_actions']); routes.update(r['route_choices'])
            summary.append(dict(goal=goal,profile=profile['id'],games=len(group),wins=sum(r['won'] for r in group),
                losses=sum(r['lost'] for r in group),draws=sum(not r['won'] and not r['lost'] for r in group),
                mean_survivors=float(np.mean([r['alive'] for r in group])),actions=dict(actions),routes=dict(routes),
                route_switches=sum(r['route_switches'] for r in group),
                median_team_p50_ms=float(np.median([r['team_decision_p50_ms'] for r in group]))))
    probe=diagnostic_probe(); assert before==core_hashes()
    result=dict(format='headless-combat-v1',battles=len(runs),seeds=seeds,replayed_decisions=checked,
        core_before=before,core_after=core_hashes(),summary=summary,probe=probe,runs=runs,
        limitations=['authored fully public turn-synchronous 3v3 test, not commercial/real-time combat',
            'rules/objective/LOS/path and effect proxies are combat-owned; not rules-only automatic generalization',
            'stationary/posture rival assumption and uniform-target incoming damage proxy can be wrong',
            'no opponent learning or coordinated joint action planning in this adapter',
            'simultaneous destination conflicts and ally kits can be wasted; measured instead of hidden',
            'route desirability mapping is game-authored, not a validated psychological trait-to-tactic mapping',
            'small development maps/seeds and fixed comparator; not broad combat intelligence evidence',
            'team timings include JSON/schema/route conversion but exclude enemy decisions/world/logging'])
    write_json(output/'evaluation.json',result)
    lines=['# GUIなし戦闘への人格コア接続','',
        f'{len(runs)} actual 3v3 battles, {checked} decisions replayed; core/runtime unchanged.', '',
        '公開9×5盤面、射程4、壁の射線遮断、遮蔽物/防御の命中率低下、HP9/弾3/救急品1。移動/射撃/装填/防御/隣接味方への回復を同時に選ぶ。撃破・拠点確保・どちらか(OR)・両方(AND)の4条件。拠点は3tick多数占有で確保実績を保存、ANDは確保実績と敵全滅を両方満たし、自軍生存も必要。',
        '', '|goal|profile|games|wins|losses|draws|survivors|team p50 ms|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summary: lines.append(f'|{s["goal"]}|{s["profile"]}|{s["games"]}|{s["wins"]}|{s["losses"]}|{s["draws"]}|{s["mean_survivors"]:.2f}|{s["median_team_p50_ms"]:.2f}|')
    lines+=['','## 得たものと追加費用','',
        '経済ゲームの科学/文化/勢力を参照せず、既存の共通routes.choose_routeとPolicyをそのまま再利用。ルートの選択/切替と、合法な戦闘行動の人格評価を分離。生存/安全/承認の欲求、援護/権力/達成の価値をゲーム側から渡す。',
        '味方最大3人の判断は同じ選択前盤面から1つのPolicyバッチで採点。全員が確定した後に移動/回復/同時射撃を解決。相手の現在選択や命中乱数はコアへ渡さない。全実判断でバッチ/個別が一致。',
        '撃破の未達代理値は敵HP減少.65＋可射撃位置への近さ.35。射撃地点への静的地形距離は歩行できる場所と射程/射線から求める。接近を撃破達成には数えず、移動占有/相手移動/正しい終局EVは未計算。',
        '追加したのは戦闘ルールと接続評価。射程/射線/地形距離/生存価値/拠点進行のゲーム別計算は必要。物理演算/描画/リアルタイム照準/霧情報/読み学習/チーム連携計画は未追加。',
        '', '## 同一盤面の人格差','',f'{probe["persona_different_actors"]}/{probe["actor_scenes"]} actor-scenes have action differences across four profiles; {probe["batch_single_identity"]} batch/single matches.', '',
        '|scene|actor|profile|route|action|','|---|---:|---|---|---|']
    lines += [f'|{r["scene"]}|{r["actor"]}|{r["profile"]}|{r["route"]}|{r["action"]}|' for r in probe['rows']]
    lines+=['','## 限界','']+[f'- {x}' for x in result['limitations']]+['']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
