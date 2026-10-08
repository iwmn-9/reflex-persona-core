"""CPU entry points, data export and separate compute/validation benchmarks."""
import argparse
import copy
import json
from pathlib import Path
import platform
import statistics
import time
import numpy as np
from .core import Policy, compile_batch, digest, VERSION
from .examples import dataset, families
from .teachers import prompt, write_json, approved_examples
from .training import fit, evaluate
from .runtime import Population


def export(root, variants=12):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    seeds=dataset(variants)
    (root/"design_seeds.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in seeds),encoding="utf-8")
    (root/"teacher_requests.jsonl").write_text("".join(json.dumps(dict(case_id=r["id"],context=r["context"],
        context_hash=digest(r["context"]),prompt=prompt(r["context"]),quality="awaiting_generation"),ensure_ascii=False)+"\n" for r in families()),encoding="utf-8")
    write_json(root/"manifest.json",dict(version=VERSION,semantic_examples=len(families()),rows=len(seeds),
        hash=digest(seeds),source="authored design fixtures, not generated LLM answers",
        evaluated=evaluate(seeds),accepted_llm_rows_created_by_export=0))
    lines=["# 判断事例の相談用一覧", "", "設計した教師シード。LLMの回答・人間の採用済みデータではない。数値の正本はdesign_seeds.jsonl。", "", "|事例|狙った選択|理由|", "|---|---|---|"]
    lines += [f"|{r['id']}|{', '.join(r['preferred'])}|{r['reason']}|" for r in families()]
    (root/"example_review.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return seeds


def benchmark(output, sizes=(1,100,1000,10000), repeats=15):
    rows=[]; sample=families()[0]["context"]; p=Policy()
    workloads=[("two_candidates",sample,sizes)]
    stress=copy.deepcopy(sample); stress["actions"]=[]
    for j in range(16):
        stress["actions"].append(dict(id=f"option-{j:02d}",target=None,legal=True,confidence=.8,familiarity=.5,
            known_failure=False,switch_cost=0,outcomes=[dict(p=.25,objective=(j-h)/32,needs={"physiology":(j%3)/4},values={},style={},cost=.1) for h in range(4)]))
    workloads.append(("sixteen_candidates_four_outcomes",stress,(100,1000,10000)))
    for name,sample,populations in workloads:
      for n in populations:
        contexts=[copy.deepcopy(sample) for _ in range(n)]
        for i,c in enumerate(contexts): c["scope"]["npc"]=f"actor-{i}"
        # Measure the full validated boundary and numeric scorer separately.
        start=time.perf_counter(); b=compile_batch(contexts); pack_ms=(time.perf_counter()-start)*1000
        p.decide(b); times=[]
        for _ in range(repeats):
            start=time.perf_counter(); p.decide(b); times.append((time.perf_counter()-start)*1000)
        population=Population(contexts); population.step(); runtime_times=[]
        for _ in range(repeats):
            start=time.perf_counter(); population.step(); runtime_times.append((time.perf_counter()-start)*1000)
        # Includes validation, per-NPC hash/RNG creation and scoring; no cached-input claim.
        boundary=[]
        for _ in range(3):
            start=time.perf_counter(); fresh=compile_batch(contexts); p.decide(fresh); boundary.append((time.perf_counter()-start)*1000)
        rows.append(dict(workload=name,npcs=n,candidates=b.legal.shape[1],outcomes=b.probability.shape[2],
            compile_once_ms=pack_ms,score_p50_ms=statistics.median(times),score_p95_ms=float(np.percentile(times,95)),
            persistent_runtime_p50_ms=statistics.median(runtime_times),persistent_runtime_p95_ms=float(np.percentile(runtime_times,95)),
            validated_end_to_end_p50_ms=statistics.median(boundary),
            decisions_per_second=n/(statistics.median(times)/1000)))
    result=dict(version=VERSION,platform=platform.platform(),processor=platform.processor(),python=platform.python_version(),
        numpy=np.__version__,repeats=repeats,scope="CPU numeric batch; compile separate; excludes game simulation, reading, LLM and IO",rows=rows)
    write_json(output,result); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("export","demo","benchmark","seed-smoke","train-reviewed","lab","cross-games","reversi","quoridor","board-planning","monte-carlo","goofspiel","opponent-reading","adaptive-reading","rule-baseline","rolling-plans","mahjong","judgment","resource-world","contests","victory-routes","combat","npc-scale","decision-loop","validation","purpose","stability","purposeful-wait","recovery-options","execution-alignment","self-continuation","root-collision","intertemporal","payback-cycle","purpose-recovery","observed-transfer","search-transfer","continuity-transfer"))
    parser.add_argument("--output",default="reflex_artifacts"); parser.add_argument("--reviews")
    args=parser.parse_args(); root=Path(args.output)
    if args.command=="export": print("Exported",len(export(root)),"authored seed rows; LLM generation pending")
    elif args.command=="demo":
        p=Policy()
        for row in families():
            d=p.choose(row["context"],False)
            print(f"{row['id']}: {d['action_id']} / {row['reason']}")
    elif args.command=="benchmark": print(json.dumps(benchmark(root/"benchmark.json"),ensure_ascii=False,indent=2))
    elif args.command=="seed-smoke": print(json.dumps(fit(export(root),root/"seed_smoke_ranker.json",True),ensure_ascii=False,indent=2))
    elif args.command=="lab":
        from .laboratory import experiment
        result=experiment(root/"laboratory")
        print(json.dumps({k:v for k,v in result.items() if k!="runs"},ensure_ascii=False,indent=2))
    elif args.command=="cross-games":
        from .cross_games import experiment
        result=experiment(root/"cross_games")
        print(json.dumps({k:v for k,v in result.items() if k!="runs"},ensure_ascii=False,indent=2))
    elif args.command=="reversi":
        from .reversi import experiment
        result=experiment(root/"reversi")
        summary={k:v for k,v in result.items() if k!="runs"}
        summary['personality_probe']={k:v for k,v in result['personality_probe'].items() if k!='rows'}
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    elif args.command=="quoridor":
        from .quoridor import experiment
        result=experiment(root/"quoridor")
        print(json.dumps({k:v for k,v in result.items() if k!="personality_probe"},ensure_ascii=False,indent=2))
    elif args.command=="board-planning":
        from .board_planning import experiment
        result=experiment(root/"board_planning")
        print(json.dumps({k:v for k,v in result.items() if k!="tactical_probe"},ensure_ascii=False,indent=2))
    elif args.command=="monte-carlo":
        from .monte_carlo_comparison import experiment
        result=experiment(root/"monte_carlo")
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.command=="goofspiel":
        from .goofspiel_experiment import experiment
        result=experiment(root/"goofspiel",progress=print)
        print(json.dumps({k:v for k,v in result.items() if k not in ('exact_probe','population_probe')},ensure_ascii=False,indent=2))
    elif args.command=="opponent-reading":
        from .opponent_reading_experiment import experiment
        result=experiment(root/"opponent_reading",progress=print)
        print(json.dumps({k:v for k,v in result.items() if k!='paired'},ensure_ascii=False,indent=2))
    elif args.command=="adaptive-reading":
        from .adaptive_reading_experiment import experiment
        result=experiment(root/"adaptive_reading",progress=print)
        print(json.dumps({k:v for k,v in result.items() if k not in ('paired','own_adversity_probe')},ensure_ascii=False,indent=2))
    elif args.command=="rule-baseline":
        from .rule_baseline_experiment import experiment
        result=experiment(root/"rule_baseline",progress=print)
        print(json.dumps({k:v for k,v in result.items() if k!='personality_probe'},ensure_ascii=False,indent=2))
    elif args.command=="rolling-plans":
        from .rolling_plan_experiment import experiment
        result=experiment(root/"rolling_plans",progress=print)
        print(json.dumps({k:v for k,v in result.items() if k not in ('paired','score_change_probe')},ensure_ascii=False,indent=2))
    elif args.command=="mahjong":
        from .mahjong_experiment import experiment
        result=experiment(root/"mahjong",progress=print)
        print((root/'mahjong'/'REPORT.md').read_text(encoding='utf-8'))
    elif args.command=="judgment":
        from .judgment_experiment import experiment
        experiment(root/"judgment",progress=print)
        print((root/'judgment'/'REPORT.md').read_text(encoding='utf-8'))
    elif args.command=="resource-world":
        from .resource_experiment import experiment
        experiment(root/"resource_world",progress=print)
        print((root/'resource_world'/'REPORT.md').read_text(encoding='utf-8'))
    elif args.command=='root-collision':
        from .root_collision_experiment import experiment
        result=experiment(root/'root_collision')
        print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=False,indent=2))
    elif args.command=='self-continuation':
        from .continuation_experiment import experiment
        result=experiment(root/'persona_continuation',progress=print)
        print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=False,indent=2))
    elif args.command=='execution-alignment':
        from .alignment_experiment import experiment
        result=experiment(root/'execution_alignment',progress=print)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.command=='purpose-recovery':
        from .purpose_recovery import experiment
        result=experiment(root/'purpose_recovery',progress=print)
        print(json.dumps({k:v for k,v in result.items() if k not in ('runs','source_hashes')},ensure_ascii=False,indent=2))
    elif args.command=='continuity-transfer':
        from .continuity_transfer import experiment
        result=experiment(root/'continuity_transfer',progress=print,workers=4)
        print(json.dumps({k:v for k,v in result.items() if k not in ('runs','source_hashes')},ensure_ascii=False,indent=2))
    elif args.command=='search-transfer':
        from .search_transfer import experiment
        result=experiment(root/'search_transfer',progress=print,workers=4)
        print(json.dumps({k:v for k,v in result.items() if k not in ('runs','source_hashes')},ensure_ascii=False,indent=2))
    elif args.command=='observed-transfer':
        from .observed_transfer import experiment
        result=experiment(root/'observed_transfer',progress=print,workers=4)
        print(json.dumps({k:v for k,v in result.items() if k not in ('runs','source_hashes')},ensure_ascii=False,indent=2))
    elif args.command=='payback-cycle':
        from .payback_cycle import experiment
        result=experiment(root/'closed_loop_payback',progress=print)
        print(json.dumps({k:v for k,v in result.items() if k not in ('runs','source_hashes')},ensure_ascii=False,indent=2))
    elif args.command=='intertemporal':
        from .intertemporal_experiment import experiment
        experiment(root/'intertemporal_integrated',progress=print)
        print((root/'intertemporal_integrated'/'REPORT.md').read_text(encoding='utf-8'))
    elif args.command=='recovery-options':
        from .recovery_experiment import experiment
        experiment(root/'recovery_options',progress=print)
        print((root/'recovery_options'/'REPORT.md').read_text(encoding='utf-8'))
    elif args.command=='purposeful-wait':
        from .wait_experiment import experiment
        experiment(root/'purposeful_wait',progress=print)
        print((root/'purposeful_wait'/'REPORT.md').read_text(encoding='utf-8'))
    elif args.command=='stability':
        from .stability_experiment import experiment
        experiment(root/'stability',progress=print)
        print((root/'stability'/'REPORT.md').read_text(encoding='utf-8'))
    elif args.command=='purpose':
        from .purpose_experiment import experiment
        experiment(root/'purpose',progress=print)
        print((root/'purpose'/'REPORT.md').read_text(encoding='utf-8'))
    elif args.command=='validation':
        from .validation_experiment import experiment
        experiment(root/'validation',progress=print)
        print((root/'validation'/'REPORT.md').read_text(encoding='utf-8'))
    elif args.command=='decision-loop':
        from .loop_experiment import experiment
        experiment(root/'decision_loop',progress=print)
        print((root/'decision_loop'/'REPORT.md').read_text(encoding='utf-8'))
    elif args.command=='npc-scale':
        from .scale_experiment import experiment
        experiment(root/'npc_scale',progress=print)
        print((root/'npc_scale'/'REPORT.md').read_text(encoding='utf-8'))
    elif args.command=='combat':
        from .combat_experiment import experiment
        experiment(root/'combat',progress=print)
        print((root/'combat'/'REPORT.md').read_text(encoding='utf-8'))
    elif args.command in ('contests','victory-routes'):
        if args.command=='contests':
            from .contests_experiment import experiment
            folder='contests'
        else:
            from .route_experiment import experiment
            folder='victory_routes'
        experiment(root/folder,progress=print)
        print((root/folder/'REPORT.md').read_text(encoding='utf-8'))
    else:
        if not args.reviews: parser.error("--reviews required")
        rows=approved_examples(args.reviews)
        # Freeze split per episode family; human-reviewed data only.
        for row in rows:
            bucket=int(digest(row["family"])[:8],16)%10
            row["split"]="test" if bucket==9 else "validation" if bucket==8 else "train"
        print(json.dumps(fit(rows,root/"reviewed_ranker.json"),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
