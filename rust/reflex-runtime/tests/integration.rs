#[path = "../examples/worlds/mod.rs"]
mod worlds;
use reflex_runtime::{immediate::*, *};
use serde::{Deserialize, Serialize};
use worlds::*;

fn setup(n: usize) -> (Runtime, Vec<Config>, DemoWorld) {
    let configs = configs(Kind::Resources, n);
    let world = initial(Kind::Resources, &configs);
    (
        Runtime::new(session(Kind::Resources), 0, &configs).unwrap(),
        configs,
        world,
    )
}
fn inputs(runtime: &Runtime, world: &DemoWorld) -> Vec<Input> {
    runtime
        .actors()
        .filter_map(|(npc, a)| {
            world.observe(a.config()).unwrap().map(|observation| Input {
                npc: npc.into(),
                observation,
            })
        })
        .collect()
}
fn receipts(batch: &Batch) -> Vec<Receipt> {
    batch
        .decisions()
        .map(|(npc, d)| Receipt {
            npc: npc.into(),
            execution: Execution::Executed {
                action_id: d.action_id.clone(),
                target: d.target.clone(),
            },
        })
        .collect()
}
fn saved(runtime: &Runtime) -> Vec<u8> {
    runtime
        .checkpoint(runtime.world_revision(), "test-v1", b"world")
        .unwrap()
}

#[test]
fn batch_matches_individuals_and_parallel_is_order_independent() {
    let (runtime, _, world) = setup(128);
    let input = inputs(&runtime, &world);
    let before = saved(&runtime);
    let a = runtime.prepare(0, &input, true, 1).unwrap();
    let mut reverse = input.clone();
    reverse.reverse();
    let b = runtime.prepare(0, &reverse, true, 4).unwrap();
    assert_eq!(
        a.decisions().collect::<Vec<_>>(),
        b.decisions().collect::<Vec<_>>()
    );
    for (npc, decision) in a.decisions() {
        let row = input.iter().find(|i| i.npc == npc).unwrap();
        let independent = runtime
            .actor(npc)
            .unwrap()
            .preview(&row.observation, true)
            .unwrap();
        assert_eq!(decision, independent.decision());
    }
    assert_eq!(saved(&runtime), before);
}

#[test]
fn bad_last_receipt_never_partly_commits_population() {
    for fault in 0..7 {
        let (mut runtime, _, world) = setup(3);
        let before = saved(&runtime);
        let batch = runtime
            .prepare(0, &inputs(&runtime, &world), true, 2)
            .unwrap();
        let mut rs = receipts(&batch);
        match fault {
            0 => {
                rs[2].execution = Execution::Executed {
                    action_id: "not-selected".into(),
                    target: None,
                }
            }
            1 => {
                rs[2].execution = Execution::Executed {
                    action_id: batch.decisions().last().unwrap().1.action_id.clone(),
                    target: Some("wrong".into()),
                }
            }
            2 => {
                rs.pop();
            }
            3 => rs[2].npc = rs[0].npc.clone(),
            4 => rs[2].npc = "unrelated".into(),
            5 => {
                rs[2].execution = Execution::NotExecuted {
                    reason: String::new(),
                }
            }
            _ => {}
        }
        assert!(runtime
            .resolve(batch, if fault == 6 { 0 } else { 1 }, &rs)
            .is_err());
        assert_eq!(saved(&runtime), before, "fault {fault}");
    }
}

#[test]
fn blocked_execution_preserves_intent_and_tick_but_world_advances() {
    let (mut runtime, _, world) = setup(2);
    let input = inputs(&runtime, &world);
    let skipped = runtime.actor(&input[1].npc).unwrap().checkpoint().unwrap();
    let batch = runtime.prepare(0, &input, true, 1).unwrap();
    let stale = runtime.prepare(0, &input, true, 1).unwrap();
    let mut rs = receipts(&batch);
    rs[1].execution = Execution::NotExecuted {
        reason: "blocked by simultaneous action".into(),
    };
    runtime.resolve(batch, 1, &rs).unwrap();
    assert_eq!(runtime.world_revision(), 1);
    assert_eq!(runtime.actor(&input[0].npc).unwrap().last_tick(), Some(0));
    assert_eq!(
        runtime.actor(&input[1].npc).unwrap().checkpoint().unwrap(),
        skipped
    );
    assert!(runtime.validate_batch(&stale).is_err());
    // RNG is derived from the actor's observation tick, not attempts or batch order.
    let retry = runtime.prepare(1, &input[1..], true, 1).unwrap();
    let solo = runtime
        .actor(&input[1].npc)
        .unwrap()
        .preview(&input[1].observation, true)
        .unwrap();
    assert_eq!(retry.decisions().next().unwrap().1, solo.decision());
}

#[test]
fn ownership_restore_roster_changes_and_world_updates_expire_proposals() {
    let (mut runtime, configs, world) = setup(2);
    let input = inputs(&runtime, &world);
    let wrong_runtime = Runtime::new(session(Kind::Resources), 0, &configs).unwrap();
    assert!(wrong_runtime
        .validate_batch(&runtime.prepare(0, &input, true, 1).unwrap())
        .is_err());
    let b = runtime.prepare(0, &input, true, 1).unwrap();
    let (restored, _) = Runtime::restore(
        session(Kind::Resources),
        &configs,
        "test-v1",
        &saved(&runtime),
    )
    .unwrap();
    assert!(restored.validate_batch(&b).is_err());
    let b = runtime.prepare(0, &input, true, 1).unwrap();
    let actor = runtime.remove(&configs[0].scope.npc).unwrap();
    runtime.insert(actor.config().clone()).unwrap(); // same ID/config is a new incarnation
    assert!(runtime.validate_batch(&b).is_err());
    let b = runtime.prepare(0, &input, true, 1).unwrap();
    runtime.reset_intent(&configs[0].scope.npc).unwrap();
    assert!(runtime.validate_batch(&b).is_err());
    let b = runtime.prepare(0, &input, true, 1).unwrap();
    runtime.advance_world(0, 7).unwrap();
    assert!(runtime.validate_batch(&b).is_err());
    assert!(runtime.prepare(0, &input, true, 1).is_err());
}

#[test]
fn boundary_errors_do_not_consume_state_and_have_actor_identity() {
    let (mut runtime, configs, world) = setup(3);
    let mut input = inputs(&runtime, &world);
    let before = saved(&runtime);
    input[2].observation.actions[0].outcomes[0].p = f64::NAN;
    let error = runtime.prepare(0, &input, true, 3).unwrap_err();
    assert!(error.to_string().contains(&configs[2].scope.npc));
    assert_eq!(saved(&runtime), before);
    let mut bad = configs[0].clone();
    bad.scope.episode = "wrong-world".into();
    assert!(runtime.insert(bad).is_err());
    assert!(runtime.insert(configs[0].clone()).is_err());
    assert!(runtime.remove("missing").is_err());
    assert!(runtime.reset_intent("missing").is_err());
    assert!(runtime.advance_world(1, 2).is_err());
    assert!(runtime.checkpoint(1, "test-v1", b"world").is_err());
    let input = inputs(&runtime, &world);
    for workers in [0, MAX_WORKERS + 1] {
        assert!(runtime.prepare(0, &input, true, workers).is_err());
    }
    assert!(runtime.prepare(0, &[], true, 1).is_err());
    assert!(runtime
        .prepare(0, &[input[0].clone(), input[0].clone()], true, 1)
        .is_err());
    assert_eq!(saved(&runtime), before);
}

#[test]
fn save_checks_exact_roster_config_schema_world_and_integrity() {
    let (runtime, configs, _) = setup(3);
    let bytes = saved(&runtime);
    let s = session(Kind::Resources);
    let (r, w) = Runtime::restore(s.clone(), &configs, "test-v1", &bytes).unwrap();
    assert_eq!(w, b"world");
    assert_eq!(bytes, saved(&r));
    assert!(Runtime::restore(s.clone(), &configs[..2], "test-v1", &bytes).is_err());
    let mut wrong = configs.clone();
    wrong[0].personality[0] = 0.123;
    assert!(Runtime::restore(s.clone(), &wrong, "test-v1", &bytes).is_err());
    assert!(Runtime::restore(s.clone(), &configs, "another-format", &bytes).is_err());
    assert!(Runtime::restore(session(Kind::Combat), &configs, "test-v1", &bytes).is_err());
    for fault in 0..5 {
        let mut json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        match fault {
            0 => json["payload"]["world"][0] = 0.into(),
            1 => json["payload"]["world_revision"] = 5.into(),
            2 => json["payload"]["version"] = "future".into(),
            3 => json["payload"]["actors"][0]["checkpoint"] = "{}".into(),
            _ => json["unknown"] = true.into(),
        }
        assert!(Runtime::restore(
            s.clone(),
            &configs,
            "test-v1",
            &serde_json::to_vec(&json).unwrap()
        )
        .is_err());
    }
    assert!(Runtime::restore(s, &configs, "test-v1", &bytes[..bytes.len() - 1]).is_err());
}

#[test]
fn both_worlds_run_through_shared_api_and_restart_exactly() {
    for kind in [Kind::Resources, Kind::Combat] {
        let configs = configs(kind, 6);
        let mut whole = GameSession::new(session(kind), &configs, initial(kind, &configs)).unwrap();
        let mut resumed =
            GameSession::new(session(kind), &configs, initial(kind, &configs)).unwrap();
        let mut executed = 0;
        let mut blocked = 0;
        for _ in 0..256 {
            let left = whole.step(true, 1).unwrap();
            let right = resumed.step(true, 3).unwrap();
            assert_eq!(
                serde_json::to_value(&left).unwrap(),
                serde_json::to_value(&right).unwrap()
            );
            for r in left {
                match r.execution {
                    Execution::Executed { .. } => executed += 1,
                    Execution::NotExecuted { .. } => blocked += 1,
                }
            }
            assert_eq!(whole.world(), resumed.world());
            let save = resumed.checkpoint().unwrap();
            resumed = GameSession::<DemoWorld>::restore(session(kind), &configs, &save).unwrap();
            assert_eq!(save, resumed.checkpoint().unwrap());
        }
        assert!(executed > 0);
        if kind == Kind::Resources {
            assert!(blocked > 0);
        }
        assert_eq!(whole.checkpoint().unwrap(), resumed.checkpoint().unwrap());
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct FaultWorld {
    inner: DemoWorld,
    fault: u8,
}
impl World for FaultWorld {
    const SAVE_FORMAT: &'static str = "fault-test-v1";
    fn revision(&self) -> u64 {
        self.inner.tick
    }
    fn observe(&self, c: &Config) -> reflex_runtime::Result<Option<Observation>> {
        if self.fault == 0 {
            return Err(reflex_runtime::Error::Invalid("perception unavailable"));
        }
        self.inner.observe(c)
    }
    fn advance(&self, c: &[Choice<'_>]) -> reflex_runtime::Result<(Self, Vec<Receipt>)> {
        let (mut next, mut receipts) = self.inner.advance(c)?;
        match self.fault {
            1 => return Err(reflex_runtime::Error::Invalid("rule failure after staging")),
            2 => {
                receipts.pop();
            }
            3 => next.tick = self.inner.tick,
            _ => {}
        }
        Ok((
            Self {
                inner: next,
                fault: self.fault,
            },
            receipts,
        ))
    }
}
#[test]
fn integrated_errors_rollback_world_and_entire_population() {
    let configs = configs(Kind::Resources, 3);
    for fault in 0..4 {
        let mut game = GameSession::new(
            session(Kind::Resources),
            &configs,
            FaultWorld {
                inner: initial(Kind::Resources, &configs),
                fault,
            },
        )
        .unwrap();
        let before = game.checkpoint().unwrap();
        assert!(game.step(true, 2).is_err());
        assert_eq!(game.checkpoint().unwrap(), before, "fault {fault}");
    }
}

#[test]
fn empty_population_and_late_join_follow_world_without_resetting_it() {
    let configs = configs(Kind::Resources, 3);
    let mut game = GameSession::new(
        session(Kind::Resources),
        &[],
        initial(Kind::Resources, &configs),
    )
    .unwrap();
    for _ in 0..37 {
        assert!(game.step(true, 1).unwrap().is_empty());
    }
    game.insert(configs[1].clone()).unwrap();
    game.step(true, 1).unwrap();
    assert_eq!(
        game.runtime()
            .actor(&configs[1].scope.npc)
            .unwrap()
            .last_tick(),
        Some(37)
    );
    game.remove(&configs[1].scope.npc).unwrap();
    let before = game.world().clone();
    assert!(game.replace_world(before.clone()).is_err());
    assert_eq!(game.world(), &before);
    let mut next = before;
    next.tick += 5;
    game.replace_world(next).unwrap();
    assert_eq!(game.runtime().world_revision(), 43);
}

#[test]
fn ten_thousand_actor_boundary_is_real_and_order_stable() {
    let (runtime, _, world) = setup(MAX_AGENTS);
    // Exercise the population bound without benchmarking O(N^2) demo perception.
    let template = world
        .observe(runtime.actors().next().unwrap().1.config())
        .unwrap()
        .unwrap();
    let mut input: Vec<_> = runtime
        .actors()
        .map(|(npc, _)| Input {
            npc: npc.into(),
            observation: template.clone(),
        })
        .collect();
    input.reverse();
    let batch = runtime.prepare(0, &input, false, 2).unwrap();
    assert_eq!(batch.decisions().len(), MAX_AGENTS);
    assert_eq!(batch.decisions().next().unwrap().0, "npc-0000");
    let mut oversized = configs(Kind::Resources, MAX_AGENTS + 1);
    assert!(Runtime::new(session(Kind::Resources), 0, &oversized).is_err());
    oversized.truncate(MAX_AGENTS);
    let save = saved(&runtime);
    let (restored, _) =
        Runtime::restore(session(Kind::Resources), &oversized, "test-v1", &save).unwrap();
    assert_eq!(save, saved(&restored));
}
