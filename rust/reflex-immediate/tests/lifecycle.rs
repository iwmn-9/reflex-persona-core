use reflex_immediate::*;
use std::collections::BTreeMap;

fn config() -> Config {
    Config {
        scope: Scope {
            game: "test".into(),
            episode: "persistent".into(),
            npc: "人α".into(),
        },
        seed: 42,
        personality: [0.5; 5],
        values: [0.2; 10],
        objective: "survive".into(),
        policy: Policy::default(),
    }
}
fn observation(tick: u64) -> Observation {
    Observation {
        tick,
        needs: std::array::from_fn(|_| NeedInput {
            supported: true,
            enabled: true,
            deficit: Some(0.3),
        }),
        facts: BTreeMap::from([("seen".into(), "public observation".into())]),
        actions: ["wait", "work"]
            .iter()
            .map(|key| Action {
                id: (*key).into(),
                target: None,
                legal: true,
                known_failure: false,
                confidence: 0.8,
                familiarity: 0.4,
                switch_cost: 0.02,
                outcomes: vec![Outcome {
                    p: 1.,
                    objective: 0.2,
                    needs: [0.1; 5],
                    values: [0.1; 10],
                    style: [0.; 5],
                    cost: 0.,
                }],
            })
            .collect(),
        opponent: None,
    }
}
fn execute(a: &mut Agent, o: &Observation) -> Decision {
    let p = a.preview(o, true).unwrap();
    let key = p.decision().action_id.clone();
    a.commit(p, &key).unwrap()
}

#[test]
fn reference_reduction_does_not_reverse_nearly_tied_outcome_permutations() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("data/reduction.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let request = &case["request"];
        let config: Config = serde_json::from_value(request["config"].clone()).unwrap();
        let state: State = serde_json::from_value(request["state"].clone()).unwrap();
        let observation: Observation =
            serde_json::from_value(request["observation"].clone()).unwrap();
        let agent = Agent::from_state(config, state, None).unwrap();
        let p = agent.preview(&observation, false).unwrap();
        assert_eq!(p.decision().action_id, case["action"].as_str().unwrap());
        for (actual, expected) in p
            .decision()
            .scores
            .iter()
            .zip(case["scores"].as_array().unwrap())
        {
            assert_eq!(actual.score, expected.as_f64().unwrap());
        }
    }
}

#[test]
fn preview_is_pure_commit_is_bound_and_cannot_be_replayed() {
    let c = config();
    let mut a = Agent::new(c.clone()).unwrap();
    let saved = a.checkpoint().unwrap();
    let p = a.preview(&observation(1), true).unwrap();
    let second = a.preview(&observation(1), true).unwrap();
    let key = p.decision().action_id.clone();
    assert_eq!(p.decision(), second.decision());
    assert_eq!(saved, a.checkpoint().unwrap());
    let mut other = c;
    other.scope.npc = "another".into();
    let mut other = Agent::new(other).unwrap();
    assert!(other.commit(second, &key).is_err());
    let wrong = a.preview(&observation(1), true).unwrap();
    assert!(a.commit(wrong, "not-executed").is_err());
    assert_eq!(saved, a.checkpoint().unwrap());
    let stale = a.preview(&observation(2), true).unwrap();
    a.commit(p, &key).unwrap();
    let after = a.checkpoint().unwrap();
    assert!(a.commit(stale, &key).is_err());
    assert!(a.preview(&observation(1), true).is_err());
    assert_eq!(after, a.checkpoint().unwrap());
}

#[test]
fn save_restore_continues_bit_exactly_for_512_decisions() {
    let c = config();
    let mut whole = Agent::new(c.clone()).unwrap();
    let mut resumed = whole.clone();
    for tick in 0..512 {
        let mut o = observation(tick);
        o.needs[4].deficit = Some((tick % 17) as f64 / 17.);
        o.actions[0].outcomes[0].objective = (tick % 11) as f64 / 11.;
        assert_eq!(execute(&mut whole, &o), execute(&mut resumed, &o));
        let bytes = resumed.checkpoint().unwrap();
        resumed = Agent::restore(&c, &bytes).unwrap();
        assert_eq!(bytes, resumed.checkpoint().unwrap());
    }
    assert_eq!(whole.checkpoint().unwrap(), resumed.checkpoint().unwrap());
}

#[test]
fn snapshot_rejects_changed_character_goal_policy_version_corruption_and_unknown_fields() {
    let c = config();
    let mut a = Agent::new(c.clone()).unwrap();
    execute(&mut a, &observation(0));
    let bytes = a.checkpoint().unwrap();
    for field in 0..6 {
        let mut wrong = c.clone();
        match field {
            0 => wrong.scope.npc = "different".into(),
            1 => wrong.personality[0] = 0.8,
            2 => wrong.values[0] = 0.8,
            3 => wrong.objective = "different".into(),
            4 => wrong.seed = 10,
            _ => wrong.policy.priority = Priority::Finite,
        }
        assert!(Agent::restore(&wrong, &bytes).is_err());
    }
    for field in 0..3 {
        let mut changed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        match field {
            0 => changed["payload"]["state"]["age"] = 400.into(),
            1 => changed["payload"]["version"] = "future-v99".into(),
            _ => changed["ignored"] = true.into(),
        }
        assert!(Agent::restore(&c, &serde_json::to_vec(&changed).unwrap()).is_err());
    }
    assert!(Agent::restore(&c, &bytes[..bytes.len() - 1]).is_err());
    assert!(Agent::restore(&c, &vec![b' '; MAX_SNAPSHOT_BYTES + 1]).is_err());
}

#[test]
fn invalid_observations_fail_without_advancing_or_panicking() {
    let a = Agent::new(config()).unwrap();
    let saved = a.checkpoint().unwrap();
    for case in 0..16 {
        let mut o = observation(0);
        match case {
            0 => o.actions.clear(),
            1 => o.actions.push(o.actions[0].clone()),
            2 => o.actions[0].confidence = f64::NAN,
            3 => o.actions[0].outcomes[0].p = 0.5,
            4 => o.actions[0].outcomes.clear(),
            5 => o.actions[0].outcomes = vec![o.actions[0].outcomes[0].clone(); 9],
            6 => o.actions.iter_mut().for_each(|a| a.known_failure = true),
            7 => o.actions.iter_mut().for_each(|a| a.legal = false),
            8 => o.needs[0].supported = false,
            9 => o.needs[0].enabled = false,
            10 => o.needs[0].deficit = None,
            11 => o.facts.clear(),
            12 => o.actions[0].outcomes[0].style[4] = f64::INFINITY,
            13 => o.tick = 1 << 63,
            14 => o.actions[0].id = "字".repeat(161),
            _ => o.actions[0].outcomes[0].cost = -0.1,
        }
        assert!(a.preview(&o, true).is_err(), "case {case}");
        assert_eq!(saved, a.checkpoint().unwrap());
    }
}

#[test]
fn typed_and_json_boundaries_reject_invalid_configuration_and_extra_capabilities() {
    for case in 0..5 {
        let mut c = config();
        match case {
            0 => c.policy.priority = Priority::Personal { rigidity: f64::NAN },
            1 => c.policy.residual[3] = 0.09,
            2 => c.personality[0] = -0.01,
            3 => c.values[1] = f64::NEG_INFINITY,
            _ => c.seed = 1 << 63,
        }
        assert!(Agent::new(c).is_err());
    }
    let mut o = serde_json::to_value(observation(0)).unwrap();
    o["return_frame"] = serde_json::json!({"scale": 4});
    assert!(Observation::from_json(&serde_json::to_vec(&o).unwrap()).is_err());
    o.as_object_mut().unwrap().remove("return_frame");
    o["actions"][0]["legal"] = 1.into();
    assert!(Observation::from_json(&serde_json::to_vec(&o).unwrap()).is_err());
    assert!(Observation::from_json(&vec![b' '; MAX_JSON_BYTES + 1]).is_err());
    assert!(Config::from_json(br#"{"unknown":true}"#).is_err());
    let bad_state = State {
        age: 1_000_001,
        ..State::default()
    };
    assert!(Agent::from_state(config(), bad_state, None).is_err());
}

#[test]
fn readings_require_current_observed_evidence_and_known_actions() {
    let a = Agent::new(config()).unwrap();
    let mut o = observation(0);
    o.opponent = Some(Reading {
        confidence: 0.8,
        uncertainty: 0.1,
        threatened: true,
        maintains_advantage: true,
        evidence_ids: vec!["unseen".into()],
        deltas: BTreeMap::from([("wait".into(), 0.2)]),
    });
    assert!(a.preview(&o, false).is_err());
    o.opponent.as_mut().unwrap().evidence_ids = vec!["seen".into()];
    assert!(a.preview(&o, false).unwrap().decision().reading_used);
    o.opponent
        .as_mut()
        .unwrap()
        .deltas
        .insert("missing".into(), 0.1);
    assert!(a.preview(&o, false).is_err());
}

#[test]
fn changing_need_availability_requires_explicit_intent_reset() {
    let mut a = Agent::new(config()).unwrap();
    execute(&mut a, &observation(0));
    let pending = a.preview(&observation(1), false).unwrap();
    let key = pending.decision().action_id.clone();
    let mut o = observation(1);
    o.needs[a.state().primary_need.unwrap() as usize] = NeedInput {
        supported: false,
        enabled: false,
        deficit: None,
    };
    assert!(a.preview(&o, false).is_err());
    a.reset_intent();
    assert!(a.commit(pending, &key).is_err());
    assert!(a.preview(&o, false).is_ok());
    assert_eq!(a.config(), &config());
    assert_eq!(a.last_tick(), Some(0));
}

#[test]
fn candidate_order_batch_order_and_threads_do_not_share_random_state() {
    let agents: Vec<_> = (0..16)
        .map(|i| {
            let mut c = config();
            c.scope.npc = format!("npc-{i}");
            Agent::new(c).unwrap()
        })
        .collect();
    let o = observation(77);
    let expected: Vec<_> = agents
        .iter()
        .map(|a| a.preview(&o, true).unwrap().decision().clone())
        .collect();
    let mut reversed = o.clone();
    reversed.actions.reverse();
    let items: Vec<_> = agents.iter().rev().map(|a| (a, &reversed)).collect();
    let batch = preview_batch(&items, true).unwrap();
    assert_eq!(
        batch
            .iter()
            .rev()
            .map(|p| p.decision().clone())
            .collect::<Vec<_>>(),
        expected
    );
    std::thread::scope(|s| {
        let handles: Vec<_> = agents
            .iter()
            .map(|a| {
                let o = &o;
                s.spawn(move || a.preview(o, true).unwrap().decision().clone())
            })
            .collect();
        assert_eq!(
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>(),
            expected
        );
    });
}

#[test]
fn maximum_capacity_and_unicode_ids_are_accepted_but_overflow_is_not() {
    let a = Agent::new(config()).unwrap();
    let mut o = observation(0);
    let template = o.actions[0].clone();
    o.actions = (0..256)
        .map(|i| {
            let mut a = template.clone();
            a.id = format!("行動-{i:03}");
            let mut e = a.outcomes[0].clone();
            e.p = 0.125;
            a.outcomes = vec![e; 8];
            a
        })
        .collect();
    assert_eq!(
        a.preview(&o, false).unwrap().decision().action_id,
        "行動-000"
    );
    o.actions.push(template);
    assert!(a.preview(&o, false).is_err());
}
