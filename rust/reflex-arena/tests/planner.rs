#[allow(dead_code)]
#[path = "../../reflex-owned/tests/support/mod.rs"]
mod fixture;
use reflex_arena::{
    games::{Combat, Encounters, Hagetaka},
    planner::{Config, Planner},
    ArenaGame,
};
use reflex_owned::Owner;
fn check<G: ArenaGame>(game: G) {
    let mut cfg = fixture::config("a");
    cfg.contract = "arena-estimated-model-v1".into();
    cfg.predictor.names = vec!["0".into(), "1".into(), "2".into()];
    let mut actor = Owner::new(cfg).unwrap();
    let initial = actor.state().unwrap().clone();
    let mut planner = Planner::new(
        &actor,
        Config {
            horizon: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let proposal = planner.propose(&actor, &game, 0).unwrap();
    assert_eq!(actor.state().unwrap(), &initial);
    let action = proposal.action().to_owned();
    let mut choices = game.modeled_choices(0, "0").unwrap();
    choices[0] = action.clone();
    let (next, feedback) = game.advance(&choices, 987).unwrap();
    planner
        .resolve(&mut actor, proposal, &action, &feedback[0], next.purpose(0))
        .unwrap();
    assert_eq!(actor.state().unwrap().last_tick, Some(game.tick()));
    let bytes = planner.checkpoint().unwrap();
    let restored = Planner::restore(&actor, &bytes).unwrap();
    assert_eq!(restored.config().horizon, 2);
    let mut damaged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    damaged["planner"]["config"]["horizon"] = 3.into();
    assert!(Planner::restore(&actor, &serde_json::to_vec(&damaged).unwrap()).is_err());
    assert!(Planner::restore(&Owner::new(actor.config().clone()).unwrap(), &bytes).is_err());
    let mut divergent = actor.state().unwrap().clone();
    divergent.intent.mode_urgency = if divergent.intent.mode_urgency < 0.5 {
        0.75
    } else {
        0.25
    };
    let different_history = Owner::from_state(actor.config().clone(), divergent).unwrap();
    assert!(Planner::restore(&different_history, &bytes).is_err());
    let mut wrong = actor.config().clone();
    wrong.actor.personality[0] = 0.9;
    assert!(Planner::restore(&Owner::new(wrong).unwrap(), &bytes).is_err());
}
#[test]
fn three_games_plan_and_resolve() {
    check(Hagetaka::new(7));
    check(Encounters::new(7, 10));
    check(Combat::new(7, "open", "elimination", 10));
}

#[test]
fn rejected_execution_keeps_actor_and_planner_unchanged() {
    let game = Hagetaka::new(7);
    let mut cfg = fixture::config("a");
    cfg.contract = "arena-estimated-model-v1".into();
    cfg.predictor.names = vec!["0".into(), "1".into(), "2".into()];
    let mut actor = Owner::new(cfg).unwrap();
    let mut planner = Planner::new(
        &actor,
        Config {
            horizon: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let owner_before = actor.state().unwrap().clone();
    let planner_before = planner.checkpoint().unwrap();
    let proposal = planner.propose(&actor, &game, 0).unwrap();
    assert!(planner
        .resolve(
            &mut actor,
            proposal,
            "wrong-action",
            &fixture::feedback(),
            game.purpose(0)
        )
        .is_err());
    assert_eq!(actor.state().unwrap(), &owner_before);
    assert_eq!(planner.checkpoint().unwrap(), planner_before);
}

#[test]
fn ordinary_only_runs_three_immediate_tails_per_legal_root() {
    let game = Hagetaka::new(7);
    let mut cfg = fixture::config("a");
    cfg.contract = "arena-estimated-model-v1".into();
    cfg.predictor.names = vec!["0".into(), "1".into(), "2".into()];
    let actor = Owner::new(cfg).unwrap();
    let config = Config {
        horizon: 3,
        second_moves: 0,
        ..Default::default()
    };
    let planner = Planner::new(&actor, config).unwrap();
    let input = game.input(0).unwrap();
    let roots = actor
        .prepare_estimated(&input, false)
        .unwrap()
        .allowed()
        .len();
    let before = actor.state().unwrap().clone();
    let proposal = planner.propose(&actor, &game, 0).unwrap();
    assert_eq!(proposal.audit.simulated_nodes, roots * 3 * 3);
    assert_eq!(actor.state().unwrap(), &before);
    let bytes = planner.checkpoint().unwrap();
    assert_eq!(
        Planner::restore(&actor, &bytes)
            .unwrap()
            .config()
            .second_moves,
        0
    );
    let mut old: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    old["version"] = "arena-planner-v2".into();
    assert!(Planner::restore(&actor, &serde_json::to_vec(&old).unwrap()).is_err());
}

#[test]
fn default_second_move_budget_matches_explicit_eight_and_is_bounded() {
    let game = Hagetaka::new(7);
    let mut cfg = fixture::config("a");
    cfg.contract = "arena-estimated-model-v1".into();
    cfg.predictor.names = vec!["0".into(), "1".into(), "2".into()];
    let actor = Owner::new(cfg).unwrap();
    let implicit = Planner::new(
        &actor,
        Config {
            horizon: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let explicit = Planner::new(
        &actor,
        Config {
            horizon: 2,
            second_moves: 8,
            ..Default::default()
        },
    )
    .unwrap();
    let a = implicit.propose(&actor, &game, 0).unwrap();
    let b = explicit.propose(&actor, &game, 0).unwrap();
    assert_eq!(a.action(), b.action());
    assert_eq!(
        serde_json::to_value(a.audit).unwrap(),
        serde_json::to_value(b.audit).unwrap()
    );
    assert!(Planner::new(
        &actor,
        Config {
            second_moves: 9,
            ..Default::default()
        }
    )
    .is_err());
}

#[test]
fn multiple_scenario_batches_are_bounded_replayable_and_leave_the_owner_untouched() {
    let game = Hagetaka::new(7);
    let mut cfg = fixture::config("a");
    cfg.contract = "arena-estimated-model-v1".into();
    cfg.predictor.names = vec!["0".into(), "1".into(), "2".into()];
    let actor = Owner::new(cfg).unwrap();
    let before = actor.state().unwrap().clone();
    let input = game.input(0).unwrap();
    let roots = actor
        .prepare_estimated(&input, false)
        .unwrap()
        .allowed()
        .len();
    let planner = Planner::new(
        &actor,
        Config {
            horizon: 2,
            second_moves: 0,
            scenario_batches: 3,
            ..Default::default()
        },
    )
    .unwrap();
    let proposal = planner.propose(&actor, &game, 0).unwrap();
    assert_eq!(proposal.audit.simulated_nodes, roots * 9 * 2);
    assert_eq!(actor.state().unwrap(), &before);
    let checkpoint = planner.checkpoint().unwrap();
    let restored = Planner::restore(&actor, &checkpoint).unwrap();
    assert_eq!(restored.config().scenario_batches, 3);
    let replay = restored.propose(&actor, &game, 0).unwrap();
    assert_eq!(proposal.action(), replay.action());
    assert_eq!(
        serde_json::to_value(&proposal.audit).unwrap(),
        serde_json::to_value(&replay.audit).unwrap()
    );
    let max = Planner::new(
        &actor,
        Config {
            horizon: 2,
            second_moves: 0,
            scenario_batches: 4,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        max.propose(&actor, &game, 0).unwrap().audit.simulated_nodes,
        roots * 12 * 2
    );
    let mut old: serde_json::Value = serde_json::from_slice(&checkpoint).unwrap();
    old["version"] = "arena-planner-v3".into();
    assert!(Planner::restore(&actor, &serde_json::to_vec(&old).unwrap()).is_err());
    for count in [0, 5] {
        assert!(Planner::new(
            &actor,
            Config {
                scenario_batches: count,
                ..Default::default()
            }
        )
        .is_err());
    }
}

#[test]
fn default_scenario_batch_matches_explicit_one() {
    let game = Hagetaka::new(7);
    let mut cfg = fixture::config("a");
    cfg.contract = "arena-estimated-model-v1".into();
    cfg.predictor.names = vec!["0".into(), "1".into(), "2".into()];
    let actor = Owner::new(cfg).unwrap();
    let a = Planner::new(
        &actor,
        Config {
            horizon: 2,
            second_moves: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let b = Planner::new(
        &actor,
        Config {
            horizon: 2,
            second_moves: 0,
            scenario_batches: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let first = a.propose(&actor, &game, 0).unwrap();
    let second = b.propose(&actor, &game, 0).unwrap();
    assert_eq!(first.action(), second.action());
    assert_eq!(
        serde_json::to_value(&first.audit).unwrap(),
        serde_json::to_value(&second.audit).unwrap()
    );
}

#[test]
fn agreement_evaluation_uses_every_batch_without_mutating_factual_memory() {
    use reflex_arena::behavior::Beliefs;
    let game = Encounters::new(7, 5);
    let mut cfg = fixture::config("a");
    cfg.contract = "arena-estimated-model-v1".into();
    cfg.predictor.names = vec!["0".into(), "1".into(), "2".into()];
    let actor = Owner::new(cfg).unwrap();
    let planner = Planner::new(
        &actor,
        Config {
            horizon: 1,
            scenario_batches: 3,
            ..Default::default()
        },
    )
    .unwrap();
    let scope = serde_json::to_string(&(
        &actor.config().actor.scope.game,
        &actor.config().actor.scope.episode,
    ))
    .unwrap();
    let beliefs = Beliefs::new(scope, 0, game.behavior(0, 0).unwrap().alphabet).unwrap();
    let owner_before = actor.state().unwrap().clone();
    let beliefs_before = beliefs.checkpoint().unwrap();
    let announced = vec!["rest".to_owned(); game.players()];
    let offers = game.offers(&announced).unwrap();
    let relevant = offers.iter().filter(|o| o.roots.contains_key(&0)).count();
    let consent = planner
        .consent_allow_indifference(&actor, &game, 0, &beliefs, &announced, &offers)
        .unwrap();
    assert_eq!(consent.simulated_nodes, (relevant + 1) * 9);
    assert_eq!(actor.state().unwrap(), &owner_before);
    assert_eq!(beliefs.checkpoint().unwrap(), beliefs_before);
}

#[test]
fn unfiltered_tier_audit_matches_selected_root_when_purpose_band_covers_all_roots() {
    let game = Hagetaka::new(7);
    let mut cfg = fixture::config("a");
    cfg.contract = "arena-estimated-model-v1".into();
    cfg.predictor.names = vec!["0".into(), "1".into(), "2".into()];
    let actor = Owner::new(cfg).unwrap();
    let planner = Planner::new(
        &actor,
        Config {
            horizon: 2,
            second_moves: 0,
            max_regret: 1.,
            ..Default::default()
        },
    )
    .unwrap();
    let before = actor.state().unwrap().clone();
    let proposal = planner.propose(&actor, &game, 0).unwrap();
    assert_eq!(proposal.audit.pre_filter_action, proposal.action());
    assert_eq!(
        proposal.audit.pre_filter_purpose,
        proposal.audit.modeled_purpose
    );
    assert!((-1.0..=1.0).contains(&proposal.audit.pre_filter_purpose));
    assert_eq!(actor.state().unwrap(), &before);
}

#[test]
fn agreement_proposals_carry_the_same_valid_prefilter_selection() {
    use reflex_arena::behavior::Beliefs;
    let game = Encounters::new(7, 5);
    let owners: Vec<_> = (0..game.players())
        .map(|i| {
            let mut cfg = fixture::config(&format!("p{i}"));
            cfg.contract = "arena-estimated-model-v1".into();
            cfg.predictor.names = vec!["0".into(), "1".into(), "2".into()];
            Owner::new(cfg).unwrap()
        })
        .collect();
    let planners: Vec<_> = owners
        .iter()
        .map(|o| {
            Planner::new(
                o,
                Config {
                    horizon: 1,
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect();
    let beliefs: Vec<_> = owners
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let scope = serde_json::to_string(&(
                &o.config().actor.scope.game,
                &o.config().actor.scope.episode,
            ))
            .unwrap();
            Beliefs::new(scope, i, game.behavior(i, i).unwrap().alphabet).unwrap()
        })
        .collect();
    let announced: Vec<_> = owners
        .iter()
        .enumerate()
        .map(|(i, o)| {
            planners[i]
                .propose_with_beliefs(o, &game, i, &beliefs[i])
                .unwrap()
                .action()
                .to_owned()
        })
        .collect();
    let offers = game.offers(&announced).unwrap();
    let mut proposals = 0;
    for i in 0..game.players() {
        let consent = planners[i]
            .consent(&owners[i], &game, i, &beliefs[i], &announced, &offers)
            .unwrap();
        let mut common = None;
        for proposal in consent.proposals.values() {
            assert!(!proposal.audit.pre_filter_action.is_empty());
            assert!((-1.0..=1.0).contains(&proposal.audit.pre_filter_purpose));
            let choice = (
                &proposal.audit.pre_filter_action,
                proposal.audit.pre_filter_purpose,
            );
            if let Some(prior) = &common {
                assert_eq!(*prior, choice);
            } else {
                common = Some(choice);
            }
            proposals += 1;
        }
    }
    assert!(
        proposals > 0,
        "test must exercise at least one signed proposal"
    );
}
