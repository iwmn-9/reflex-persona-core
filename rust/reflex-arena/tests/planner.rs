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
