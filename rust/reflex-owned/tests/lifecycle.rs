use reflex_immediate::*;
use reflex_owned::{Feedback, Owner};
use serde_json::json;
mod support;
use support::{config, feedback, input};

#[test]
fn feedback_is_atomic_missing_is_not_failure_and_receipts_are_owned() {
    let mut a = Owner::new(config("a")).unwrap();
    let mut b = Owner::new(config("b")).unwrap();
    let initial = a.state().unwrap().clone();
    let p = a.prepare(&input(0), false).unwrap();
    assert_eq!(a.state().unwrap(), &initial);
    let ticket = a.commit(p, "act").unwrap();
    assert!(a.checkpoint(&json!({})).is_err());
    assert!(a.prepare(&input(1), false).is_err());
    let tb = b
        .commit(b.prepare(&input(0), false).unwrap(), "act")
        .unwrap();
    assert!(b.resolve(ticket, &feedback()).is_err());
    let mut bad = feedback();
    bad.needs.as_mut().unwrap().progress[4] = Some(f64::NAN);
    assert!(a.resolve(ticket, &bad).is_err());
    bad = feedback();
    bad.observed.as_mut().unwrap()[0] = -0.8;
    assert!(a.resolve(ticket, &bad).is_err());
    a.resolve(ticket, &feedback()).unwrap();
    b.resolve(tb, &feedback()).unwrap();
    assert!(a.resolve(ticket, &feedback()).is_err());
    assert_eq!(a.state().unwrap().prediction.observations, 1);
    let before = a.state().unwrap().clone();
    let t = a
        .commit(a.prepare(&input(1), false).unwrap(), "act")
        .unwrap();
    a.resolve(
        t,
        &Feedback {
            observed: None,
            response: None,
            needs: None,
            purpose: None,
        },
    )
    .unwrap();
    assert_eq!(a.state().unwrap().prediction, before.prediction);
    assert_eq!(a.state().unwrap().experience, before.experience);
    assert_eq!(a.state().unwrap().pressure, before.pressure);
}
#[test]
fn fork_save_restore_and_stale_proposals_preserve_one_factual_owner() {
    let a = Owner::new(config("a")).unwrap();
    let mut branch = a.fork_model().unwrap();
    let t = branch
        .commit(branch.prepare(&input(0), false).unwrap(), "act")
        .unwrap();
    branch.resolve(t, &feedback()).unwrap();
    assert_eq!(a.state().unwrap().prediction.observations, 0);
    let old = branch.prepare(&input(1), false).unwrap();
    let world = json!({"tick":1,"rng":17});
    let saved = branch.checkpoint(&world).unwrap();
    let mut restored = Owner::restore(&config("a"), &world, &saved).unwrap();
    assert_eq!(branch.state().unwrap(), restored.state().unwrap());
    assert!(restored.commit(old, "act").is_err());
    assert_eq!(
        branch.prepare(&input(1), false).unwrap().decision(),
        restored.prepare(&input(1), false).unwrap().decision()
    );
    assert!(Owner::restore(&config("b"), &world, &saved).is_err());
    assert!(Owner::restore(&config("a"), &json!({"tick":2,"rng":17}), &saved).is_err());
    let mut changed = config("a");
    changed.predictor.window = 64;
    assert!(Owner::restore(&changed, &world, &saved).is_err());
    let mut value: serde_json::Value = serde_json::from_slice(&saved).unwrap();
    value["payload"]["state"]["prediction"]["observations"] = json!(10);
    assert!(Owner::restore(&config("a"), &world, &serde_json::to_vec(&value).unwrap()).is_err());
}
#[test]
fn conditional_validation_and_long_selection_do_not_change_factual_memory() {
    let a = Owner::new(config("a")).unwrap();
    let mut i = input(0);
    i.responses.models[0][0] = f64::NAN;
    assert!(a.prepare(&i, false).is_err());
    i = input(0);
    i.bindings.get_mut("act").unwrap().estimated = vec![22];
    assert!(a.prepare(&i, false).is_err());
    let stage = a.prepare(&input(0), false).unwrap();
    let forecast = Continuation {
        owner: a.config().actor.clone(),
        state: a.state().unwrap().intent.clone(),
        tick: 0,
        target: "purpose".into(),
        horizon: 32,
        max_regret: 0.15,
        units: ReturnUnits {
            unit: "decisions".into(),
            effects: [32.; 22],
            root_terms: 32.,
        },
        plans: vec![ContinuationPlan {
            root: "act".into(),
            action: stage.context().actions[0].clone(),
            purpose: 0.2,
            correction: Some([0.; 22]),
        }],
    };
    let before = a.state().unwrap().clone();
    let selected = a.with_continuation(stage, &forecast).unwrap();
    assert!(selected.is_planned());
    assert_eq!(a.state().unwrap(), &before);
    assert!(a
        .with_continuation(a.prepare(&input(0), true).unwrap(), &forecast)
        .is_err());
    let mut wrong = forecast;
    wrong.owner.personality[0] = 0.9;
    assert!(a
        .with_continuation(a.prepare(&input(0), false).unwrap(), &wrong)
        .is_err());
}

#[test]
fn impossible_initial_histories_are_rejected_before_use() {
    let initial = Owner::new(config("a")).unwrap().state().unwrap().clone();
    let mut changed = initial.clone();
    changed.progress.stalls = 1;
    assert!(Owner::from_state(config("a"), changed).is_err());
    let mut changed = initial;
    changed.pressure.debt[0] = 0.1;
    assert!(Owner::from_state(config("a"), changed).is_err());
}
