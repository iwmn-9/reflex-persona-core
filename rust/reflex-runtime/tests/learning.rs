use reflex_immediate::{Continuation, ContinuationPlan, ReturnUnits};
use reflex_owned::{Input, Owner, Stage};
use reflex_runtime::learning::{Control, LearningSession, Receipt, World};
use reflex_runtime::{Choice, Execution, Result, SessionId};
use serde::{Deserialize, Serialize};
#[path = "../../reflex-owned/tests/support/mod.rs"]
mod support;

#[derive(Clone, Serialize, Deserialize)]
struct Simulation {
    tick: u64,
    bad_last: bool,
    cancel_first: bool,
}
impl World for Simulation {
    const SAVE_FORMAT: &'static str = "learning-test-v1";
    fn revision(&self) -> u64 {
        self.tick
    }
    fn observe(&self, _: &Owner) -> Result<Option<Input>> {
        Ok(Some(support::input(self.tick)))
    }
    fn forecast(&self, owner: &Owner, stage: &Stage) -> Result<Continuation> {
        Ok(Continuation {
            owner: owner.config().actor.clone(),
            state: owner.state().unwrap().intent.clone(),
            tick: self.tick,
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
        })
    }
    fn advance(&self, choices: &[Choice<'_>]) -> Result<(Self, Vec<Receipt>)> {
        let mut next = self.clone();
        next.tick += 1;
        let receipts = choices
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut fb = support::feedback();
                if self.bad_last && i == choices.len() - 1 {
                    fb.response = Some("outside rules".into());
                }
                if self.cancel_first && i == 0 {
                    Receipt {
                        npc: c.npc.into(),
                        execution: Execution::NotExecuted {
                            reason: "occupied".into(),
                        },
                        feedback: None,
                    }
                } else {
                    Receipt {
                        npc: c.npc.into(),
                        execution: Execution::Executed {
                            action_id: c.decision.action_id.clone(),
                            target: c.decision.target.clone(),
                        },
                        feedback: Some(fb),
                    }
                }
            })
            .collect();
        Ok((next, receipts))
    }
}
fn session(control: Control, bad_last: bool, cancel_first: bool) -> LearningSession<Simulation> {
    LearningSession::new(
        SessionId {
            game: "test".into(),
            episode: "one".into(),
        },
        control,
        &[support::config("a"), support::config("b")],
        Simulation {
            tick: 0,
            bad_last,
            cancel_first,
        },
    )
    .unwrap()
}
#[test]
fn a_bad_last_feedback_rolls_back_every_owner_and_the_world() {
    for control in [Control::Immediate, Control::Fixed32] {
        let mut s = session(control, true, false);
        let before = s.checkpoint().unwrap();
        assert!(s.step(false).is_err());
        assert_eq!(s.checkpoint().unwrap(), before);
        assert_eq!(s.world().tick, 0);
        assert_eq!(
            s.actor("a")
                .unwrap()
                .state()
                .unwrap()
                .prediction
                .observations,
            0
        );
    }
}
#[test]
fn cancelled_actors_do_not_learn_and_one_save_restores_the_entire_session() {
    let mut s = session(Control::Fixed32, false, true);
    let initial = s.checkpoint().unwrap();
    assert!(s.step(true).is_err());
    assert_eq!(s.checkpoint().unwrap(), initial);
    for _ in 0..8 {
        s.step(false).unwrap();
    }
    assert_eq!(s.actor("a").unwrap().state().unwrap().last_tick, None);
    assert_eq!(
        s.actor("b")
            .unwrap()
            .state()
            .unwrap()
            .prediction
            .observations,
        8
    );
    let bytes = s.checkpoint().unwrap();
    let identity = SessionId {
        game: "test".into(),
        episode: "one".into(),
    };
    let configs = [support::config("a"), support::config("b")];
    let mut restored = LearningSession::<Simulation>::restore(
        identity.clone(),
        Control::Fixed32,
        &configs,
        &bytes,
    )
    .unwrap();
    for _ in 0..8 {
        assert_eq!(
            serde_json::to_value(s.step(false).unwrap()).unwrap(),
            serde_json::to_value(restored.step(false).unwrap()).unwrap()
        );
        assert_eq!(s.checkpoint().unwrap(), restored.checkpoint().unwrap());
    }
    assert!(
        LearningSession::<Simulation>::restore(identity, Control::Immediate, &configs, &bytes)
            .is_err()
    );
}
