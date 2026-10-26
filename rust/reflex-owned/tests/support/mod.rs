use reflex_immediate::*;
use reflex_owned::{experience::Binding, prediction, progress, Feedback, Input, OwnerConfig};
use std::collections::BTreeMap;

pub fn config(npc: &str) -> OwnerConfig {
    OwnerConfig {
        actor: Config {
            scope: Scope {
                game: "test".into(),
                episode: "one".into(),
                npc: npc.into(),
            },
            seed: 42,
            personality: [0.5; 5],
            values: [0.5; 10],
            objective: "purpose".into(),
            policy: Policy {
                priority: Priority::Personal { rigidity: 0.5 },
                residual: [0.; 8],
            },
        },
        predictor: prediction::Definition {
            model_id: "rules-v1".into(),
            factors: vec![],
            names: vec!["model".into()],
            window: 128,
        },
        contract: "composition-v1".into(),
        purpose_target: "purpose".into(),
    }
}
pub fn input(tick: u64) -> Input {
    let effects = vec![
        Outcome {
            p: 0.5,
            objective: 0.2,
            needs: [0.; 5],
            values: [0.; 10],
            style: [0.; 5],
            cost: 0.,
        },
        Outcome {
            p: 0.5,
            objective: -0.1,
            needs: [0.; 5],
            values: [0.; 10],
            style: [0.; 5],
            cost: 0.,
        },
    ];
    let action = Action {
        id: "act".into(),
        target: None,
        legal: true,
        known_failure: false,
        confidence: 1.,
        familiarity: 0.,
        switch_cost: 0.,
        outcomes: effects.clone(),
    };
    Input {
        observation: Observation {
            tick,
            needs: std::array::from_fn(|_| NeedInput {
                supported: true,
                enabled: true,
                deficit: Some(0.2),
            }),
            facts: BTreeMap::from([("observed".into(), "public".into())]),
            actions: vec![action],
            opponent: None,
        },
        bindings: BTreeMap::from([(
            "act".into(),
            Binding {
                method: "act".into(),
                situation: "condition".into(),
                estimated: (0..22).collect(),
            },
        )]),
        responses: prediction::Inputs {
            context: vec![],
            legal: vec!["a".into(), "b".into()],
            models: vec![vec![0.6, 0.4]],
        },
        conditional: BTreeMap::from([("act".into(), effects)]),
        purpose: progress::Purpose {
            level: 0.,
            readiness: 0.,
            maintained: false,
            activities: BTreeMap::from([(
                "act".into(),
                progress::Activity {
                    kind: progress::Kind::Uncertain,
                    reason: "public attempt".into(),
                    condition: "condition".into(),
                    patience: 3,
                    release: String::new(),
                    replacement: true,
                },
            )]),
        },
        threatened: true,
        maintains_advantage: false,
    }
}
pub fn feedback() -> Feedback {
    let mut observed = [0.; 22];
    observed[0] = 0.2;
    Feedback {
        observed: Some(observed),
        response: Some("a".into()),
        needs: Some(progress::NeedFeedback {
            progress: [Some(0.); 5],
            maintained: [false; 5],
            completed: [false; 5],
        }),
        purpose: Some(progress::PurposeFeedback {
            level: 0.,
            readiness: 0.,
            maintained: false,
            completed: false,
        }),
    }
}
