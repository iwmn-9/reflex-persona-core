//! Reappraise existing scenario batches without drawing new world samples.
use reflex_immediate::{Agent, Config, Continuation, Error, Observation, Result, State};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Support {
    pub supported_batches: usize,
    pub total_batches: usize,
    pub supported: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn check(
    config: &Config,
    state: &State,
    last_tick: Option<u64>,
    observation: &Observation,
    allowed: &BTreeSet<String>,
    forecast: &Continuation,
    selected_plan_id: &str,
    base_plan_id: &str,
    batch_size: usize,
) -> Result<Support> {
    check_with_appraisal(
        config,
        state,
        last_tick,
        observation,
        allowed,
        forecast,
        selected_plan_id,
        base_plan_id,
        batch_size,
        &BTreeMap::new(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn check_with_appraisal(
    config: &Config,
    state: &State,
    last_tick: Option<u64>,
    observation: &Observation,
    allowed: &BTreeSet<String>,
    forecast: &Continuation,
    selected_plan_id: &str,
    base_plan_id: &str,
    batch_size: usize,
    appraisal: &BTreeMap<String, f64>,
) -> Result<Support> {
    if appraisal.values().any(|v| !v.is_finite() || v.abs() > 1.5)
        || appraisal
            .keys()
            .any(|id| !forecast.plans.iter().any(|p| &p.action.id == id))
    {
        return Err(Error::Invalid("valid plan appraisal required"));
    }
    let actor = Agent::from_state(config.clone(), state.clone(), last_tick)?;
    let mut full = forecast.clone();
    full.max_regret = 2.;
    // Validate all original probabilities, identities, root eligibility and bindings first.
    actor.preview_continuation(observation, &full, allowed)?;
    let selected = full
        .plans
        .iter()
        .find(|p| p.action.id == selected_plan_id)
        .ok_or(Error::Invalid("selected plan id required"))?;
    let base = full
        .plans
        .iter()
        .find(|p| p.action.id == base_plan_id)
        .ok_or(Error::Invalid("base plan id required"))?;
    if !allowed.contains(&selected.root) || !allowed.contains(&base.root) {
        return Err(Error::Invalid("selected and base roots must be allowed"));
    }
    let probs: Vec<_> = selected.action.outcomes.iter().map(|o| o.p).collect();
    if batch_size == 0
        || probs.is_empty()
        || probs.len() % batch_size != 0
        || full.plans.iter().any(|p| {
            p.correction.is_some()
                || p.action.outcomes.len() != probs.len()
                || p.action.outcomes.iter().zip(&probs).any(|(o, p)| o.p != *p)
        })
    {
        return Err(Error::Invalid(
            "matching uncorrected scenario batches required",
        ));
    }
    let total_batches = probs.len() / batch_size;
    let mut support = Support {
        supported_batches: 0,
        total_batches,
        supported: false,
    };
    if total_batches < 2 {
        return Ok(support);
    }
    for batch in 0..total_batches {
        let start = batch * batch_size;
        let end = start + batch_size;
        let mass: f64 = probs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < start || *i >= end)
            .map(|(_, p)| p)
            .sum();
        if !mass.is_finite() || mass <= 0. {
            return Err(Error::Invalid(
                "remaining batch probability must be positive",
            ));
        }
        let mut reduced = full.clone();
        for plan in &mut reduced.plans {
            plan.action.outcomes = plan
                .action
                .outcomes
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < start || *i >= end)
                .map(|(_, row)| {
                    let mut row = row.clone();
                    row.p /= mass;
                    row
                })
                .collect();
        }
        let prepared = actor.preview_continuation(observation, &reduced, allowed)?;
        let scores = &prepared.decision().scores;
        let selected = scores
            .iter()
            .find(|s| s.id == selected_plan_id)
            .ok_or(Error::Invalid("selected plan score missing"))?;
        let base = scores
            .iter()
            .find(|s| s.id == base_plan_id)
            .ok_or(Error::Invalid("base plan score missing"))?;
        let selected_score =
            selected.score + appraisal.get(selected_plan_id).copied().unwrap_or(0.);
        let base_score = base.score + appraisal.get(base_plan_id).copied().unwrap_or(0.);
        let social_floor_ok = if appraisal.values().any(|v| *v != 0.) {
            let baseline_id = prepared
                .selected_plan()
                .ok_or(Error::Invalid("batch baseline required"))?;
            let purpose = |id: &str| {
                reduced
                    .plans
                    .iter()
                    .find(|p| p.action.id == id)
                    .map(|p| p.purpose)
                    .ok_or(Error::Invalid("batch purpose required"))
            };
            let best = scores
                .iter()
                .filter(|s| s.eligible)
                .map(|s| purpose(&s.id))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max);
            purpose(selected_plan_id)? >= purpose(baseline_id)?.min(best - 0.15) - 1e-9
        } else {
            true
        };
        if selected.eligible
            && social_floor_ok
            && (!base.eligible || selected_score + 1e-12 >= base_score)
        {
            support.supported_batches += 1;
        }
    }
    support.supported = support.supported_batches == support.total_batches;
    Ok(support)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reflex_immediate::{Action, ContinuationPlan, Mode, Outcome, Priority, ReturnUnits};

    #[allow(dead_code)]
    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../reflex-owned/tests/support/mod.rs"
        ));
    }

    fn data(
        candidate: &[f64],
        base: &[f64],
    ) -> (Config, State, Observation, BTreeSet<String>, Continuation) {
        let mut config = fixture::config("actor").actor;
        config.policy.priority = Priority::Finite;
        let state = State::default();
        let mut obs = fixture::input(0).observation;
        let mut second = obs.actions[0].clone();
        second.id = "other".into();
        obs.actions.push(second);
        let make = |id: &str, root: &str, samples: &[f64]| ContinuationPlan {
            root: root.into(),
            action: Action {
                id: id.into(),
                target: None,
                legal: true,
                known_failure: false,
                confidence: 1.,
                familiarity: 0.,
                switch_cost: 0.,
                outcomes: samples
                    .iter()
                    .map(|x| Outcome {
                        p: 1. / samples.len() as f64,
                        objective: *x,
                        needs: [0.; 5],
                        values: [0.; 10],
                        style: [0.; 5],
                        cost: 0.,
                    })
                    .collect(),
            },
            purpose: 0.,
            correction: None,
        };
        let forecast = Continuation {
            owner: config.clone(),
            state: state.clone(),
            tick: 0,
            target: "purpose".into(),
            horizon: 1,
            max_regret: 0.,
            units: ReturnUnits {
                unit: "turn".into(),
                effects: [1.; 22],
                root_terms: 1.,
            },
            plans: vec![
                make("chosen", "act", candidate),
                make("base", "other", base),
            ],
        };
        (
            config,
            state,
            obs,
            BTreeSet::from(["act".into(), "other".into()]),
            forecast,
        )
    }

    #[test]
    fn one_sample_can_reverse_and_all_batches_can_support() {
        let (c, s, o, a, f) = data(&[0.8, 0.8, -0.8], &[0.2, 0.2, 0.2]);
        let before = (s.clone(), f.clone());
        let r = check(&c, &s, None, &o, &a, &f, "chosen", "base", 1).unwrap();
        assert_eq!(r.total_batches, 3);
        assert!(!r.supported && r.supported_batches < 3);
        assert_eq!((s.clone(), f.clone()), before);
        let (c, s, o, a, f) = data(&[0.8, 0.7, 0.6], &[0.2, 0.2, 0.2]);
        let r = check(&c, &s, None, &o, &a, &f, "chosen", "base", 1).unwrap();
        assert_eq!(
            r,
            Support {
                supported_batches: 3,
                total_batches: 3,
                supported: true
            }
        );
    }

    #[test]
    fn appraisal_is_applied_to_each_batch_without_changing_tier() {
        let (c, s, o, a, f) = data(&[0.1, 0.1, 0.1], &[0.2, 0.2, 0.2]);
        assert!(
            !check(&c, &s, None, &o, &a, &f, "chosen", "base", 1)
                .unwrap()
                .supported
        );
        let bias = BTreeMap::from([("chosen".into(), 0.3)]);
        assert!(
            check_with_appraisal(&c, &s, None, &o, &a, &f, "chosen", "base", 1, &bias)
                .unwrap()
                .supported
        );
    }

    #[test]
    fn persona_tier_precedes_score() {
        let (mut c, mut s, o, a, mut f) = data(&[-0.8, -0.8], &[0.8, 0.8]);
        c.policy.priority = Priority::Lexicographic;
        c.values = [0.; 10];
        c.values[5] = 1.;
        s.mode = Some(Mode::Principle);
        s.primary_need = Some(reflex_immediate::Need::Physiology);
        s.mode_urgency = 0.2;
        f.owner = c.clone();
        f.state = s.clone();
        for x in &mut f.plans[0].action.outcomes {
            x.values[5] = 0.5;
        }
        let agent = Agent::from_state(c.clone(), s.clone(), None).unwrap();
        f.max_regret = 2.;
        let preview = agent.preview_continuation(&o, &f, &a).unwrap();
        let chosen = preview
            .decision()
            .scores
            .iter()
            .find(|x| x.id == "chosen")
            .unwrap();
        let base = preview
            .decision()
            .scores
            .iter()
            .find(|x| x.id == "base")
            .unwrap();
        assert!(chosen.eligible && !base.eligible && chosen.score < base.score);
        assert!(
            check(&c, &s, None, &o, &a, &f, "chosen", "base", 1)
                .unwrap()
                .supported
        );
        assert!(
            !check(&c, &s, None, &o, &a, &f, "base", "chosen", 1)
                .unwrap()
                .supported
        );
    }

    #[test]
    fn rejects_inconsistent_inputs_and_insufficient_batches() {
        let (c, s, o, a, mut f) = data(&[0.5, 0.5], &[0.2, 0.2]);
        let single = check(&c, &s, None, &o, &a, &f, "chosen", "base", 2).unwrap();
        assert_eq!(
            single,
            Support {
                supported_batches: 0,
                total_batches: 1,
                supported: false
            }
        );
        assert!(check(&c, &s, None, &o, &a, &f, "unknown", "base", 1).is_err());
        assert!(check(&c, &s, None, &o, &a, &f, "chosen", "base", 0).is_err());
        assert!(check(&c, &s, None, &o, &a, &f, "chosen", "base", 3).is_err());
        assert!(check(&c, &s, Some(0), &o, &a, &f, "chosen", "base", 1).is_err());
        f.plans[1].action.outcomes[0].p = 0.4;
        f.plans[1].action.outcomes[1].p = 0.6;
        assert!(check(&c, &s, None, &o, &a, &f, "chosen", "base", 1).is_err());
    }
}
