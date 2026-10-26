//! Single-owner, framed continuation selection. The generator supplies modeled
//! paths; this boundary never records their effects as factual experience.
use crate::{
    numeric, validate, Action, Agent, Config, Error, Observation, Prepared, Result, State,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReturnUnits {
    pub unit: String,
    pub effects: [f64; 22],
    pub root_terms: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuationPlan {
    pub root: String,
    pub action: Action,
    pub purpose: f64,
    pub correction: Option<[f64; 22]>,
}

/// Explicit actor/state binding. Estimates cannot silently change disposition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Continuation {
    pub owner: Config,
    pub state: State,
    pub tick: u64,
    pub target: String,
    pub horizon: u32,
    pub max_regret: f64,
    pub units: ReturnUnits,
    pub plans: Vec<ContinuationPlan>,
}

impl Agent {
    /// Purpose corridor -> unchanged personal preference -> actual legal root.
    /// No partial forecast substitution; all checks precede the pure proposal.
    pub fn preview_continuation(
        &self,
        base: &Observation,
        forecast: &Continuation,
        allowed: &BTreeSet<String>,
    ) -> Result<Prepared> {
        validate::observation(base, self.state())?;
        if forecast.owner != *self.config()
            || forecast.state != *self.state()
            || forecast.tick != base.tick
            || self.last_tick().is_some_and(|t| base.tick <= t)
        {
            return Err(Error::Invalid("continuation owner/state/tick mismatch"));
        }
        validate::id(&forecast.target)?;
        validate::id(&forecast.units.unit)?;
        if !(1..=256).contains(&forecast.horizon)
            || forecast.plans.is_empty()
            || forecast.plans.len() > 256
        {
            return Err(Error::Invalid("bounded complete continuation required"));
        }
        validate::number(forecast.max_regret, 0., 2.)?;
        for scale in forecast.units.effects {
            validate::number(scale, 1., 512.)?;
        }
        validate::number(forecast.units.root_terms, 1., 256.)?;
        let viable: BTreeSet<_> = base
            .actions
            .iter()
            .filter(|a| a.legal && !a.known_failure)
            .map(|a| a.id.clone())
            .collect();
        if !allowed.is_subset(&viable) {
            return Err(Error::Invalid("supported roots must be viable"));
        }
        let mut ids = BTreeSet::new();
        let mut corrections = BTreeMap::new();
        let mut best = f64::NEG_INFINITY;
        let corrected = forecast.plans[0].correction.is_some();
        for plan in &forecast.plans {
            if !viable.contains(&plan.root)
                || !ids.insert(plan.action.id.clone())
                || !plan.action.legal
                || plan.action.known_failure
                || corrected != plan.correction.is_some()
            {
                return Err(Error::Invalid(
                    "complete unique feasible plans with consistent mean coverage required",
                ));
            }
            validate::number(plan.purpose, -1., 1.)?;
            if let Some(delta) = plan.correction {
                if plan.action.confidence != 1. {
                    return Err(Error::Invalid("corrected means require unit confidence"));
                }
                for v in delta {
                    validate::number(v, -1024., 1024.)?;
                }
                corrections.insert(plan.action.id.clone(), delta);
            }
            if allowed.contains(&plan.root) {
                best = best.max(plan.purpose);
            }
        }
        if !best.is_finite() {
            return Err(Error::Invalid("no supported continuation root"));
        }
        let mut observation = base.clone();
        // Match Python: preserve observed needs/facts/reading; replace only actions.
        observation.actions = forecast
            .plans
            .iter()
            .map(|plan| {
                let mut a = plan.action.clone();
                a.legal = allowed.contains(&plan.root)
                    && plan.purpose >= best - forecast.max_regret - 1e-12;
                a
            })
            .collect();
        validate::observation_with_limit(&observation, self.state(), 256)?;
        let mut decision = numeric::choose_framed(
            self.config(),
            self.state(),
            &observation,
            false,
            Some((&forecast.units, &corrections)),
        )?;
        let plan = forecast
            .plans
            .iter()
            .find(|p| p.action.id == decision.action_id)
            .expect("selected plan exists");
        let root = base
            .actions
            .iter()
            .find(|a| a.id == plan.root)
            .expect("validated real root");
        decision.action_id = root.id.clone();
        decision.target = root.target.clone();
        decision.reading_used = true;
        decision.next_state.intent_action = Some(root.id.clone());
        decision.next_state.age = if self.state().intent_action.as_ref() == Some(&root.id) {
            (self.state().age + 1).min(1_000_000)
        } else {
            0
        };
        Ok(Prepared {
            before: self.current.clone(),
            tick: base.tick,
            decision,
        })
    }
}
