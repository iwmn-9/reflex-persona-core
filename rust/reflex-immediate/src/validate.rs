use crate::{Config, Error, Observation, Priority, Result, State};
use std::collections::BTreeSet;

pub(crate) fn id(s: &str) -> Result<()> {
    if s.is_empty() || s.chars().count() > 160 {
        return Err(Error::Invalid(
            "identifier must contain 1..160 Unicode characters",
        ));
    }
    Ok(())
}

pub(crate) fn number(x: f64, lo: f64, hi: f64) -> Result<()> {
    if !x.is_finite() || x < lo || x > hi {
        return Err(Error::Invalid(
            "finite number within declared bounds required",
        ));
    }
    Ok(())
}

pub(crate) fn tick(x: u64) -> Result<()> {
    if x >= (1_u64 << 63) {
        return Err(Error::Invalid("tick/seed must be below 2^63"));
    }
    Ok(())
}

pub(crate) fn config(c: &Config) -> Result<()> {
    for s in [&c.scope.game, &c.scope.episode, &c.scope.npc, &c.objective] {
        id(s)?;
    }
    tick(c.seed)?;
    for x in c.personality.iter().chain(&c.values) {
        number(*x, 0., 1.)?;
    }
    for x in c.policy.residual {
        number(x, -0.08 - 1e-9, 0.08 + 1e-9)?;
    }
    if let Priority::Personal { rigidity } = c.policy.priority {
        number(rigidity, 0., 1.)?;
    }
    Ok(())
}

pub(crate) fn state(s: &State) -> Result<()> {
    number(s.mode_urgency, 0., 1.)?;
    if s.age > 1_000_000 {
        return Err(Error::Invalid("decision age exceeds 1000000"));
    }
    if let Some(s) = &s.intent_action {
        id(s)?;
    }
    Ok(())
}

pub(crate) fn observation(o: &Observation, s: &State) -> Result<()> {
    observation_with_limit(o, s, 8)
}

pub(crate) fn observation_with_limit(
    o: &Observation,
    s: &State,
    outcome_limit: usize,
) -> Result<()> {
    tick(o.tick)?;
    for n in &o.needs {
        if n.enabled {
            if !n.supported {
                return Err(Error::Invalid("unsupported need enabled"));
            }
            number(
                n.deficit
                    .ok_or(Error::Invalid("enabled need requires deficit"))?,
                0.,
                1.,
            )?;
        } else if n.deficit.is_some() {
            return Err(Error::Invalid("disabled need requires null deficit"));
        }
    }
    if let Some(need) = s.primary_need {
        if !o.needs[need as usize].enabled {
            return Err(Error::Invalid(
                "primary need disabled; explicitly reset intent before changing need availability",
            ));
        }
    }
    if o.facts.is_empty() || o.facts.len() > 256 {
        return Err(Error::Invalid("1..256 observed facts required"));
    }
    for (k, v) in &o.facts {
        id(k)?;
        id(v)?;
    }
    if o.actions.is_empty() || o.actions.len() > 256 {
        return Err(Error::Invalid("1..256 action candidates required"));
    }
    let mut ids = BTreeSet::new();
    let mut viable = false;
    for a in &o.actions {
        id(&a.id)?;
        if !ids.insert(&a.id) {
            return Err(Error::Invalid("duplicate action identifier"));
        }
        if let Some(t) = &a.target {
            id(t)?;
        }
        for x in [a.confidence, a.familiarity, a.switch_cost] {
            number(x, 0., 1.)?;
        }
        if a.outcomes.is_empty() || a.outcomes.len() > outcome_limit {
            return Err(Error::Invalid(
                "outcomes exceed the declared observation limit",
            ));
        }
        let mut mass = 0.;
        for e in &a.outcomes {
            number(e.p, 0., 1.)?;
            number(e.cost, 0., 1.)?;
            for x in e.effects().iter().take(21) {
                number(*x, -1., 1.)?;
            }
            mass += e.p;
        }
        if (mass - 1.).abs() > 1e-8 {
            return Err(Error::Invalid("outcome probabilities must sum to one"));
        }
        viable |= a.legal && !a.known_failure;
    }
    if !viable {
        return Err(Error::Invalid(
            "no viable action; game must supply a recovery choice",
        ));
    }
    if let Some(r) = &o.opponent {
        number(r.confidence, 0., 1.)?;
        number(r.uncertainty, 0., 1.)?;
        if r.evidence_ids.is_empty() || r.evidence_ids.len() > 256 {
            return Err(Error::Invalid("bounded observed reading evidence required"));
        }
        for k in &r.evidence_ids {
            if !o.facts.contains_key(k) {
                return Err(Error::Invalid("reading cites unobserved evidence"));
            }
        }
        for (k, v) in &r.deltas {
            if !ids.contains(k) {
                return Err(Error::Invalid("reading cites unknown action"));
            }
            number(*v, -1., 1.)?;
        }
    }
    Ok(())
}
