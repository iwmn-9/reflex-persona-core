//! Finite, persona-scored route selection, separate from action intent.
use reflex_immediate::{Agent, Config, Error, Observation, Result, State};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteState {
    pub chosen: Option<String>,
    pub switches: u64,
}

impl RouteState {
    pub fn validate(&self) -> Result<()> {
        if self
            .chosen
            .as_ref()
            .is_some_and(|s| s.is_empty() || s.chars().count() > 160)
        {
            return Err(Error::Invalid(
                "route identifier must contain 1..160 Unicode characters",
            ));
        }
        Ok(())
    }

    pub fn checkpoint(&self, owner: &Config, tick: u64) -> Result<Vec<u8>> {
        Agent::new(owner.clone())?;
        self.validate()?;
        if tick >= (1_u64 << 63) {
            return Err(Error::Invalid("tick must be below 2^63"));
        }
        let payload = RoutePayload {
            version: "arena-routes-v1".into(),
            owner: owner.clone(),
            tick,
            state: self.clone(),
        };
        let digest = format!("{:x}", Sha256::digest(serde_json::to_vec(&payload)?));
        let bytes = serde_json::to_vec(&RouteEnvelope {
            payload,
            sha256: digest,
        })?;
        if bytes.len() > reflex_immediate::MAX_SNAPSHOT_BYTES {
            return Err(Error::Invalid("route snapshot byte limit exceeded"));
        }
        Ok(bytes)
    }

    pub fn restore(expected_owner: &Config, expected_tick: u64, bytes: &[u8]) -> Result<Self> {
        Agent::new(expected_owner.clone())?;
        if expected_tick >= (1_u64 << 63) || bytes.len() > reflex_immediate::MAX_SNAPSHOT_BYTES {
            return Err(Error::Invalid("invalid route snapshot tick or size"));
        }
        let envelope: RouteEnvelope = serde_json::from_slice(bytes)?;
        let digest = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&envelope.payload)?)
        );
        if digest != envelope.sha256
            || envelope.payload.version != "arena-routes-v1"
            || envelope.payload.owner != *expected_owner
            || envelope.payload.tick != expected_tick
        {
            return Err(Error::Invalid(
                "route snapshot integrity or identity mismatch",
            ));
        }
        envelope.payload.state.validate()?;
        Ok(envelope.payload.state)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RoutePayload {
    version: String,
    owner: Config,
    tick: u64,
    state: RouteState,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteEnvelope {
    payload: RoutePayload,
    sha256: String,
}

impl<'de> Deserialize<'de> for RouteState {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            chosen: Option<String>,
            switches: u64,
        }
        let raw = Raw::deserialize(deserializer)?;
        let state = Self {
            chosen: raw.chosen,
            switches: raw.switches,
        };
        state.validate().map_err(serde::de::Error::custom)?;
        Ok(state)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RouteAudit {
    pub chosen: String,
    pub candidate: String,
    pub reason: String,
    pub switched: bool,
    pub scores: BTreeMap<String, f64>,
    pub eligible: BTreeMap<String, bool>,
    pub required_gain: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RouteChoice {
    pub state: RouteState,
    pub audit: RouteAudit,
}

pub fn choose(
    config: &Config,
    intent: &State,
    observation: &Observation,
    previous: &RouteState,
    uncertainty: f64,
    switch_margin: f64,
) -> Result<RouteChoice> {
    previous.validate()?;
    if !uncertainty.is_finite()
        || !(0.0..=1.0).contains(&uncertainty)
        || !switch_margin.is_finite()
        || !(0.0..=1.0).contains(&switch_margin)
    {
        return Err(Error::Invalid(
            "route uncertainty and margin must be within 0..1",
        ));
    }
    let mut scoped = intent.clone();
    scoped.intent_action = None;
    let agent = Agent::from_state(config.clone(), scoped, None)?;
    let prepared = agent.preview(observation, false)?;
    let decision = prepared.decision();
    let scores: BTreeMap<_, _> = decision
        .scores
        .iter()
        .map(|s| (s.id.clone(), s.score))
        .collect();
    let eligible: BTreeMap<_, _> = decision
        .scores
        .iter()
        .map(|s| (s.id.clone(), s.eligible))
        .collect();
    let candidate = decision.action_id.clone();
    let required_gain = switch_margin + 0.1 * uncertainty;
    let mut chosen = candidate.clone();
    let reason = match previous.chosen.as_ref() {
        None => "initial_preference",
        Some(old) if old == &candidate => "same_route",
        Some(old) if !eligible.get(old).copied().unwrap_or(false) => {
            "old_route_unavailable_or_rejected"
        }
        Some(old) if scores[&candidate] - scores[old] <= required_gain => {
            chosen = old.clone();
            "retain_axis_below_switch_margin"
        }
        Some(_) => "material_persona_and_opportunity_gain",
    };
    let switched = previous.chosen.as_ref().is_some_and(|old| old != &chosen);
    let switches = previous
        .switches
        .checked_add(u64::from(switched))
        .ok_or(Error::Invalid("route switch count overflow"))?;
    Ok(RouteChoice {
        state: RouteState {
            chosen: Some(chosen.clone()),
            switches,
        },
        audit: RouteAudit {
            chosen,
            candidate,
            reason: reason.into(),
            switched,
            scores,
            eligible,
            required_gain,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reflex_immediate::{Action, NeedInput, Outcome, Policy, Scope};

    fn fixture() -> (Config, State, Observation) {
        let config = Config {
            scope: Scope {
                game: "test".into(),
                episode: "one".into(),
                npc: "person".into(),
            },
            seed: 1,
            personality: [0.5; 5],
            values: [0.2; 10],
            objective: "goal".into(),
            policy: Policy::default(),
        };
        let actions = [("a", 0.0), ("b", 0.5)]
            .into_iter()
            .map(|(id, objective)| Action {
                id: id.into(),
                target: None,
                legal: true,
                known_failure: false,
                confidence: 0.8,
                familiarity: 0.4,
                switch_cost: 0.5,
                outcomes: vec![Outcome {
                    p: 1.,
                    objective,
                    needs: [0.; 5],
                    values: [0.; 10],
                    style: [0.; 5],
                    cost: 0.,
                }],
            })
            .collect();
        let observation = Observation {
            tick: 0,
            needs: std::array::from_fn(|_| NeedInput {
                supported: true,
                enabled: true,
                deficit: Some(0.3),
            }),
            facts: BTreeMap::from([("scene".into(), "public".into())]),
            actions,
            opponent: None,
        };
        (config, State::default(), observation)
    }

    #[test]
    fn threshold_retains_and_unavailable_switches_without_mutation() {
        let (config, mut intent, mut observation) = fixture();
        intent.intent_action = Some("a".into());
        let before = intent.clone();
        let previous = RouteState {
            chosen: Some("a".into()),
            switches: 4,
        };
        let base = choose(&config, &intent, &observation, &previous, 0., 0.).unwrap();
        assert_eq!(base.audit.candidate, "b");
        let gain = base.audit.scores["b"] - base.audit.scores["a"];
        assert!(gain > 0. && gain < 1.);
        let held = choose(&config, &intent, &observation, &previous, 0., gain).unwrap();
        assert_eq!(held.state, previous);
        assert_eq!(held.audit.reason, "retain_axis_below_switch_margin");
        assert_eq!(base.state.chosen.as_deref(), Some("b"));
        assert_eq!(base.state.switches, 5);
        assert_eq!(intent, before);
        observation.actions[0].legal = false;
        let rejected = choose(&config, &intent, &observation, &previous, 1., 1.).unwrap();
        assert_eq!(rejected.audit.reason, "old_route_unavailable_or_rejected");
        assert_eq!(rejected.state.chosen.as_deref(), Some("b"));
    }

    #[test]
    fn unknown_and_renamed_routes_and_intent_are_independent() {
        let (config, mut intent, mut observation) = fixture();
        let previous = RouteState {
            chosen: Some("missing".into()),
            switches: 0,
        };
        let first = choose(&config, &intent, &observation, &previous, 0., 0.).unwrap();
        assert_eq!(first.audit.reason, "old_route_unavailable_or_rejected");
        intent.intent_action = Some("a".into());
        assert_eq!(
            first,
            choose(&config, &intent, &observation, &previous, 0., 0.).unwrap()
        );
        observation.actions[0].id = "x".into();
        observation.actions[1].id = "y".into();
        let renamed = choose(
            &config,
            &intent,
            &observation,
            &RouteState::default(),
            0.,
            0.,
        )
        .unwrap();
        assert_eq!(renamed.audit.chosen, "y");
        assert_eq!(first.audit.scores["b"], renamed.audit.scores["y"]);
    }

    #[test]
    fn rejects_invalid_restored_state_and_controls() {
        assert!(serde_json::from_str::<RouteState>(r#"{"chosen":"","switches":0}"#).is_err());
        assert!(serde_json::from_str::<RouteState>(r#"{"chosen":null,"switches":-1}"#).is_err());
        assert!(
            serde_json::from_str::<RouteState>(r#"{"chosen":null,"switches":0,"extra":1}"#)
                .is_err()
        );
        let (config, intent, observation) = fixture();
        let invalid = RouteState {
            chosen: Some("長".repeat(161)),
            switches: 0,
        };
        assert!(choose(&config, &intent, &observation, &invalid, 0., 0.).is_err());
        assert!(choose(
            &config,
            &intent,
            &observation,
            &RouteState::default(),
            f64::NAN,
            0.
        )
        .is_err());
        assert!(choose(
            &config,
            &intent,
            &observation,
            &RouteState::default(),
            0.,
            1.01
        )
        .is_err());
        let ok = RouteState {
            chosen: Some("長".repeat(160)),
            switches: 0,
        };
        assert_eq!(
            serde_json::from_slice::<RouteState>(&serde_json::to_vec(&ok).unwrap()).unwrap(),
            ok
        );
    }

    #[test]
    fn checkpoint_binds_owner_tick_and_integrity() {
        let (config, _, _) = fixture();
        let state = RouteState {
            chosen: Some("a".into()),
            switches: 2,
        };
        let bytes = state.checkpoint(&config, 9).unwrap();
        assert_eq!(RouteState::restore(&config, 9, &bytes).unwrap(), state);
        let mut other = config.clone();
        other.scope.npc = "other".into();
        assert!(RouteState::restore(&other, 9, &bytes).is_err());
        assert!(RouteState::restore(&config, 10, &bytes).is_err());
        let mut modified: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        modified["payload"]["state"]["switches"] = 3.into();
        assert!(RouteState::restore(&config, 9, &serde_json::to_vec(&modified).unwrap()).is_err());
        assert!(state.checkpoint(&config, 1_u64 << 63).is_err());
        assert!(RouteState::restore(
            &config,
            9,
            &vec![b' '; reflex_immediate::MAX_SNAPSHOT_BYTES + 1]
        )
        .is_err());
    }
}
