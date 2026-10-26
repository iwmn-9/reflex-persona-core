//! Immediate and explicit-continuation preference choices. Estimates and learning belong to the caller.
//! Preview is pure; commit only after the chosen action actually executes.
//! Checkpoints contain resolved intent state, not world state or learned models.
mod continuation;
mod numeric;
mod types;
mod validate;
pub use continuation::{Continuation, ContinuationPlan, ReturnUnits};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub use types::*;

pub const SNAPSHOT_VERSION: &str = "reflex-immediate-v1";
pub const MAX_JSON_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_SNAPSHOT_BYTES: usize = 64 * 1024;

#[derive(Debug)]
pub enum Error {
    Invalid(&'static str),
    Json(serde_json::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) => f.write_str(s),
            Self::Json(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for Error {}

fn checksum(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

impl Observation {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_JSON_BYTES {
            return Err(Error::Invalid("observation byte limit exceeded"));
        }
        let out: Self = serde_json::from_slice(bytes)?;
        validate::observation(&out, &State::default())?;
        Ok(out)
    }
}
impl Config {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_SNAPSHOT_BYTES {
            return Err(Error::Invalid("config byte limit exceeded"));
        }
        let out: Self = serde_json::from_slice(bytes)?;
        validate::config(&out)?;
        Ok(out)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    version: String,
    config: Config,
    state: State,
    last_tick: Option<u64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    payload: Payload,
    sha256: String,
}

#[derive(Clone, Debug)]
pub struct Agent {
    current: Payload,
}

/// Opaque proposal bound to the exact resolved owner state.
/// It cannot be deserialized or have its decision/next state replaced.
#[derive(Debug)]
pub struct Prepared {
    before: Payload,
    tick: u64,
    decision: Decision,
}
impl Prepared {
    pub fn decision(&self) -> &Decision {
        &self.decision
    }
}

impl Agent {
    pub fn new(config: Config) -> Result<Self> {
        Self::from_state(config, State::default(), None)
    }

    /// Explicit validated state import, e.g. from a reference implementation.
    /// Normal saved-game loading should use restore(), which also checks integrity.
    pub fn from_state(config: Config, state: State, last_tick: Option<u64>) -> Result<Self> {
        validate::config(&config)?;
        validate::state(&state)?;
        if let Some(t) = last_tick {
            validate::tick(t)?;
        }
        Ok(Self {
            current: Payload {
                version: SNAPSHOT_VERSION.into(),
                config,
                state,
                last_tick,
            },
        })
    }
    pub fn config(&self) -> &Config {
        &self.current.config
    }
    pub fn state(&self) -> &State {
        &self.current.state
    }
    pub fn last_tick(&self) -> Option<u64> {
        self.current.last_tick
    }

    /// Deliberate tactical-state reset; personality, policy and clock are retained.
    /// Use when the game changes which need dimensions exist/are enabled.
    pub fn reset_intent(&mut self) {
        self.current.state = State::default();
    }

    pub fn preview(&self, observation: &Observation, stochastic: bool) -> Result<Prepared> {
        validate::observation(observation, self.state())?;
        if self.last_tick().is_some_and(|t| observation.tick <= t) {
            return Err(Error::Invalid("decision tick must advance"));
        }
        let decision = numeric::choose(self.config(), self.state(), observation, stochastic)?;
        Ok(Prepared {
            before: self.current.clone(),
            tick: observation.tick,
            decision,
        })
    }

    pub fn commit(&mut self, prepared: Prepared, executed_action: &str) -> Result<Decision> {
        if prepared.before != self.current {
            return Err(Error::Invalid("stale proposal or wrong owner/policy"));
        }
        if executed_action != prepared.decision.action_id {
            return Err(Error::Invalid(
                "executed action differs from proposed action",
            ));
        }
        if self.last_tick().is_some_and(|t| prepared.tick <= t) {
            return Err(Error::Invalid("proposal already committed"));
        }
        self.current.state = prepared.decision.next_state.clone();
        self.current.last_tick = Some(prepared.tick);
        Ok(prepared.decision)
    }

    pub fn checkpoint(&self) -> Result<Vec<u8>> {
        let bytes = serde_json::to_vec(&self.current)?;
        Ok(serde_json::to_vec(&Envelope {
            payload: self.current.clone(),
            sha256: checksum(&bytes),
        })?)
    }

    /// Config is supplied by the game, never silently adopted from a save file.
    /// Checksum detects accidental corruption; it is not authentication.
    pub fn restore(expected: &Config, bytes: &[u8]) -> Result<Self> {
        validate::config(expected)?;
        if bytes.len() > MAX_SNAPSHOT_BYTES {
            return Err(Error::Invalid("snapshot byte limit exceeded"));
        }
        let saved: Envelope = serde_json::from_slice(bytes)?;
        if saved.payload.version != SNAPSHOT_VERSION {
            return Err(Error::Invalid("unsupported snapshot version"));
        }
        if checksum(&serde_json::to_vec(&saved.payload)?) != saved.sha256 {
            return Err(Error::Invalid("snapshot checksum mismatch"));
        }
        if &saved.payload.config != expected {
            return Err(Error::Invalid("snapshot owner or policy mismatch"));
        }
        Self::from_state(
            saved.payload.config,
            saved.payload.state,
            saved.payload.last_tick,
        )
    }
}

/// Pure batch preview; errors leave every actor untouched. No shared random state.
pub fn preview_batch(items: &[(&Agent, &Observation)], stochastic: bool) -> Result<Vec<Prepared>> {
    items
        .iter()
        .map(|(a, o)| a.preview(o, stochastic))
        .collect()
}
