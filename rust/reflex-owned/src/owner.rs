//! One factual owner with detached proposals and atomic feedback. Known public
//! conditional effects and rule alphabets are host input, never hidden truth.
use crate::{evidence, experience, prediction, progress, Error, Result};
use reflex_immediate::{
    Agent, Config, Continuation, Decision, Observation, Outcome, Prepared, State,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);
pub const SAVE_VERSION: &str = "reflex-owned-v1";
pub const MAX_SAVE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnerConfig {
    pub actor: Config,
    pub predictor: prediction::Definition,
    /// Versions all adapter effects, goal, feature bindings, and generator semantics.
    pub contract: String,
    pub purpose_target: String,
}
impl OwnerConfig {
    fn validate(&self) -> Result<()> {
        Agent::new(self.actor.clone())?;
        self.predictor.validate()?;
        crate::labels(std::slice::from_ref(&self.contract), 1)?;
        crate::labels(std::slice::from_ref(&self.purpose_target), 1)?;
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub observation: Observation,
    pub bindings: BTreeMap<String, experience::Binding>,
    pub responses: prediction::Inputs,
    /// Every action, every response, in responses.legal order. Probabilities
    /// are replaced by the owned learner; supplied effects are known rules.
    pub conditional: BTreeMap<String, Vec<Outcome>>,
    pub purpose: progress::Purpose,
    pub threatened: bool,
    pub maintains_advantage: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Feedback {
    pub observed: Option<[f64; 22]>,
    pub response: Option<String>,
    pub needs: Option<progress::NeedFeedback>,
    pub purpose: Option<progress::PurposeFeedback>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedState {
    pub intent: State,
    pub last_tick: Option<u64>,
    pub experience: experience::Memory,
    pub evidence: evidence::Evidence,
    pub prediction: prediction::Memory,
    pub predictor_tick: Option<u64>,
    pub pressure: progress::Pressure,
    pub pressure_tick: Option<u64>,
    pub progress: progress::Progress,
}
#[derive(Debug)]
pub struct Stage {
    instance: u64,
    before: LearnedState,
    input: Input,
    effective: Observation,
    allowed: BTreeSet<String>,
    proposal: Prepared,
    conditional_read: bool,
    planned: bool,
    stochastic: bool,
    shadow: Observation,
    probabilities: Vec<f64>,
}
impl Stage {
    pub fn decision(&self) -> &Decision {
        self.proposal.decision()
    }
    pub fn context(&self) -> &Observation {
        &self.effective
    }
    pub fn allowed(&self) -> &BTreeSet<String> {
        &self.allowed
    }
    pub fn is_planned(&self) -> bool {
        self.planned
    }
}
/// Process-local receipt capability. It cannot be forged/deserialized or used
/// against another NPC, even if both actors have the same simulation tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticket {
    instance: u64,
    tick: u64,
}
#[derive(Debug)]
struct Pending {
    ticket: u64,
    input: Input,
    effective: Observation,
    action: String,
    conditional_read: bool,
    shadow: Observation,
    probabilities: Vec<f64>,
}
#[derive(Debug)]
pub struct Owner {
    config: OwnerConfig,
    state: LearnedState,
    agent: Agent,
    pending: Option<Pending>,
    instance: u64,
}
impl Owner {
    pub fn new(config: OwnerConfig) -> Result<Self> {
        Self::from_state(config, LearnedState::default())
    }
    pub fn from_state(config: OwnerConfig, state: LearnedState) -> Result<Self> {
        config.validate()?;
        let agent = Agent::from_state(config.actor.clone(), state.intent.clone(), state.last_tick)?;
        state.prediction.validate(&config.predictor)?;
        state.experience.validate()?;
        state.evidence.validate()?;
        state.pressure.validate()?;
        state.progress.validate(state.last_tick.is_some())?;
        if state
            .pressure_tick
            .is_some_and(|t| state.last_tick.is_none_or(|last| t > last))
        {
            return Err(Error::Invalid(
                "pressure timeline exceeds the factual owner",
            ));
        }
        if state.pressure_tick.is_none() && state.pressure != progress::Pressure::default() {
            return Err(Error::Invalid("unobserved need pressure must be empty"));
        }
        if state.prediction.observations == 0 && state.predictor_tick.is_some()
            || state.prediction.observations > 0 && state.predictor_tick.is_none()
            || state.predictor_tick.is_some_and(|t| {
                state.last_tick.is_none_or(|last| t > last) || t < state.prediction.observations - 1
            })
            || state.last_tick.is_none()
                && (!state.experience.entries.is_empty()
                    || !state.evidence.entries.is_empty()
                    || state.intent != State::default()
                    || state.pressure != progress::Pressure::default())
        {
            return Err(Error::Invalid("inconsistent learned state timeline"));
        }
        let instance = NEXT_INSTANCE
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| Error::Invalid("owner instance counter exhausted"))?;
        Ok(Self {
            config,
            state,
            agent,
            pending: None,
            instance,
        })
    }
    pub fn config(&self) -> &OwnerConfig {
        &self.config
    }
    pub fn state(&self) -> Result<&LearnedState> {
        if self.pending.is_some() {
            return Err(Error::Invalid("resolve actual feedback first"));
        }
        Ok(&self.state)
    }
    pub fn fork_model(&self) -> Result<Self> {
        Self::from_state(self.config.clone(), self.state()?.clone())
    }
    pub fn predict(&self, inputs: &prediction::Inputs) -> Result<Vec<f64>> {
        self.state()?
            .prediction
            .predict(&self.config.predictor, inputs)
    }
    /// The supported frozen profile uses the default 64-node read gate, default
    /// pressure/progress, segmented public events, and estimated effect bindings.
    pub fn prepare(&self, input: &Input, stochastic: bool) -> Result<Stage> {
        self.state()?;
        // Validate the unmodified raw observation before adding any owned state.
        self.agent.preview(&input.observation, stochastic)?;
        let probabilities = self.predict(&input.responses)?;
        let roots: BTreeSet<_> = input
            .observation
            .actions
            .iter()
            .map(|a| a.id.clone())
            .collect();
        if input.conditional.keys().cloned().collect::<BTreeSet<_>>() != roots {
            return Err(Error::Invalid(
                "known conditional rules must cover all roots",
            ));
        }
        let mut known = input.observation.clone();
        for a in &mut known.actions {
            let rows = &input.conditional[&a.id];
            if rows.len() != probabilities.len() {
                return Err(Error::Invalid(
                    "one conditional effect per legal response required",
                ));
            }
            a.outcomes = rows
                .iter()
                .zip(&probabilities)
                .map(|(row, p)| {
                    let mut row = row.clone();
                    row.p = *p;
                    row
                })
                .collect();
        }
        // Even unused forecasts must be well formed. Reference immediate packets
        // allow at most eight outcomes, including the conditional response table.
        self.agent.preview(&known, stochastic)?;
        let mut raw = input.observation.clone();
        self.state.pressure.apply(&mut raw);
        let allowed = self.state.progress.allowed(&raw, &input.purpose)?;
        let mut effective = self.state.experience.prepare(&raw, &input.bindings)?;
        let shadow = effective.clone();
        let mask = |o: &Observation| {
            let mut o = o.clone();
            for a in &mut o.actions {
                a.legal &= allowed.contains(&a.id);
            }
            o
        };
        let mut proposal = self.agent.preview(&mask(&effective), stochastic)?;
        let d = proposal.decision();
        let selected = effective
            .actions
            .iter()
            .find(|a| a.id == d.action_id)
            .unwrap();
        let worst = selected
            .outcomes
            .iter()
            .filter(|r| r.p > 0.)
            .map(|r| r.objective)
            .fold(f64::INFINITY, f64::min);
        let mut scores: Vec<_> = d
            .scores
            .iter()
            .filter(|r| r.eligible)
            .map(|r| r.score)
            .collect();
        scores.sort_by(f64::total_cmp);
        let close = scores.len() > 1
            && scores[scores.len() - 1] - scores[scores.len() - 2] < 0.05
            && 1. - selected.confidence > 0.1;
        let read = (input.threatened || close || worst < -0.3)
            && (!input.maintains_advantage || input.threatened)
            && effective.actions.len() * probabilities.len() <= 64;
        if read {
            known = effective.clone();
            for a in &mut known.actions {
                a.outcomes = input.conditional[&a.id]
                    .iter()
                    .zip(&probabilities)
                    .map(|(row, p)| {
                        let mut row = row.clone();
                        row.p = *p;
                        row
                    })
                    .collect();
            }
            let candidate = self.agent.preview(&mask(&known), stochastic)?;
            let old = candidate
                .decision()
                .scores
                .iter()
                .find(|s| s.id == proposal.decision().action_id)
                .unwrap();
            let new = candidate
                .decision()
                .scores
                .iter()
                .find(|s| s.id == candidate.decision().action_id)
                .unwrap();
            if !old.eligible || new.score - old.score > 0.05 {
                proposal = candidate;
                effective = known;
            }
        }
        Ok(Stage {
            instance: self.instance,
            before: self.state.clone(),
            input: input.clone(),
            effective,
            allowed,
            proposal,
            conditional_read: read,
            planned: false,
            stochastic,
            shadow,
            probabilities,
        })
    }
    fn check(&self, stage: &Stage) -> Result<()> {
        self.state()?;
        if stage.instance != self.instance || stage.before != self.state {
            return Err(Error::Invalid("stale/foreign owner proposal"));
        }
        Ok(())
    }
    /// Apply the selected composition's complete 32-step forecast. The host's
    /// generator remains a separate component; no fallback on incomplete input.
    pub fn with_continuation(&self, mut stage: Stage, forecast: &Continuation) -> Result<Stage> {
        self.check(&stage)?;
        if stage.stochastic {
            return Err(Error::Invalid(
                "fixed-H32 requires deterministic preparation",
            ));
        }
        // FixedHorizonOwner clips its reported prefix at an earlier terminal;
        // the configured budget is 32, not a requirement to invent post-terminal actions.
        if !(1..=32).contains(&forecast.horizon) || forecast.target != self.config.purpose_target {
            return Err(Error::Invalid(
                "matching fixed-H32 purpose forecast required",
            ));
        }
        stage.proposal =
            self.agent
                .preview_continuation(&stage.effective, forecast, &stage.allowed)?;
        stage.planned = true;
        Ok(stage)
    }
    /// Execute the chosen real root first. A stochastic failure still counts as
    /// executed; a cancelled proposal is simply dropped, consuming no state.
    pub fn commit(&mut self, stage: Stage, executed_action: &str) -> Result<Ticket> {
        self.check(&stage)?;
        let tick = stage.input.observation.tick;
        let mut agent = self.agent.clone();
        let decision = agent.commit(stage.proposal, executed_action)?;
        self.state.intent = decision.next_state;
        self.state.last_tick = Some(tick);
        self.agent = agent;
        self.pending = Some(Pending {
            ticket: tick,
            input: stage.input,
            effective: stage.effective,
            action: executed_action.into(),
            conditional_read: stage.conditional_read,
            shadow: stage.shadow,
            probabilities: stage.probabilities,
        });
        Ok(Ticket {
            instance: self.instance,
            tick,
        })
    }
    /// All channels validate and update on a private copy. A rejected event,
    /// impossible effect, or bad progress value leaves the pending trial intact.
    pub fn resolve(&mut self, ticket: Ticket, feedback: &Feedback) -> Result<()> {
        if ticket.instance != self.instance {
            return Err(Error::Invalid("foreign factual ticket"));
        }
        let ticket = ticket.tick;
        let p = self
            .pending
            .as_ref()
            .filter(|p| p.ticket == ticket)
            .ok_or(Error::Invalid("matching pending factual ticket required"))?;
        let mut next = self.state.clone();
        if let Some(response) = &feedback.response {
            next.prediction
                .observe(&self.config.predictor, &p.input.responses, response)?;
            next.predictor_tick = Some(ticket);
        }
        if let Some(observed) = &feedback.observed {
            experience::validate_vector(observed)?;
            if p.conditional_read {
                let rows = &p.input.conditional[&p.action];
                let matches = |row: &Outcome| {
                    experience::vector(row)
                        .iter()
                        .zip(observed)
                        .all(|(a, b)| (a - b).abs() <= 1e-8)
                };
                let valid = if let Some(response) = &feedback.response {
                    p.input
                        .responses
                        .legal
                        .iter()
                        .position(|k| k == response)
                        .is_some_and(|i| matches(&rows[i]))
                } else {
                    rows.iter().any(matches)
                };
                if !valid {
                    return Err(Error::Invalid(
                        "actual effect contradicts known conditional rules",
                    ));
                }
            }
            let selected = p
                .effective
                .actions
                .iter()
                .find(|a| a.id == p.action)
                .unwrap();
            next.experience
                .observe(&p.input.bindings[&p.action], &selected.outcomes, observed)?;
            let prior = &p
                .input
                .observation
                .actions
                .iter()
                .find(|a| a.id == p.action)
                .unwrap()
                .outcomes;
            let candidate = &p
                .shadow
                .actions
                .iter()
                .find(|a| a.id == p.action)
                .unwrap()
                .outcomes;
            next.evidence
                .experience(&p.input.bindings[&p.action], prior, candidate, observed);
        }
        if let Some(response) = &feedback.response {
            if p.conditional_read {
                let index = p
                    .input
                    .responses
                    .legal
                    .iter()
                    .position(|z| z == response)
                    .expect("validated response");
                next.evidence.response(&p.probabilities, index);
            }
        }
        if let Some(fb) = &feedback.needs {
            next.pressure.observe(fb)?;
            next.pressure_tick = Some(ticket);
        }
        next.progress
            .observe(&p.action, &p.input.purpose, feedback.purpose.as_ref())?;
        self.state = next;
        self.pending = None;
        Ok(())
    }
    pub fn checkpoint(&self, world: &serde_json::Value) -> Result<Vec<u8>> {
        let payload = Save {
            version: SAVE_VERSION.into(),
            config: self.config.clone(),
            state: self.state()?.clone(),
            world: world.clone(),
        };
        let bytes = serde_json::to_vec(&payload)?;
        let result = serde_json::to_vec(&Envelope {
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            payload,
        })?;
        if result.len() > MAX_SAVE_BYTES {
            return Err(Error::Invalid("save byte limit exceeded"));
        }
        Ok(result)
    }
    pub fn restore(
        expected: &OwnerConfig,
        world: &serde_json::Value,
        bytes: &[u8],
    ) -> Result<Self> {
        if bytes.len() > MAX_SAVE_BYTES {
            return Err(Error::Invalid("save byte limit exceeded"));
        }
        let e: Envelope = serde_json::from_slice(bytes)?;
        if e.payload.version != SAVE_VERSION
            || &e.payload.config != expected
            || &e.payload.world != world
            || e.sha256 != format!("{:x}", Sha256::digest(serde_json::to_vec(&e.payload)?))
        {
            return Err(Error::Invalid(
                "save version/config/world/checksum mismatch",
            ));
        }
        Self::from_state(expected.clone(), e.payload.state)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Save {
    version: String,
    config: OwnerConfig,
    state: LearnedState,
    world: serde_json::Value,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    payload: Save,
    sha256: String,
}
