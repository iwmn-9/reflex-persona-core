//! A committed population in one game episode. No game rules or learning policy.
//! Proposals are immutable and disposable. Resolution validates the whole batch
//! before changing any actor. World execution and durable writes remain host-owned.
#![forbid(unsafe_code)]
mod host;
mod snapshot;
pub use host::{Choice, GameSession, World};
use immediate::{Agent, Config, Decision, Observation, Prepared};
pub use reflex_immediate as immediate;
use serde::{Deserialize, Serialize};
pub use snapshot::{MAX_SAVE_BYTES, MAX_WORLD_BYTES, SNAPSHOT_VERSION};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

pub const MAX_AGENTS: usize = 10_000;
pub const MAX_WORKERS: usize = 64;
static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub enum Error {
    Invalid(&'static str),
    Actor {
        npc: String,
        source: immediate::Error,
    },
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
            Self::Actor { npc, source } => write!(f, "actor {npc}: {source}"),
            Self::Json(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Actor { source, .. } => Some(source),
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}
fn actor_error(npc: &str, source: immediate::Error) -> Error {
    Error::Actor {
        npc: npc.into(),
        source,
    }
}
fn identifier(s: &str) -> Result<()> {
    if s.is_empty() || s.chars().count() > 160 {
        return Err(Error::Invalid("identifier requires 1..160 characters"));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionId {
    pub game: String,
    pub episode: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub npc: String,
    pub observation: Observation,
}

/// Executed means attempted in the real world, including stochastic failure.
/// NotExecuted means no action took place; it consumes no actor intent or RNG tick.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Execution {
    Executed {
        action_id: String,
        target: Option<String>,
    },
    NotExecuted {
        reason: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub npc: String,
    pub execution: Execution,
}

#[derive(Clone, Debug, Serialize)]
pub struct Resolution {
    pub npc: String,
    pub execution: Execution,
    pub decision: Decision,
}

/// Process-local, non-serializable proposal. A restored runtime needs fresh proposals.
#[derive(Debug)]
pub struct Batch {
    instance: u64,
    generation: u64,
    world_revision: u64,
    prepared: Vec<(String, Prepared)>,
}
impl Batch {
    /// Sorted by actor ID, independent of request order and worker scheduling.
    pub fn decisions(&self) -> impl ExactSizeIterator<Item = (&str, &Decision)> {
        self.prepared
            .iter()
            .map(|(id, p)| (id.as_str(), p.decision()))
    }
    pub fn world_revision(&self) -> u64 {
        self.world_revision
    }
}

/// Mutable state has one owner. Parallel preparation only borrows this owner.
/// No Clone: to fork a world explicitly save/restore, invalidating old proposals.
#[derive(Debug)]
pub struct Runtime {
    session: SessionId,
    world_revision: u64,
    actors: BTreeMap<String, Agent>,
    instance: u64,
    generation: u64,
}
impl Runtime {
    pub fn new(session: SessionId, world_revision: u64, configs: &[Config]) -> Result<Self> {
        identifier(&session.game)?;
        identifier(&session.episode)?;
        if configs.len() > MAX_AGENTS {
            return Err(Error::Invalid("population limit exceeded"));
        }
        let mut actors = BTreeMap::new();
        for config in configs {
            if config.scope.game != session.game || config.scope.episode != session.episode {
                return Err(Error::Invalid("actor belongs to a different world episode"));
            }
            let id = &config.scope.npc;
            if actors.contains_key(id) {
                return Err(Error::Invalid("duplicate actor"));
            }
            let agent = Agent::new(config.clone()).map_err(|e| actor_error(id, e))?;
            actors.insert(id.clone(), agent);
        }
        let instance = NEXT_INSTANCE
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Invalid("runtime instance counter exhausted"))?;
        Ok(Self {
            session,
            world_revision,
            actors,
            instance,
            generation: 0,
        })
    }
    pub fn session(&self) -> &SessionId {
        &self.session
    }
    pub fn world_revision(&self) -> u64 {
        self.world_revision
    }
    pub fn actor(&self, npc: &str) -> Option<&Agent> {
        self.actors.get(npc)
    }
    pub fn actors(&self) -> impl ExactSizeIterator<Item = (&str, &Agent)> {
        self.actors.iter().map(|(id, a)| (id.as_str(), a))
    }
    fn next_generation(&self) -> Result<u64> {
        self.generation
            .checked_add(1)
            .ok_or(Error::Invalid("runtime generation exhausted"))
    }
    pub fn insert(&mut self, config: Config) -> Result<()> {
        if self.actors.len() >= MAX_AGENTS {
            return Err(Error::Invalid("population limit exceeded"));
        }
        if config.scope.game != self.session.game || config.scope.episode != self.session.episode {
            return Err(Error::Invalid("actor belongs to a different world episode"));
        }
        let id = config.scope.npc.clone();
        if self.actors.contains_key(&id) {
            return Err(Error::Invalid("duplicate actor"));
        }
        let agent = Agent::new(config).map_err(|e| actor_error(&id, e))?;
        let generation = self.next_generation()?;
        self.actors.insert(id, agent);
        self.generation = generation;
        Ok(())
    }
    pub fn remove(&mut self, npc: &str) -> Result<Agent> {
        if !self.actors.contains_key(npc) {
            return Err(Error::Invalid("unknown actor"));
        }
        let generation = self.next_generation()?;
        let actor = self.actors.remove(npc).expect("checked actor");
        self.generation = generation;
        Ok(actor)
    }
    pub fn reset_intent(&mut self, npc: &str) -> Result<()> {
        let generation = self.next_generation()?;
        self.actors
            .get_mut(npc)
            .ok_or(Error::Invalid("unknown actor"))?
            .reset_intent();
        self.generation = generation;
        Ok(())
    }
    /// Host announces a world change without an NPC batch (e.g. player input).
    pub fn advance_world(&mut self, expected: u64, next: u64) -> Result<()> {
        self.check_revision(expected, next)?;
        let generation = self.next_generation()?;
        self.world_revision = next;
        self.generation = generation;
        Ok(())
    }
    fn check_revision(&self, expected: u64, next: u64) -> Result<()> {
        if expected != self.world_revision || next <= expected {
            return Err(Error::Invalid("world revision must match and then advance"));
        }
        Ok(())
    }
    /// Supply only each NPC's perceived facts and available actions. A subset is
    /// allowed for sleeping/removed actors. Empty batches use advance_world().
    /// workers=1 is serial; >1 uses bounded scoped threads without shared RNG.
    pub fn prepare(
        &self,
        world_revision: u64,
        inputs: &[Input],
        stochastic: bool,
        workers: usize,
    ) -> Result<Batch> {
        if world_revision != self.world_revision {
            return Err(Error::Invalid("stale world observation"));
        }
        if inputs.is_empty() || inputs.len() > MAX_AGENTS {
            return Err(Error::Invalid("1..MAX_AGENTS inputs required"));
        }
        if workers == 0 || workers > MAX_WORKERS {
            return Err(Error::Invalid("1..MAX_WORKERS workers required"));
        }
        let mut ids = BTreeSet::new();
        for input in inputs {
            if !ids.insert(&input.npc) {
                return Err(Error::Invalid("duplicate actor input"));
            }
            if !self.actors.contains_key(&input.npc) {
                return Err(Error::Invalid("unknown actor"));
            }
        }
        let evaluate = |slice: &[Input]| -> Result<Vec<(String, Prepared)>> {
            slice
                .iter()
                .map(|i| {
                    let p = self.actors[&i.npc]
                        .preview(&i.observation, stochastic)
                        .map_err(|e| actor_error(&i.npc, e))?;
                    Ok((i.npc.clone(), p))
                })
                .collect()
        };
        let mut prepared = if workers == 1 || inputs.len() == 1 {
            evaluate(inputs)?
        } else {
            std::thread::scope(|scope| -> Result<Vec<(String, Prepared)>> {
                let mut handles = Vec::new();
                let mut spawn_failed = false;
                for chunk in inputs.chunks(inputs.len().div_ceil(workers)) {
                    match std::thread::Builder::new().spawn_scoped(scope, || evaluate(chunk)) {
                        Ok(handle) => handles.push(handle),
                        Err(_) => {
                            spawn_failed = true;
                            break;
                        }
                    }
                }
                // Join all threads even when one fails; no detached calculation.
                let joined: Vec<_> = handles.into_iter().map(|h| h.join()).collect();
                if spawn_failed {
                    return Err(Error::Invalid("unable to start preparation worker"));
                }
                let mut result = Vec::with_capacity(inputs.len());
                for part in joined {
                    result
                        .extend(part.map_err(|_| Error::Invalid("preparation worker panicked"))??);
                }
                Ok(result)
            })?
        };
        prepared.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(Batch {
            instance: self.instance,
            generation: self.generation,
            world_revision,
            prepared,
        })
    }
    /// Reject stale proposals before the host attempts any world actions.
    /// The host must retain exclusive world/runtime access until resolve finishes.
    pub fn validate_batch(&self, batch: &Batch) -> Result<()> {
        if batch.instance != self.instance
            || batch.generation != self.generation
            || batch.world_revision != self.world_revision
        {
            return Err(Error::Invalid("stale batch or different runtime"));
        }
        Ok(())
    }
    /// All receipts must be present exactly once. Stage on clones first, so a bad
    /// receipt never partly commits this population. This cannot roll back an
    /// external engine: stage world execution too, or persist an execution log.
    pub fn resolve(
        &mut self,
        batch: Batch,
        next_revision: u64,
        receipts: &[Receipt],
    ) -> Result<Vec<Resolution>> {
        self.validate_batch(&batch)?;
        self.check_revision(batch.world_revision, next_revision)?;
        if receipts.len() != batch.prepared.len() {
            return Err(Error::Invalid("one receipt per proposal required"));
        }
        let mut by_id = BTreeMap::new();
        for receipt in receipts {
            if by_id
                .insert(receipt.npc.as_str(), &receipt.execution)
                .is_some()
            {
                return Err(Error::Invalid("duplicate receipt"));
            }
            if let Execution::NotExecuted { reason } = &receipt.execution {
                identifier(reason)?;
            }
        }
        let generation = self.next_generation()?;
        let mut staged = Vec::new();
        let mut result = Vec::with_capacity(batch.prepared.len());
        for (npc, prepared) in batch.prepared {
            let execution = by_id
                .remove(npc.as_str())
                .ok_or(Error::Invalid("missing or unrelated receipt"))?;
            let decision = match execution {
                Execution::Executed { action_id, target } => {
                    if target != &prepared.decision().target {
                        return Err(Error::Invalid("executed target differs from proposal"));
                    }
                    let mut actor = self.actors[&npc].clone();
                    let decision = actor
                        .commit(prepared, action_id)
                        .map_err(|e| actor_error(&npc, e))?;
                    staged.push((npc.clone(), actor));
                    decision
                }
                Execution::NotExecuted { .. } => prepared.decision().clone(),
            };
            result.push(Resolution {
                npc,
                execution: execution.clone(),
                decision,
            });
        }
        for (id, actor) in staged {
            self.actors.insert(id, actor);
        }
        self.world_revision = next_revision;
        self.generation = generation;
        Ok(result)
    }
}
