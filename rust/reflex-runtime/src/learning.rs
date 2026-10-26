//! Integrated factual learning session. Pure host transitions stage a new world;
//! every actor's feedback must validate before any world/actor is replaced.
use crate::{
    actor_error, identifier, Choice, Error, Execution, Resolution, Result, SessionId, MAX_AGENTS,
    MAX_SAVE_BYTES, MAX_WORLD_BYTES,
};
use owned::{Feedback, Input, Owner, OwnerConfig, Stage};
use reflex_immediate::Continuation;
pub use reflex_owned as owned;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Control {
    Immediate,
    Fixed32,
}

pub struct Receipt {
    pub npc: String,
    pub execution: Execution,
    /// None only for a genuinely unexecuted proposal. Executed-but-unobserved
    /// actions use Feedback with independently missing observation channels.
    pub feedback: Option<Feedback>,
}

/// Input and forecast providers are pure. A forecast may use the Python H32
/// generator over owner.state(), never an independently maintained shadow NPC.
pub trait World: Sized {
    const SAVE_FORMAT: &'static str;
    fn revision(&self) -> u64;
    fn observe(&self, owner: &Owner) -> Result<Option<Input>>;
    fn forecast(&self, owner: &Owner, prepared: &Stage) -> Result<Continuation>;
    fn advance(&self, choices: &[Choice<'_>]) -> Result<(Self, Vec<Receipt>)>;
}

pub struct LearningSession<W> {
    session: SessionId,
    control: Control,
    actors: BTreeMap<String, Owner>,
    world: W,
}
impl<W: World> LearningSession<W> {
    pub fn new(
        session: SessionId,
        control: Control,
        configs: &[OwnerConfig],
        world: W,
    ) -> Result<Self> {
        identifier(&session.game)?;
        identifier(&session.episode)?;
        if configs.len() > MAX_AGENTS {
            return Err(Error::Invalid("population limit exceeded"));
        }
        let mut actors = BTreeMap::new();
        for config in configs {
            if config.actor.scope.game != session.game
                || config.actor.scope.episode != session.episode
            {
                return Err(Error::Invalid("owner belongs to another world"));
            }
            let id = &config.actor.scope.npc;
            if actors.contains_key(id) {
                return Err(Error::Invalid("duplicate owner"));
            }
            actors.insert(
                id.clone(),
                Owner::new(config.clone()).map_err(|e| actor_error(id, e))?,
            );
        }
        Ok(Self {
            session,
            control,
            actors,
            world,
        })
    }
    pub fn world(&self) -> &W {
        &self.world
    }
    pub fn actor(&self, npc: &str) -> Option<&Owner> {
        self.actors.get(npc)
    }
    pub fn step(&mut self, stochastic: bool) -> Result<Vec<Resolution>> {
        if self.control == Control::Fixed32 && stochastic {
            return Err(Error::Invalid("fixed-H32 uses deterministic preparation"));
        }
        let mut replacements = BTreeMap::new();
        let mut prepared = Vec::new();
        for (id, original) in &self.actors {
            if let Some(input) = self.world.observe(original)? {
                let actor = original.fork_model().map_err(|e| actor_error(id, e))?;
                let mut stage = actor
                    .prepare(&input, stochastic)
                    .map_err(|e| actor_error(id, e))?;
                if self.control == Control::Fixed32 {
                    // An error is a failed transaction, never an implicit reflex fallback.
                    let forecast = self.world.forecast(&actor, &stage)?;
                    stage = actor
                        .with_continuation(stage, &forecast)
                        .map_err(|e| actor_error(id, e))?;
                }
                prepared.push((id.clone(), stage));
                replacements.insert(id.clone(), actor);
            }
        }
        let choices: Vec<_> = prepared
            .iter()
            .map(|(id, s)| Choice {
                npc: id,
                decision: s.decision(),
            })
            .collect();
        let (next, receipts) = self.world.advance(&choices)?;
        if next.revision() <= self.world.revision() {
            return Err(Error::Invalid("world revision must advance"));
        }
        let mut receipt_map = BTreeMap::new();
        for receipt in receipts {
            if receipt_map.insert(receipt.npc.clone(), receipt).is_some() {
                return Err(Error::Invalid("duplicate execution receipt"));
            }
        }
        if receipt_map.keys().collect::<BTreeSet<_>>()
            != replacements.keys().collect::<BTreeSet<_>>()
        {
            return Err(Error::Invalid("complete scheduled receipts required"));
        }
        let mut resolutions = Vec::new();
        for (id, stage) in prepared {
            let receipt = receipt_map.remove(&id).expect("validated receipt");
            let decision = stage.decision().clone();
            match &receipt.execution {
                Execution::NotExecuted { reason } => {
                    identifier(reason)?;
                    if receipt.feedback.is_some() {
                        return Err(Error::Invalid(
                            "unexecuted proposal cannot receive factual feedback",
                        ));
                    }
                    replacements.remove(&id);
                }
                Execution::Executed { action_id, target } => {
                    if action_id != &decision.action_id || target != &decision.target {
                        return Err(Error::Invalid("execution differs from chosen real root"));
                    }
                    let fb = receipt
                        .feedback
                        .as_ref()
                        .ok_or(Error::Invalid("explicit executed feedback required"))?;
                    let actor = replacements.get_mut(&id).expect("staged actor");
                    let ticket = actor
                        .commit(stage, action_id)
                        .map_err(|e| actor_error(&id, e))?;
                    actor.resolve(ticket, fb).map_err(|e| actor_error(&id, e))?;
                }
            }
            resolutions.push(Resolution {
                npc: id,
                execution: receipt.execution,
                decision,
            });
        }
        self.actors.extend(replacements);
        self.world = next;
        Ok(resolutions)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    version: String,
    session: SessionId,
    control: Control,
    world_format: String,
    world: Value,
    actors: BTreeMap<String, String>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    payload: Payload,
    sha256: String,
}
fn checksum(p: &Payload) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(p)?)))
}
impl<W: World + Serialize + DeserializeOwned> LearningSession<W> {
    pub fn checkpoint(&self) -> Result<Vec<u8>> {
        let world = serde_json::to_value(&self.world)?;
        if serde_json::to_vec(&world)?.len() > MAX_WORLD_BYTES {
            return Err(Error::Invalid("world save limit exceeded"));
        }
        let mut actors = BTreeMap::new();
        for (id, actor) in &self.actors {
            let bytes = actor.checkpoint(&world).map_err(|e| actor_error(id, e))?;
            actors.insert(
                id.clone(),
                String::from_utf8(bytes).map_err(|_| Error::Invalid("UTF8 owner save required"))?,
            );
        }
        let payload = Payload {
            version: "reflex-learning-session-v1".into(),
            session: self.session.clone(),
            control: self.control,
            world_format: W::SAVE_FORMAT.into(),
            world,
            actors,
        };
        let bytes = serde_json::to_vec(&Envelope {
            sha256: checksum(&payload)?,
            payload,
        })?;
        if bytes.len() > MAX_SAVE_BYTES {
            return Err(Error::Invalid("population save limit exceeded"));
        }
        Ok(bytes)
    }
    pub fn restore(
        session: SessionId,
        control: Control,
        configs: &[OwnerConfig],
        bytes: &[u8],
    ) -> Result<Self> {
        if bytes.len() > MAX_SAVE_BYTES {
            return Err(Error::Invalid("population save limit exceeded"));
        }
        let e: Envelope = serde_json::from_slice(bytes)?;
        let p = e.payload;
        if p.version != "reflex-learning-session-v1"
            || p.session != session
            || p.control != control
            || p.world_format != W::SAVE_FORMAT
            || checksum(&p)? != e.sha256
            || serde_json::to_vec(&p.world)?.len() > MAX_WORLD_BYTES
        {
            return Err(Error::Invalid("learning session save contract mismatch"));
        }
        let world: W = serde_json::from_value(p.world.clone())?;
        let mut result = Self::new(session, control, configs, world)?;
        if p.actors.keys().collect::<BTreeSet<_>>() != result.actors.keys().collect::<BTreeSet<_>>()
        {
            return Err(Error::Invalid("saved roster mismatch"));
        }
        for config in configs {
            let id = &config.actor.scope.npc;
            let actor = Owner::restore(config, &p.world, p.actors[id].as_bytes())
                .map_err(|e| actor_error(id, e))?;
            result.actors.insert(id.clone(), actor);
        }
        Ok(result)
    }
}
