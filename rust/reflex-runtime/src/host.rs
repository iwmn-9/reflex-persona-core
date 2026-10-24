use super::*;
use serde::de::DeserializeOwned;

pub struct Choice<'a> {
    pub npc: &'a str,
    pub decision: &'a Decision,
}

/// A rules/perception adapter, not an AI policy. advance() must stage a new world
/// without mutating self, external/shared state, files or a live engine. All real
/// side effects belong after a successful GameSession::step. World RNG and any
/// caller-owned perception/estimation state must be included in its serialization.
pub trait World: Sized {
    const SAVE_FORMAT: &'static str;
    fn revision(&self) -> u64;
    /// None explicitly means this NPC is not scheduled (e.g. sleeping/dead).
    fn observe(&self, config: &Config) -> Result<Option<Observation>>;
    /// An empty choice set must also be supported for a world-only advance.
    /// Return one receipt for every supplied choice, including blocked execution.
    fn advance(&self, choices: &[Choice<'_>]) -> Result<(Self, Vec<Receipt>)>;
}

/// Integrated in-memory transaction for a pure world adapter. On any returned
/// error, both the world and every actor remain unchanged. This is not a database
/// transaction or an executor for irreversible actions in another engine.
pub struct GameSession<W> {
    runtime: Runtime,
    world: W,
}
impl<W: World> GameSession<W> {
    pub fn new(session: SessionId, configs: &[Config], world: W) -> Result<Self> {
        Ok(Self {
            runtime: Runtime::new(session, world.revision(), configs)?,
            world,
        })
    }
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }
    pub fn world(&self) -> &W {
        &self.world
    }
    pub fn insert(&mut self, config: Config) -> Result<()> {
        self.runtime.insert(config)
    }
    pub fn remove(&mut self, npc: &str) -> Result<Agent> {
        self.runtime.remove(npc)
    }
    pub fn reset_intent(&mut self, npc: &str) -> Result<()> {
        self.runtime.reset_intent(npc)
    }
    /// External/player effects must likewise be staged by the host.
    pub fn replace_world(&mut self, next: W) -> Result<()> {
        self.runtime
            .advance_world(self.world.revision(), next.revision())?;
        self.world = next;
        Ok(())
    }
    pub fn step(&mut self, stochastic: bool, workers: usize) -> Result<Vec<Resolution>> {
        if workers == 0 || workers > MAX_WORKERS {
            return Err(Error::Invalid("1..MAX_WORKERS workers required"));
        }
        let mut inputs = Vec::new();
        for (npc, actor) in self.runtime.actors() {
            if let Some(observation) = self.world.observe(actor.config())? {
                inputs.push(Input {
                    npc: npc.into(),
                    observation,
                });
            }
        }
        let (next, results) = if inputs.is_empty() {
            let (next, receipts) = self.world.advance(&[])?;
            if !receipts.is_empty() {
                return Err(Error::Invalid("unscheduled execution receipt"));
            }
            self.runtime
                .advance_world(self.world.revision(), next.revision())?;
            (next, Vec::new())
        } else {
            let batch =
                self.runtime
                    .prepare(self.world.revision(), &inputs, stochastic, workers)?;
            let choices: Vec<_> = batch
                .decisions()
                .map(|(npc, decision)| Choice { npc, decision })
                .collect();
            let (next, receipts) = self.world.advance(&choices)?;
            let results = self.runtime.resolve(batch, next.revision(), &receipts)?;
            (next, results)
        };
        self.world = next;
        Ok(results)
    }
}
impl<W: World + Serialize + DeserializeOwned> GameSession<W> {
    pub fn checkpoint(&self) -> Result<Vec<u8>> {
        self.runtime.checkpoint(
            self.world.revision(),
            W::SAVE_FORMAT,
            &serde_json::to_vec(&self.world)?,
        )
    }
    pub fn restore(session: SessionId, configs: &[Config], bytes: &[u8]) -> Result<Self> {
        let (runtime, bytes) = Runtime::restore(session, configs, W::SAVE_FORMAT, bytes)?;
        let world: W = serde_json::from_slice(&bytes)?;
        if world.revision() != runtime.world_revision() {
            return Err(Error::Invalid(
                "restored world revision differs from population",
            ));
        }
        Ok(Self { runtime, world })
    }
}
