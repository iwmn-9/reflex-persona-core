use super::*;
use sha2::{Digest, Sha256};

pub const SNAPSHOT_VERSION: &str = "reflex-runtime-v1";
pub const MAX_WORLD_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_SAVE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedActor {
    npc: String,
    checkpoint: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    version: String,
    session: SessionId,
    world_revision: u64,
    world_format: String,
    world: Vec<u8>,
    actors: Vec<SavedActor>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    payload: Payload,
    sha256: String,
}
fn digest(payload: &Payload) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(payload)?)
    ))
}
impl Runtime {
    /// Snapshot only a committed world at the same revision. The host owns world
    /// serialization (including its RNG, perception, needs and learned estimates).
    /// Persist these bytes atomically; never save the two sides independently.
    pub fn checkpoint(
        &self,
        world_revision: u64,
        world_format: &str,
        world: &[u8],
    ) -> Result<Vec<u8>> {
        if world_revision != self.world_revision {
            return Err(Error::Invalid("world and population save revisions differ"));
        }
        identifier(world_format)?;
        if world.len() > MAX_WORLD_BYTES {
            return Err(Error::Invalid("world snapshot limit exceeded"));
        }
        let actors = self
            .actors
            .iter()
            .map(|(npc, actor)| {
                let bytes = actor.checkpoint().map_err(|e| actor_error(npc, e))?;
                Ok(SavedActor {
                    npc: npc.clone(),
                    checkpoint: String::from_utf8(bytes).expect("JSON is UTF8"),
                })
            })
            .collect::<Result<_>>()?;
        let payload = Payload {
            version: SNAPSHOT_VERSION.into(),
            session: self.session.clone(),
            world_revision,
            world_format: world_format.into(),
            world: world.to_vec(),
            actors,
        };
        let sha256 = digest(&payload)?;
        let bytes = serde_json::to_vec(&Envelope { payload, sha256 })?;
        if bytes.len() > MAX_SAVE_BYTES {
            return Err(Error::Invalid("population snapshot limit exceeded"));
        }
        Ok(bytes)
    }
    /// Supply the exact expected roster/configs and world schema independently of
    /// the untrusted save. Restores a new owner: old process-local proposals expire.
    /// SHA256 detects corruption, not malicious replacement or rollback attacks.
    pub fn restore(
        expected: SessionId,
        configs: &[Config],
        world_format: &str,
        bytes: &[u8],
    ) -> Result<(Self, Vec<u8>)> {
        if bytes.len() > MAX_SAVE_BYTES {
            return Err(Error::Invalid("population snapshot limit exceeded"));
        }
        identifier(world_format)?;
        let saved: Envelope = serde_json::from_slice(bytes)?;
        let p = saved.payload;
        if p.version != SNAPSHOT_VERSION || p.session != expected || p.world_format != world_format
        {
            return Err(Error::Invalid(
                "snapshot version, episode or world schema mismatch",
            ));
        }
        if p.actors.len() > MAX_AGENTS
            || p.world.len() > MAX_WORLD_BYTES
            || digest(&p)? != saved.sha256
        {
            return Err(Error::Invalid("snapshot bounds or checksum mismatch"));
        }
        let mut runtime = Self::new(expected, p.world_revision, configs)?;
        if p.actors.len() != runtime.actors.len() {
            return Err(Error::Invalid("snapshot roster mismatch"));
        }
        let mut seen = BTreeSet::new();
        for entry in p.actors {
            if !seen.insert(entry.npc.clone()) {
                return Err(Error::Invalid("duplicate saved actor"));
            }
            let expected = runtime
                .actors
                .get(&entry.npc)
                .ok_or(Error::Invalid("snapshot roster mismatch"))?
                .config();
            let agent = Agent::restore(expected, entry.checkpoint.as_bytes())
                .map_err(|e| actor_error(&entry.npc, e))?;
            runtime.actors.insert(entry.npc, agent);
        }
        Ok((runtime, p.world))
    }
}
