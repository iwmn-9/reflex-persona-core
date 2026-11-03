//! Recipient-specific factual relationship memory and bounded action appraisal.
use reflex_immediate::{Config, Decision, Observation};
use reflex_owned::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct Binding {
    pub target: String,
    pub support: f64,
    pub distress: f64,
}
#[derive(Clone, Debug)]
pub struct Event {
    pub target: String,
    pub event_id: String,
    pub tick: u64,
    pub benefit: f64,
    pub agency: bool,
    pub confidence: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Status {
    pub attitude: f64,
    pub confidence: f64,
    pub evidence: f64,
    pub signal: f64,
    pub stance: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Score {
    pub target: String,
    pub status: Status,
    pub adjustment: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    target: String,
    attitude: f64,
    evidence: f64,
    tick: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Memory {
    config: Config,
    capacity: usize,
    half_life: f64,
    entries: Vec<Entry>,
    events: Vec<(String, u64)>,
    last_tick: Option<u64>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Save {
    version: String,
    world: String,
    memory: Memory,
    sha256: String,
}
fn label(s: &str) -> bool {
    !s.is_empty() && s.chars().count() <= 160
}
fn bounded(x: f64, lo: f64, hi: f64) -> bool {
    x.is_finite() && (lo..=hi).contains(&x)
}
fn digest<T: Serialize>(x: &T) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(x)?)))
}
const MAX_TICK: u64 = i64::MAX as u64 - 1;
impl Memory {
    pub fn new(config: &Config) -> Result<Self> {
        Self::with_limits(config, 64, 24.)
    }
    pub fn with_limits(config: &Config, capacity: usize, half_life: f64) -> Result<Self> {
        reflex_immediate::Agent::new(config.clone())
            .map_err(|_| Error::Invalid("valid social owner required"))?;
        if !(1..=1024).contains(&capacity) || !bounded(half_life, 1., 4096.) {
            return Err(Error::Invalid("bounded social limits required"));
        }
        Ok(Self {
            config: config.clone(),
            capacity,
            half_life,
            entries: vec![],
            events: vec![],
            last_tick: None,
        })
    }
    pub fn validate_for(&self, config: &Config, tick: Option<u64>) -> Result<()> {
        Self::with_limits(config, self.capacity, self.half_life)?;
        if &self.config != config
            || self.last_tick != tick
            || tick.is_some_and(|t| t > MAX_TICK)
            || self.entries.len() > self.capacity
            || self.events.len() > 4 * self.capacity
        {
            return Err(Error::Invalid("social owner, clock or capacity mismatch"));
        }
        let mut targets = BTreeSet::new();
        for e in &self.entries {
            if !label(&e.target)
                || !targets.insert(&e.target)
                || !bounded(e.attitude, -1., 1.)
                || !bounded(e.evidence, 0., 64.)
                || tick.is_none_or(|t| e.tick > t)
            {
                return Err(Error::Invalid("invalid social entry"));
            }
        }
        let mut ids = BTreeSet::new();
        for (id, t) in &self.events {
            if !label(id) || !ids.insert(id) || tick.is_none_or(|clock| *t > clock) {
                return Err(Error::Invalid("invalid social event"));
            }
        }
        Ok(())
    }
    pub fn observe_batch(&mut self, events: &[Event], executed_tick: u64) -> Result<()> {
        if executed_tick > MAX_TICK || self.last_tick.is_some_and(|t| executed_tick <= t) {
            return Err(Error::Invalid("forward social clock required"));
        }
        let mut ids: BTreeSet<&str> = self.events.iter().map(|(id, _)| id.as_str()).collect();
        for e in events {
            if e.tick != executed_tick
                || !label(&e.target)
                || !label(&e.event_id)
                || !bounded(e.benefit, -1., 1.)
                || !bounded(e.confidence, 0., 1.)
                || !ids.insert(&e.event_id)
            {
                return Err(Error::Invalid(
                    "valid unique observed social events required",
                ));
            }
        }
        let mut next = self.clone();
        for e in events {
            let old = next
                .entries
                .iter()
                .find(|row| row.target == e.target)
                .cloned();
            let mut attitude = next.status(&e.target, executed_tick)?.attitude;
            let weight = if e.agency && e.benefit != 0. {
                e.confidence
            } else {
                0.
            };
            let agreeable = self.config.personality[3];
            let sensitive = self.config.personality[4];
            let rate = if e.benefit > 0. {
                0.18 + 0.3 * agreeable
            } else {
                0.18 + 0.3 * sensitive + 0.15 * (1. - agreeable)
            };
            attitude += weight * rate * (e.benefit - attitude);
            if weight != 0. {
                next.entries.retain(|row| row.target != e.target);
                next.entries.push(Entry {
                    target: e.target.clone(),
                    attitude,
                    evidence: (old.map_or(0., |row| row.evidence) + weight).min(64.),
                    tick: executed_tick,
                });
                if next.entries.len() > next.capacity {
                    next.entries.remove(0);
                }
            }
            next.events.push((e.event_id.clone(), executed_tick));
            if next.events.len() > 4 * next.capacity {
                next.events.remove(0);
            }
        }
        next.last_tick = Some(executed_tick);
        *self = next;
        Ok(())
    }
    pub fn status(&self, target: &str, tick: u64) -> Result<Status> {
        if !label(target) || tick > MAX_TICK || self.last_tick.is_some_and(|t| tick < t) {
            return Err(Error::Invalid("current social tick required"));
        }
        let (attitude, evidence) = if let Some(e) = self.entries.iter().find(|e| e.target == target)
        {
            let decay = 2f64.powf(-((tick - e.tick) as f64) / self.half_life);
            (e.attitude * decay, e.evidence * decay)
        } else {
            (0., 0.)
        };
        let confidence = evidence / (evidence + 3.);
        let signal = attitude * confidence;
        let stance = if signal.abs() < 0.15 {
            0.
        } else {
            signal.signum() * 0.5
        };
        Ok(Status {
            attitude,
            confidence,
            evidence,
            signal,
            stance,
        })
    }
    pub fn scores(
        &self,
        observation: &Observation,
        bindings: &BTreeMap<String, Binding>,
    ) -> Result<BTreeMap<String, Score>> {
        if observation.tick > MAX_TICK || self.last_tick.is_some_and(|t| observation.tick < t) {
            return Err(Error::Invalid("current social tick required"));
        }
        let mut ids = BTreeSet::new();
        for a in &observation.actions {
            if !ids.insert(&a.id) {
                return Err(Error::Invalid("unique action ids required"));
            }
        }
        let mut result = BTreeMap::new();
        for (id, b) in bindings {
            let a = observation
                .actions
                .iter()
                .find(|a| &a.id == id)
                .ok_or(Error::Invalid("social binding action required"))?;
            if !label(&b.target)
                || a.target.as_deref() != Some(b.target.as_str())
                || !bounded(b.support, -1., 1.)
                || !bounded(b.distress, 0., 1.)
            {
                return Err(Error::Invalid("valid same-recipient binding required"));
            }
            let status = self.status(&b.target, observation.tick)?;
            let weight = 1.2
                * (self.config.values[8]
                    + 0.3 * self.config.personality[3]
                    + 0.5 * self.config.values[4]);
            let concern = if status.attitude < 0. && b.support > 0. {
                1. - b.distress * self.config.values[9]
            } else {
                1.
            };
            let adjustment = if status.stance < 0. && b.support < 0. {
                0.
            } else {
                (weight * status.stance * b.support * concern).clamp(-1.5, 1.5)
            };
            result.insert(
                id.clone(),
                Score {
                    target: b.target.clone(),
                    status,
                    adjustment,
                },
            );
        }
        Ok(result)
    }
    pub fn checkpoint(&self, world: &serde_json::Value) -> Result<Vec<u8>> {
        self.validate_for(&self.config, self.last_tick)?;
        let world = digest(world)?;
        let sha256 = digest(&(&world, self))?;
        Ok(serde_json::to_vec(&Save {
            version: "arena-social-v1".into(),
            world,
            memory: self.clone(),
            sha256,
        })?)
    }
    pub fn restore(
        config: &Config,
        world: &serde_json::Value,
        tick: Option<u64>,
        bytes: &[u8],
    ) -> Result<Self> {
        if bytes.len() > 16 * 1024 * 1024 {
            return Err(Error::Invalid("social checkpoint too large"));
        }
        let save: Save = serde_json::from_slice(bytes)?;
        if save.version != "arena-social-v1"
            || save.world != digest(world)?
            || save.sha256 != digest(&(&save.world, &save.memory))?
        {
            return Err(Error::Invalid("social checkpoint integrity mismatch"));
        }
        save.memory.validate_for(config, tick)?;
        Ok(save.memory)
    }
}
/// Apply bounded social bias inside the baseline's purpose corridor.
pub fn choose(
    decision: &Decision,
    base_id: &str,
    purpose: &BTreeMap<String, f64>,
    bias: &BTreeMap<String, f64>,
    max_regret: f64,
) -> Result<String> {
    if !bounded(max_regret, 0., 2.) || decision.scores.is_empty() {
        return Err(Error::Invalid("valid social comparison required"));
    }
    let mut ids = BTreeSet::new();
    for s in &decision.scores {
        if !label(&s.id)
            || !ids.insert(&s.id)
            || !s.score.is_finite()
            || !purpose.get(&s.id).is_some_and(|v| bounded(*v, -1., 1.))
        {
            return Err(Error::Invalid("complete bounded purpose scores required"));
        }
    }
    if purpose.len() != ids.len()
        || bias
            .iter()
            .any(|(id, v)| !ids.contains(id) || !bounded(*v, -1.5, 1.5))
    {
        return Err(Error::Invalid("bounded social bias required"));
    }
    let base = decision
        .scores
        .iter()
        .find(|s| s.id == base_id && s.eligible)
        .ok_or(Error::Invalid("eligible baseline required"))?;
    if bias.values().all(|v| *v == 0.) {
        return Ok(base.id.clone());
    }
    let best = decision
        .scores
        .iter()
        .filter(|s| s.eligible)
        .map(|s| purpose[&s.id])
        .fold(f64::NEG_INFINITY, f64::max);
    let floor = purpose[base_id].min(best - max_regret);
    let mut selected: Option<(&str, f64)> = None;
    for s in &decision.scores {
        if s.eligible && purpose[&s.id] >= floor - 1e-9 {
            let score = s.score + bias.get(&s.id).copied().unwrap_or(0.);
            if selected.is_none_or(|(_, v)| score > v) {
                selected = Some((&s.id, score));
            }
        }
    }
    Ok(selected
        .ok_or(Error::Invalid("social corridor empty"))?
        .0
        .to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(dead_code)]
    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../reflex-owned/tests/support/mod.rs"
        ));
    }
    fn event(id: &str, tick: u64, benefit: f64) -> Event {
        Event {
            target: "other".into(),
            event_id: id.into(),
            tick,
            benefit,
            agency: true,
            confidence: 1.,
        }
    }
    #[test]
    fn social_atomic_decay_projection_and_checkpoint() {
        let config = fixture::config("actor").actor;
        let mut memory = Memory::with_limits(&config, 2, 24.).unwrap();
        let rows: Vec<_> = (0..8).map(|i| event(&format!("e{i}"), 0, -1.)).collect();
        memory.observe_batch(&rows, 0).unwrap();
        let status = memory.status("other", 0).unwrap();
        assert!(status.stance < 0.);
        assert!(
            (memory.status("other", 24).unwrap().attitude - status.attitude / 2.).abs() < 1e-12
        );
        let before = memory.checkpoint(&serde_json::json!({"turn":0})).unwrap();
        assert!(memory
            .observe_batch(&[event("new", 1, 1.), event("new", 1, 1.)], 1)
            .is_err());
        assert_eq!(
            before,
            memory.checkpoint(&serde_json::json!({"turn":0})).unwrap()
        );
        let mut obs = fixture::input(1).observation;
        obs.actions[0].target = Some("other".into());
        let id = obs.actions[0].id.clone();
        let mut bindings = BTreeMap::from([(
            id.clone(),
            Binding {
                target: "other".into(),
                support: -1.,
                distress: 0.,
            },
        )]);
        assert_eq!(memory.scores(&obs, &bindings).unwrap()[&id].adjustment, 0.);
        bindings.get_mut(&id).unwrap().support = 1.;
        assert!(memory.scores(&obs, &bindings).unwrap()[&id].adjustment <= 0.);
        bindings.get_mut(&id).unwrap().target = "wrong".into();
        assert!(memory.scores(&obs, &bindings).is_err());
        assert!(Memory::restore(&config, &serde_json::json!({"turn":0}), Some(0), &before).is_ok());
        assert!(
            Memory::restore(&config, &serde_json::json!({"turn":1}), Some(0), &before).is_err()
        );
        assert!(
            Memory::restore(&config, &serde_json::json!({"turn":0}), Some(1), &before).is_err()
        );
        memory.observe_batch(&[], 1).unwrap();
        assert!(memory.validate_for(&config, Some(1)).is_ok());
    }
    #[test]
    fn social_choice_preserves_baseline_and_corridor() {
        use reflex_immediate::{ActionScore, State};
        let score = |id: &str, score, eligible| ActionScore {
            id: id.into(),
            score,
            features: [0.; 8],
            eligible,
        };
        let d = Decision {
            action_id: "root".into(),
            target: None,
            reading_used: false,
            request_reading: false,
            next_state: State::default(),
            scores: vec![
                score("a", 1., true),
                score("b", 0.9, true),
                score("c", 2., false),
            ],
            rng: [0.; 3],
        };
        let purpose = BTreeMap::from([("a".into(), 0.8), ("b".into(), 0.5), ("c".into(), 1.)]);
        assert_eq!(
            choose(&d, "a", &purpose, &BTreeMap::new(), 0.1).unwrap(),
            "a"
        );
        let bias = BTreeMap::from([("b".into(), 1.)]);
        assert_eq!(choose(&d, "a", &purpose, &bias, 0.1).unwrap(), "a");
        assert_eq!(choose(&d, "a", &purpose, &bias, 0.4).unwrap(), "b");
    }

    #[test]
    fn social_python_withdrawal_parity() {
        // Values generated by the existing WithdrawalSocialMemory implementation.
        let config = fixture::config("actor").actor;
        let mut m = Memory::with_limits(&config, 2, 24.).unwrap();
        for (tick, benefit, agency, confidence) in [
            (0, -1., true, 1.),
            (1, -0.5, true, 0.6),
            (3, 1., false, 1.),
            (7, 1., true, 0.8),
            (8, -1., true, 1.),
            (9, -1., true, 1.),
            (10, -1., true, 1.),
            (11, -1., true, 1.),
            (12, -1., true, 1.),
            (13, -1., true, 1.),
            (14, -1., true, 1.),
            (15, -1., true, 1.),
        ] {
            m.observe_batch(
                &[Event {
                    target: "partner".into(),
                    event_id: tick.to_string(),
                    tick,
                    benefit,
                    agency,
                    confidence,
                }],
                tick,
            )
            .unwrap();
        }
        for (tick, attitude, evidence, confidence, signal) in [
            (
                15,
                -0.947832571288895,
                10.4,
                0.7761194029850746,
                -0.7356312493585454,
            ),
            (
                39,
                -0.4739162856444475,
                5.2,
                0.6341463414634148,
                -0.300532278701357,
            ),
        ] {
            let s = m.status("partner", tick).unwrap();
            for (actual, expected) in [
                (s.attitude, attitude),
                (s.evidence, evidence),
                (s.confidence, confidence),
                (s.signal, signal),
                (s.stance, -0.5),
            ] {
                assert!((actual - expected).abs() < 1e-12);
            }
            let mut obs = fixture::input(tick).observation;
            obs.actions[0].target = Some("partner".into());
            let mut revenge = obs.actions[0].clone();
            revenge.id = "revenge".into();
            let help = obs.actions[0].id.clone();
            obs.actions.push(revenge);
            let bindings = BTreeMap::from([
                (
                    help.clone(),
                    Binding {
                        target: "partner".into(),
                        support: 1.,
                        distress: 0.8,
                    },
                ),
                (
                    "revenge".into(),
                    Binding {
                        target: "partner".into(),
                        support: -1.,
                        distress: 0.,
                    },
                ),
            ]);
            let scores = m.scores(&obs, &bindings).unwrap();
            assert!((scores[&help].adjustment + 0.324).abs() < 1e-12);
            assert_eq!(scores["revenge"].adjustment, 0.);
        }
    }
}
