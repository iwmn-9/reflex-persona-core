//! Reference default pressure/progress rules. Fixed traits are never updated.
use crate::{bounded, labels, Error, Result};
use reflex_immediate::Observation;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

fn failure_key(root: &str, condition: &str) -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&[root, condition]).expect("string identity"))
    )
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pressure {
    pub debt: [f64; 5],
    pub stalls: [u32; 5],
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeedFeedback {
    pub progress: [Option<f64>; 5],
    pub maintained: [bool; 5],
    pub completed: [bool; 5],
}
impl Pressure {
    pub fn validate(&self) -> Result<()> {
        for v in self.debt {
            bounded(v, 0., 0.7)?;
        }
        if self.stalls.iter().any(|v| *v > 1_000_000) {
            return Err(Error::Invalid("invalid pressure counters"));
        }
        Ok(())
    }
    pub fn apply(&self, o: &mut Observation) {
        for (n, debt) in o.needs.iter_mut().zip(self.debt) {
            if n.supported && n.enabled {
                n.deficit = n.deficit.map(|v| (v + debt).min(1.));
            }
        }
    }
    pub fn observe(&mut self, fb: &NeedFeedback) -> Result<()> {
        let mut next = self.clone();
        for i in 0..5 {
            let Some(v) = fb.progress[i] else {
                if fb.maintained[i] || fb.completed[i] {
                    return Err(Error::Invalid("need labels require actual observations"));
                }
                continue;
            };
            bounded(v, -1., 1.)?;
            if fb.completed[i] {
                next.debt[i] = 0.;
                next.stalls[i] = 0;
            } else if v > 0.005 {
                next.debt[i] = (next.debt[i] - 0.25).max(0.);
                next.stalls[i] = 0;
            } else if fb.maintained[i] {
                next.stalls[i] = 0;
            } else {
                next.stalls[i] = (next.stalls[i] + 1).min(1_000_000);
                if next.stalls[i] > 3 {
                    next.debt[i] = (next.debt[i] + 0.1).min(0.7);
                }
            }
        }
        *self = next;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Attempt,
    Uncertain,
    Wait,
    Idle,
    Maintain,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Activity {
    pub kind: Kind,
    pub reason: String,
    pub condition: String,
    pub patience: u32,
    pub release: String,
    pub replacement: bool,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Purpose {
    pub level: f64,
    pub readiness: f64,
    pub maintained: bool,
    pub activities: BTreeMap<String, Activity>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PurposeFeedback {
    pub level: f64,
    pub readiness: f64,
    pub maintained: bool,
    pub completed: bool,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Failure {
    pub identity: String,
    pub count: u32,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Progress {
    pub peak: Option<f64>,
    pub ready_peak: Option<f64>,
    pub stalls: u32,
    pub wait_spent: u32,
    pub failures: Vec<Failure>,
}
impl Purpose {
    pub fn validate(&self, roots: &BTreeSet<String>) -> Result<()> {
        bounded(self.level, -1., 1.)?;
        bounded(self.readiness, -1., 1.)?;
        if self.activities.keys().cloned().collect::<BTreeSet<_>>() != *roots {
            return Err(Error::Invalid(
                "purpose activities must cover every current root",
            ));
        }
        for a in self.activities.values() {
            labels(std::slice::from_ref(&a.reason), 1)?;
            labels(std::slice::from_ref(&a.condition), 1)?;
            if !(1..=16).contains(&a.patience) || a.kind == Kind::Wait && a.release.is_empty() {
                return Err(Error::Invalid("finite supported waiting required"));
            }
            if !a.release.is_empty() {
                labels(std::slice::from_ref(&a.release), 1)?;
            }
        }
        Ok(())
    }
}
impl Progress {
    pub fn validate(&self, started: bool) -> Result<()> {
        if !started && self != &Self::default() {
            return Err(Error::Invalid("unobserved purpose history must be empty"));
        }
        if self.peak.is_some() != started
            || self.ready_peak.is_some() != started
            || self.stalls > 1_000_000
            || self.wait_spent > 1_000_000
            || self.failures.len() > 64
        {
            return Err(Error::Invalid("invalid purpose state/timeline"));
        }
        if let Some(v) = self.peak {
            bounded(v, -1., 1.)?;
        }
        if let Some(v) = self.ready_peak {
            bounded(v, -1., 1.)?;
        }
        let mut keys = BTreeSet::new();
        for f in &self.failures {
            if f.identity.len() != 64
                || !f
                    .identity
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                || !(1..=1_000_000).contains(&f.count)
                || !keys.insert(&f.identity)
            {
                return Err(Error::Invalid("invalid purpose failure memory"));
            }
        }
        Ok(())
    }
    fn improving(&self, p: &Purpose) -> bool {
        p.maintained
            || self.peak.is_some_and(|v| {
                p.level > v + 0.005 || p.readiness > self.ready_peak.unwrap() + 0.005
            })
    }
    pub fn allowed(&self, o: &Observation, p: &Purpose) -> Result<BTreeSet<String>> {
        p.validate(&o.actions.iter().map(|a| a.id.clone()).collect())?;
        let improved = self.improving(p);
        let stalls = if improved { 0 } else { self.stalls };
        let spent = if improved { 0 } else { self.wait_spent };
        let viable: BTreeSet<_> = o
            .actions
            .iter()
            .filter(|a| a.legal && !a.known_failure)
            .map(|a| a.id.clone())
            .collect();
        let allowed: BTreeSet<_> = viable
            .iter()
            .filter(|root| {
                let a = &p.activities[*root];
                !(a.kind == Kind::Wait && spent >= a.patience
                    || matches!(a.kind, Kind::Idle | Kind::Maintain) && stalls >= 3
                    || a.kind == Kind::Attempt
                        && !improved
                        && self
                            .failures
                            .iter()
                            .any(|f| f.identity == failure_key(root, &a.condition) && f.count >= 3))
            })
            .cloned()
            .collect();
        Ok(
            if allowed.is_empty() || !allowed.iter().any(|k| p.activities[k].replacement) {
                viable
            } else {
                allowed
            },
        )
    }
    pub fn observe(&mut self, root: &str, p: &Purpose, fb: Option<&PurposeFeedback>) -> Result<()> {
        let a = p
            .activities
            .get(root)
            .ok_or(Error::Invalid("selected purpose activity required"))?;
        if let Some(f) = fb {
            bounded(f.level, -1., 1.)?;
            bounded(f.readiness, -1., 1.)?;
        }
        let mut peak = self.peak.unwrap_or(p.level);
        let mut ready = self.ready_peak.unwrap_or(p.readiness);
        let mut improved = p.maintained || p.level > peak + 0.005 || p.readiness > ready + 0.005;
        if p.level > peak + 0.005 {
            peak = p.level;
        }
        if p.readiness > ready + 0.005 {
            ready = p.readiness;
        }
        if let Some(f) = fb {
            improved |= f.level > peak + 0.005
                || f.readiness > ready + 0.005
                || f.maintained
                || f.completed;
            if f.level > peak + 0.005 {
                peak = f.level;
            }
            if f.readiness > ready + 0.005 {
                ready = f.readiness;
            }
        }
        if improved {
            self.stalls = 0;
            self.wait_spent = 0;
            self.failures.clear();
        } else {
            let restored = fb.is_some_and(|f| {
                f.level >= p.level - 0.005
                    && f.readiness >= p.readiness - 0.005
                    && (f.level > p.level + 0.005 || f.readiness > p.readiness + 0.005)
            });
            if a.kind == Kind::Uncertain && fb.is_some() {
                self.wait_spent = 0;
            }
            if a.kind == Kind::Wait {
                self.wait_spent = (self.wait_spent + 1).min(1_000_000);
            }
            if fb.is_some() {
                self.stalls = (self.stalls + 1).min(1_000_000);
                if a.kind == Kind::Attempt {
                    let previous = self
                        .failures
                        .iter()
                        .position(|f| f.identity == failure_key(root, &a.condition))
                        .map(|i| self.failures.remove(i).count)
                        .unwrap_or(0);
                    if !restored {
                        self.failures.push(Failure {
                            identity: failure_key(root, &a.condition),
                            count: (previous + 1).min(1_000_000),
                        });
                    }
                    if self.failures.len() > 64 {
                        self.failures.remove(0);
                    }
                }
            }
        }
        self.peak = Some(peak);
        self.ready_peak = Some(ready);
        Ok(())
    }
}
