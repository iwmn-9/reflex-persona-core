//! Before-update diagnostics. In the public-conditional profile these warnings
//! do not revoke the empirical fallback or modify the segmented mixture.
use crate::{experience, Error, Result};
use reflex_immediate::Outcome;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Trial,
    Active,
    Revoked,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub key: String,
    pub state: Status,
    pub gains: Vec<f64>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    pub entries: Vec<Entry>,
}
fn hash(v: &impl Serialize) -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(v).expect("string only identity"))
    )
}
fn mean(xs: &[f64]) -> f64 {
    crate::prediction::sum(xs) / xs.len() as f64
}
impl Evidence {
    pub fn validate(&self) -> Result<()> {
        let mut keys = std::collections::BTreeSet::new();
        if self.entries.len() > 64 {
            return Err(Error::Invalid("bounded evidence memory required"));
        }
        for e in &self.entries {
            crate::labels(std::slice::from_ref(&e.key), 1)?;
            if !keys.insert(&e.key)
                || e.gains.is_empty()
                || e.gains.len() > 12
                || e.state == Status::Active && e.gains.len() < 4
            {
                return Err(Error::Invalid("valid evidence state and trials required"));
            }
            for x in &e.gains {
                crate::bounded(*x, -1., 1.)?;
            }
        }
        Ok(())
    }
    fn update(&mut self, key: String, gain: f64) {
        let mut e = self
            .entries
            .iter()
            .position(|e| e.key == key)
            .map(|i| self.entries.remove(i))
            .unwrap_or(Entry {
                key,
                state: Status::Trial,
                gains: Vec::new(),
            });
        e.gains.push(gain);
        if e.gains.len() > 12 {
            e.gains.remove(0);
        }
        if e.gains.len() >= 4 {
            let recent = mean(&e.gains[e.gains.len() - 4..]);
            let all = mean(&e.gains);
            if recent < -0.02 {
                e.state = Status::Revoked;
            } else if recent > 0.02 && all > 0.02 {
                e.state = Status::Active;
            } else if e.state == Status::Active && all <= 0. {
                e.state = Status::Trial;
            }
        }
        self.entries.push(e);
        if self.entries.len() > 64 {
            self.entries.remove(0);
        }
    }
    pub fn response(&mut self, predicted: &[f64], observed: usize) {
        let uniform = 1. / predicted.len() as f64;
        if predicted.iter().all(|v| *v == uniform) {
            return;
        }
        let prior: Vec<_> = (0..predicted.len())
            .map(|i| (uniform - f64::from(i == observed)).powi(2))
            .collect();
        let candidate: Vec<_> = predicted
            .iter()
            .enumerate()
            .map(|(i, p)| (p - f64::from(i == observed)).powi(2))
            .collect();
        self.update(
            format!("response:{}", hash(&"event-response")),
            mean(&prior) - mean(&candidate),
        );
    }
    pub fn experience(
        &mut self,
        b: &experience::Binding,
        prior: &[Outcome],
        candidate: &[Outcome],
        observed: &[f64; 22],
    ) {
        if b.estimated.is_empty() {
            return;
        }
        let thresholds = [-0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75];
        let cdf = |rows: &[Outcome]| {
            b.estimated
                .iter()
                .flat_map(|j| {
                    thresholds.iter().map(move |t| {
                        rows.iter()
                            .map(|r| r.p * f64::from(experience::vector(r)[*j] <= *t))
                            .sum::<f64>()
                    })
                })
                .collect::<Vec<_>>()
        };
        let a = cdf(prior);
        let b_cdf = cdf(candidate);
        if a == b_cdf {
            return;
        }
        let actual: Vec<_> = b
            .estimated
            .iter()
            .flat_map(|j| {
                thresholds
                    .iter()
                    .map(move |t| f64::from(observed[*j] <= *t))
            })
            .collect();
        let loss = |p: &[f64]| {
            mean(
                &p.iter()
                    .zip(&actual)
                    .map(|(x, y)| (x - y).powi(2))
                    .collect::<Vec<_>>(),
            )
        };
        self.update(
            format!("experience:{}", hash(&[&b.method, &b.situation])),
            loss(&a) - loss(&b_cdf),
        );
    }
}
