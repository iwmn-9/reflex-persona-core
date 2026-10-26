//! Finite-history Bayesian segment mixture, matching segmented_prediction.py.
//! Parameters are the reference prior, not game/personality-specific tuning.
use crate::{bounded, labels, Error, Result};
use serde::{Deserialize, Serialize};

const HAZARDS: [f64; 4] = [0., 1. / 8., 1. / 32., 1. / 128.];
type Group<'a> = ((Vec<String>, Vec<String>), Vec<&'a Evidence>);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Definition {
    pub model_id: String,
    pub factors: Vec<String>,
    pub names: Vec<String>,
    pub window: usize,
}
impl Definition {
    pub fn validate(&self) -> Result<()> {
        labels(std::slice::from_ref(&self.model_id), 1)?;
        labels(&self.factors, 4)?;
        labels(&self.names, 16)?;
        if !(2..=512).contains(&self.window) {
            return Err(Error::Invalid("retained event window must be in 2..512"));
        }
        Ok(())
    }
}

/// Rows follow Definition.names; columns follow sorted legal symbols.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inputs {
    pub context: Vec<String>,
    pub legal: Vec<String>,
    pub models: Vec<Vec<f64>>,
}
impl Inputs {
    pub fn validate(&self, definition: &Definition) -> Result<()> {
        if self.context.len() != definition.factors.len()
            || self
                .context
                .iter()
                .any(|s| s.is_empty() || s.chars().count() > 128)
            || self.models.len() != definition.names.len()
        {
            return Err(Error::Invalid(
                "complete declared public factors and models required",
            ));
        }
        labels(&self.legal, 256)?;
        if self.legal.is_empty() || self.legal.windows(2).any(|p| p[0] >= p[1]) {
            return Err(Error::Invalid("nonempty sorted response alphabet required"));
        }
        for row in &self.models {
            if row.len() != self.legal.len() {
                return Err(Error::Invalid("complete model response row required"));
            }
            for p in row {
                bounded(*p, 0., 1.)?;
            }
            if (sum(row) - 1.).abs() > 1e-8 {
                return Err(Error::Invalid("normalized public response model required"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    pub inputs: Inputs,
    pub outcome: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Memory {
    pub observations: u64,
    pub history: Vec<Evidence>,
}

impl Memory {
    pub fn validate(&self, d: &Definition) -> Result<()> {
        d.validate()?;
        if self.observations >= (1_u64 << 63)
            || self.history.len() != self.observations.min(d.window as u64) as usize
        {
            return Err(Error::Invalid(
                "complete bounded factual event history required",
            ));
        }
        for e in &self.history {
            e.inputs.validate(d)?;
            if !e.inputs.legal.contains(&e.outcome) {
                return Err(Error::Invalid("observed response outside declared rules"));
            }
        }
        Ok(())
    }
    /// Transactional, single observed response. Missing feedback must not call this.
    pub fn observe(&mut self, d: &Definition, inputs: &Inputs, outcome: &str) -> Result<()> {
        self.validate(d)?;
        inputs.validate(d)?;
        if !inputs.legal.iter().any(|s| s == outcome) || self.observations == (1_u64 << 63) - 1 {
            return Err(Error::Invalid(
                "bounded actual observation within the known alphabet required",
            ));
        }
        self.history.push(Evidence {
            inputs: inputs.clone(),
            outcome: outcome.into(),
        });
        if self.history.len() > d.window {
            self.history.remove(0);
        }
        self.observations += 1;
        Ok(())
    }
    pub fn predict(&self, d: &Definition, current: &Inputs) -> Result<Vec<f64>> {
        self.validate(d)?;
        current.validate(d)?;
        let mut predictions = Vec::new();
        let mut logs = Vec::new();
        // Python itertools.combinations order: size first, lexicographic within size.
        let mut masks: Vec<usize> = (0..(1 << d.factors.len())).collect();
        masks.sort_by_key(|m| {
            (
                m.count_ones(),
                (0..d.factors.len())
                    .filter(|i| m & (1 << i) != 0)
                    .collect::<Vec<_>>(),
            )
        });
        for mask in masks {
            let key = |i: &Inputs| {
                (
                    i.legal.clone(),
                    i.context
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| mask & (1 << j) != 0)
                        .map(|(_, v)| v.clone())
                        .collect::<Vec<_>>(),
                )
            };
            let mut groups: Vec<Group<'_>> = Vec::new();
            for e in &self.history {
                let k = key(&e.inputs);
                if let Some((_, rows)) = groups.iter_mut().find(|(g, _)| *g == k) {
                    rows.push(e);
                } else {
                    groups.push((k, vec![e]));
                }
            }
            let target = key(current);
            let mut probability = None;
            let mut log = -(mask.count_ones() as f64) * 2_f64.ln();
            for (k, rows) in groups {
                let model = if k == target {
                    current
                } else {
                    &rows.last().expect("nonempty").inputs
                };
                let (p, evidence) = segment(&rows, model);
                log += evidence;
                if k == target {
                    probability = Some(p);
                }
            }
            predictions.push(probability.unwrap_or_else(|| segment(&[], current).0));
            logs.push(log);
        }
        let weights = normalize_logs(&logs);
        let mut result = vec![0.; current.legal.len()];
        for (p, weight) in predictions.iter().zip(weights) {
            for (x, y) in result.iter_mut().zip(p) {
                *x += weight * y;
            }
        }
        normalize(&mut result);
        Ok(result)
    }
}

// NumPy contiguous pairwise add reduction (not a mathematical reassociation).
pub(crate) fn sum(xs: &[f64]) -> f64 {
    if xs.len() < 8 {
        return xs.iter().fold(-0., |a, x| a + x);
    }
    if xs.len() <= 128 {
        let mut lanes = [0.; 8];
        lanes.copy_from_slice(&xs[..8]);
        let mut i = 8;
        while i + 8 <= xs.len() {
            for j in 0..8 {
                lanes[j] += xs[i + j];
            }
            i += 8;
        }
        let mut out = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        for x in &xs[i..] {
            out += x;
        }
        out
    } else {
        let middle = (xs.len() / 2) / 8 * 8;
        sum(&xs[..middle]) + sum(&xs[middle..])
    }
}
fn normalize(p: &mut [f64]) {
    let total = sum(p);
    for v in p {
        *v /= total;
    }
}
fn normalize_logs(logs: &[f64]) -> Vec<f64> {
    let best = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut result: Vec<_> = logs.iter().map(|v| (v - best).exp()).collect();
    normalize(&mut result);
    result
}

fn segment(rows: &[&Evidence], current: &Inputs) -> (Vec<f64>, f64) {
    let alphabet = current.legal.len();
    let libraries = current.models.len();
    let mut prior = vec![0.5 / libraries.max(1) as f64; libraries];
    prior.push(if libraries == 0 { 1. } else { 0.5 });
    let mut predictions = Vec::new();
    let mut evidence = Vec::new();
    for hazard in HAZARDS {
        let mut weights: Vec<Vec<f64>> = Vec::new();
        let mut counts: Vec<Vec<f64>> = Vec::new();
        let mut logp = 0.;
        for (i, e) in rows.iter().enumerate() {
            let z = e
                .inputs
                .legal
                .iter()
                .position(|s| s == &e.outcome)
                .expect("validated");
            let mut reset = prior.clone();
            for (j, value) in reset.iter_mut().enumerate() {
                *value *= if j < libraries {
                    e.inputs.models[j][z]
                } else {
                    1. / alphabet as f64
                };
                *value *= if i == 0 { 1. } else { hazard };
            }
            for (row, count) in weights.iter_mut().zip(&counts) {
                let n = sum(count);
                for (j, value) in row.iter_mut().enumerate() {
                    let p = if j < libraries {
                        e.inputs.models[j][z]
                    } else {
                        (count[z] + 0.5) / (n + 0.5 * alphabet as f64)
                    };
                    *value = (1. - hazard) * *value * p;
                }
            }
            weights.insert(0, reset);
            let norm = sum(&weights.iter().flatten().copied().collect::<Vec<_>>());
            logp += norm.ln();
            for row in &mut weights {
                for p in row {
                    *p /= norm;
                }
            }
            counts.insert(0, vec![0.; alphabet]);
            for count in &mut counts {
                count[z] += 1.;
            }
        }
        let mut base = vec![prior[libraries] / alphabet as f64; alphabet];
        for (weight, row) in prior.iter().zip(&current.models) {
            for (v, p) in base.iter_mut().zip(row) {
                *v += weight * p;
            }
        }
        let mut p = if weights.is_empty() {
            base.clone()
        } else {
            base.iter().map(|v| hazard * v).collect()
        };
        if !weights.is_empty() {
            for z in 0..alphabet {
                let mut library_p = 0.;
                for j in 0..libraries {
                    let mass: f64 = weights.iter().map(|r| r[j]).sum();
                    library_p += mass * current.models[j][z];
                }
                p[z] += (1. - hazard) * library_p;
                let unknown: f64 = weights
                    .iter()
                    .zip(&counts)
                    .map(|(w, c)| w[libraries] * ((c[z] + 0.5) / (sum(c) + 0.5 * alphabet as f64)))
                    .sum();
                p[z] += (1. - hazard) * unknown;
            }
        }
        predictions.push(p);
        evidence.push(logp);
    }
    let posterior = normalize_logs(&evidence);
    let mut result = vec![0.; alphabet];
    for (p, w) in predictions.iter().zip(posterior) {
        for (x, y) in result.iter_mut().zip(p) {
            *x += w * y;
        }
    }
    normalize(&mut result);
    let maximum = evidence.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log = maximum + (evidence.iter().map(|v| (v - maximum).exp()).sum::<f64>() / 4.).ln();
    (result, log)
}
