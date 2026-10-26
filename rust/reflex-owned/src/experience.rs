//! Selected factual outcomes only; bounded empirical fallback from judgment.py.
use crate::{bounded, labels, Error, Result};
use reflex_immediate::{Observation, Outcome};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub fn vector(e: &Outcome) -> [f64; 22] {
    let mut v = [0.; 22];
    v[0] = e.objective;
    v[1..6].copy_from_slice(&e.needs);
    v[6..16].copy_from_slice(&e.values);
    v[16..21].copy_from_slice(&e.style);
    v[21] = e.cost;
    v
}
fn packet(v: [f64; 22], p: f64) -> Outcome {
    let v = v.map(|v| {
        if v > 1. && v <= 1. + 1e-12 {
            1.
        } else if (-1. - 1e-12..-1.).contains(&v) {
            -1.
        } else {
            v
        }
    });
    Outcome {
        p: if p > 1. && p <= 1. + 1e-12 { 1. } else { p },
        objective: v[0],
        needs: v[1..6].try_into().unwrap(),
        values: v[6..16].try_into().unwrap(),
        style: v[16..21].try_into().unwrap(),
        cost: v[21],
    }
}
pub fn validate_vector(v: &[f64; 22]) -> Result<()> {
    for x in v {
        bounded(*x, -1., 1.)?;
    }
    bounded(v[21], 0., 1.)
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub method: String,
    pub situation: String,
    pub estimated: Vec<usize>,
}
impl Binding {
    pub fn validate(&self) -> Result<()> {
        labels(std::slice::from_ref(&self.method), 1)?;
        labels(std::slice::from_ref(&self.situation), 1)?;
        if self.estimated.iter().any(|i| *i >= 22)
            || self
                .estimated
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.estimated.len()
        {
            return Err(Error::Invalid(
                "distinct estimated feature indices required",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub binding: Binding,
    pub count: u32,
    pub samples: Vec<[f64; 22]>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Memory {
    pub entries: Vec<Entry>,
}
impl Memory {
    pub fn validate(&self) -> Result<()> {
        if self.entries.len() > 64 {
            return Err(Error::Invalid("experience capacity exceeded"));
        }
        let mut keys = std::collections::BTreeSet::new();
        for e in &self.entries {
            e.binding.validate()?;
            if e.binding.estimated.is_empty()
                || !keys.insert((&e.binding.method, &e.binding.situation))
                || e.samples.is_empty()
                || e.samples.len() > 4
                || e.count < e.samples.len() as u32
                || e.count > 12
            {
                return Err(Error::Invalid("invalid factual experience entry"));
            }
            for v in &e.samples {
                validate_vector(v)?;
            }
        }
        Ok(())
    }
    fn find(&self, b: &Binding) -> Option<usize> {
        self.entries
            .iter()
            .position(|e| e.binding.method == b.method && e.binding.situation == b.situation)
    }
    pub(crate) fn prepare(
        &self,
        o: &Observation,
        bindings: &BTreeMap<String, Binding>,
    ) -> Result<Observation> {
        self.validate()?;
        if bindings.len() != o.actions.len() {
            return Err(Error::Invalid("bind every action"));
        }
        let mut out = o.clone();
        for a in &mut out.actions {
            let b = bindings
                .get(&a.id)
                .ok_or(Error::Invalid("missing action binding"))?;
            b.validate()?;
            let Some(i) = self.find(b) else { continue };
            let e = &self.entries[i];
            if e.binding.estimated != b.estimated {
                return Err(Error::Invalid("method semantics changed without a new key"));
            }
            if e.count < 2 || b.estimated.is_empty() {
                continue;
            }
            let weight = (e.count as f64 / (e.count + 3) as f64).min(0.75);
            let mut rows = Vec::new();
            for row in &a.outcomes {
                rows.push(packet(vector(row), row.p * (1. - weight)));
                for sample in &e.samples {
                    let mut v = vector(row);
                    for j in &b.estimated {
                        v[*j] = sample[*j];
                    }
                    rows.push(packet(v, row.p * weight / e.samples.len() as f64));
                }
            }
            a.outcomes = compress(rows, &b.estimated)?;
        }
        Ok(out)
    }
    pub(crate) fn observe(
        &mut self,
        binding: &Binding,
        predicted: &[Outcome],
        observed: &[f64; 22],
    ) -> Result<()> {
        self.validate()?;
        binding.validate()?;
        validate_vector(observed)?;
        if binding.estimated.is_empty() {
            return Ok(());
        }
        let mut entry = if let Some(i) = self.find(binding) {
            self.entries.remove(i)
        } else {
            Entry {
                binding: binding.clone(),
                count: 0,
                samples: Vec::new(),
            }
        };
        let distance = predicted
            .iter()
            .filter(|r| r.p > 0.)
            .map(|r| {
                let v = vector(r);
                binding
                    .estimated
                    .iter()
                    .map(|j| (observed[*j] - v[*j]).abs())
                    .fold(0., f64::max)
            })
            .fold(f64::INFINITY, f64::min);
        if !distance.is_finite() {
            return Err(Error::Invalid("nonempty predicted support required"));
        }
        if entry.count >= 3 && distance > 0.6 {
            entry.count = 0;
            entry.samples.clear();
        }
        entry.count = (entry.count + 1).min(12);
        entry.samples.push(*observed);
        if entry.samples.len() > 4 {
            entry.samples.remove(0);
        }
        self.entries.push(entry);
        if self.entries.len() > 64 {
            self.entries.remove(0);
        }
        Ok(())
    }
}

fn compress(rows: Vec<Outcome>, estimated: &[usize]) -> Result<Vec<Outcome>> {
    if rows.len() <= 8 {
        return Ok(rows);
    }
    let fixed: Vec<_> = (0..22).filter(|i| !estimated.contains(i)).collect();
    let mut groups: Vec<(Vec<f64>, Vec<Outcome>)> = Vec::new();
    for row in rows.into_iter().filter(|r| r.p > 0.) {
        let v = vector(&row);
        let key: Vec<_> = fixed.iter().map(|j| v[*j]).collect();
        let index = groups
            .iter()
            .position(|(k, _)| *k == key)
            .unwrap_or_else(|| {
                groups.push((key, Vec::new()));
                groups.len() - 1
            });
        let group = &mut groups[index].1;
        if let Some(old) = group.iter_mut().find(|r| vector(r) == v) {
            old.p += row.p;
        } else {
            group.push(row);
        }
    }
    if groups.len() > 8 {
        return Err(Error::Invalid("non-estimated branches exceed capacity"));
    }
    let mut slots = vec![1; groups.len()];
    let mass: Vec<f64> = groups
        .iter()
        .map(|(_, rows)| rows.iter().map(|r| r.p).sum())
        .collect();
    while slots.iter().sum::<usize>() < 8 {
        let mut best = None;
        for i in 0..groups.len() {
            if slots[i] < groups[i].1.len()
                && best.is_none_or(|j| mass[i] / slots[i] as f64 > mass[j] / slots[j] as f64)
            {
                best = Some(i);
            }
        }
        if let Some(i) = best {
            slots[i] += 1;
        } else {
            break;
        }
    }
    let mut out = Vec::new();
    for ((key, mut rows), limit) in groups.into_iter().zip(slots) {
        if rows.len() > limit {
            rows.sort_by(|a, b| a.objective.total_cmp(&b.objective));
            let start = usize::from(limit > 1);
            let mut compressed = if start == 1 {
                vec![rows[0].clone()]
            } else {
                Vec::new()
            };
            let buckets = (limit - 1).max(1);
            let n = rows.len() - start;
            let mut at = start;
            for i in 0..buckets {
                let size = n / buckets + usize::from(i < n % buckets);
                let bucket = &rows[at..at + size];
                at += size;
                let p: f64 = bucket.iter().map(|r| r.p).sum();
                if p == 0. {
                    continue;
                }
                let mut v = [0.; 22];
                for row in bucket {
                    for (a, x) in v.iter_mut().zip(vector(row)) {
                        *a += row.p * x;
                    }
                }
                for x in &mut v {
                    *x /= p;
                }
                compressed.push(packet(v, p));
            }
            rows = compressed;
        }
        for row in rows {
            let mut v = vector(&row);
            for (j, x) in fixed.iter().zip(&key) {
                v[*j] = *x;
            }
            out.push(packet(v, row.p));
        }
    }
    Ok(out)
}
