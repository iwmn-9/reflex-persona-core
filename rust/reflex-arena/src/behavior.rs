//! Public, per-person behavior prediction; no private policy or world state.
use reflex_owned::{
    prediction::{Definition, Inputs, Memory},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::{hash_map::DefaultHasher, BTreeMap, BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

/// Fixed meanings for the episode. Actions and suggestions are publicly available.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicBehavior {
    pub alphabet: Vec<String>,
    pub context: String,
    pub actions: BTreeMap<String, String>,
    pub suggestions: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Person {
    definition: Definition,
    memory: Memory,
    last_tick: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSet {
    Adapter,
    AppraisalMixture,
}

/// A viewer's separate factual memories of observed persons. Clone before a trial.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Beliefs {
    version: String,
    scope: String,
    viewer: usize,
    alphabet: Vec<String>,
    model_set: ModelSet,
    people: BTreeMap<usize, Person>,
    #[serde(skip)]
    cache: Arc<Mutex<PredictionCache>>,
}
const CACHE_LIMIT: usize = 256;
#[derive(Debug, Default)]
struct PredictionCache {
    entries: VecDeque<CacheEntry>,
}
#[derive(Debug)]
struct CacheEntry {
    hash: u64,
    definition: Definition,
    memory: Memory,
    inputs: Inputs,
    probabilities: Vec<f64>,
}

fn hash_inputs(input: &Inputs, hash: &mut impl Hasher) {
    input.context.hash(hash);
    input.legal.hash(hash);
    input.models.len().hash(hash);
    for row in &input.models {
        row.len().hash(hash);
        for p in row {
            p.to_bits().hash(hash);
        }
    }
}
fn hash_key(memory: &Memory, input: &Inputs) -> u64 {
    let mut hash = DefaultHasher::new();
    memory.observations.hash(&mut hash);
    memory.history.len().hash(&mut hash);
    for event in &memory.history {
        hash_inputs(&event.inputs, &mut hash);
        event.outcome.hash(&mut hash);
    }
    hash_inputs(input, &mut hash);
    hash.finish()
}
fn inputs_equal(a: &Inputs, b: &Inputs) -> bool {
    a.context == b.context
        && a.legal == b.legal
        && a.models.len() == b.models.len()
        && a.models.iter().zip(&b.models).all(|(a, b)| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        })
}
fn memories_equal(a: &Memory, b: &Memory) -> bool {
    a.observations == b.observations
        && a.history.len() == b.history.len()
        && a.history
            .iter()
            .zip(&b.history)
            .all(|(x, y)| x.outcome == y.outcome && inputs_equal(&x.inputs, &y.inputs))
}
fn label(s: &str) -> bool {
    !s.is_empty() && s.chars().count() <= 128
}
fn alphabet_valid(a: &[String]) -> bool {
    !a.is_empty()
        && a.len() <= 256
        && a.iter().all(|s| label(s))
        && a.windows(2).all(|w| w[0] < w[1])
}
fn definition(model_set: ModelSet) -> Definition {
    Definition {
        model_id: match model_set {
            ModelSet::Adapter => "public-behavior-v2",
            ModelSet::AppraisalMixture => "public-appraisal-mixture-v1",
        }
        .into(),
        factors: vec!["situation".into()],
        names: (0..match model_set {
            ModelSet::Adapter => 3,
            ModelSet::AppraisalMixture => 6,
        })
            .map(|i| i.to_string())
            .collect(),
        window: 64,
    }
}
impl PublicBehavior {
    fn inputs(&self, expected: &[String], model_set: ModelSet) -> Result<Inputs> {
        if self.alphabet != expected
            || !alphabet_valid(&self.alphabet)
            || !label(&self.context)
            || self.actions.is_empty()
            || self.actions.len() > 256
            || self.suggestions.len() != definition(model_set).names.len()
            || self
                .actions
                .iter()
                .any(|(a, c)| !label(a) || self.alphabet.binary_search(c).is_err())
            || self
                .suggestions
                .iter()
                .any(|a| !self.actions.contains_key(a))
        {
            return Err(Error::Invalid("valid public behavior frame required"));
        }
        let available: BTreeSet<_> = self.actions.values().cloned().collect();
        let models = self
            .suggestions
            .iter()
            .map(|suggestion| {
                let suggested = &self.actions[suggestion];
                self.alphabet
                    .iter()
                    .map(|class| {
                        if available.contains(class) {
                            0.1 / available.len() as f64 + if class == suggested { 0.9 } else { 0. }
                        } else {
                            0.
                        }
                    })
                    .collect()
            })
            .collect();
        Ok(Inputs {
            context: vec![self.context.clone()],
            legal: self.alphabet.clone(),
            models,
        })
    }
}
impl Beliefs {
    pub fn observations(&self) -> u64 {
        self.people.values().map(|p| p.memory.observations).sum()
    }
    pub fn new(scope: String, viewer: usize, alphabet: Vec<String>) -> Result<Self> {
        Self::new_with_models(scope, viewer, alphabet, ModelSet::Adapter)
    }
    pub fn new_with_models(
        scope: String,
        viewer: usize,
        alphabet: Vec<String>,
        model_set: ModelSet,
    ) -> Result<Self> {
        if !label(&scope) || viewer >= 10_000 || !alphabet_valid(&alphabet) {
            return Err(Error::Invalid(
                "bounded scope, viewer and sorted alphabet required",
            ));
        }
        Ok(Self {
            version: "public-beliefs-v3".into(),
            scope,
            viewer,
            alphabet,
            model_set,
            people: BTreeMap::new(),
            cache: Arc::default(),
        })
    }
    pub fn model_set(&self) -> ModelSet {
        self.model_set
    }
    /// Validate deserialized state against independently supplied episode identity.
    pub fn validate(&self, scope: &str, viewer: usize, alphabet: &[String]) -> Result<()> {
        if self.version != "public-beliefs-v3"
            || self.scope != scope
            || self.viewer != viewer
            || self.alphabet != alphabet
            || !label(scope)
            || viewer >= 10_000
            || !alphabet_valid(alphabet)
            || self.people.len() > 10_000
        {
            return Err(Error::Invalid("behavior identity mismatch"));
        }
        for (actor, person) in &self.people {
            if *actor >= 10_000
                || *actor == viewer
                || person.definition != definition(self.model_set)
                || (person.memory.observations == 0) != person.last_tick.is_none()
                || person
                    .last_tick
                    .is_some_and(|t| t >= (1_u64 << 63) || t < person.memory.observations - 1)
                || person
                    .memory
                    .history
                    .iter()
                    .any(|e| e.inputs.legal != self.alphabet)
            {
                return Err(Error::Invalid("invalid behavior history"));
            }
            person.memory.validate(&person.definition)?;
        }
        Ok(())
    }
    pub fn checkpoint(&self) -> Result<Vec<u8>> {
        self.validate(&self.scope, self.viewer, &self.alphabet)?;
        Ok(serde_json::to_vec(self)?)
    }
    pub fn restore(bytes: &[u8], scope: &str, viewer: usize, alphabet: &[String]) -> Result<Self> {
        let result: Self = serde_json::from_slice(bytes)?;
        result.validate(scope, viewer, alphabet)?;
        Ok(result)
    }
    pub fn restore_with_models(
        bytes: &[u8],
        scope: &str,
        viewer: usize,
        alphabet: &[String],
        model_set: ModelSet,
    ) -> Result<Self> {
        let result = Self::restore(bytes, scope, viewer, alphabet)?;
        if result.model_set != model_set {
            return Err(Error::Invalid("behavior model set mismatch"));
        }
        Ok(result)
    }
    pub fn predict(&self, frame: &PublicBehavior, actor: usize) -> Result<Vec<(String, f64)>> {
        if actor >= 10_000 || actor == self.viewer {
            return Err(Error::Invalid("bounded observed person required"));
        }
        let inputs = frame.inputs(&self.alphabet, self.model_set)?;
        let d = definition(self.model_set);
        let empty = Memory::default();
        let memory = match self.people.get(&actor) {
            Some(p) => {
                if p.definition != d {
                    return Err(Error::Invalid("behavior model mismatch"));
                }
                &p.memory
            }
            None => &empty,
        };
        let key = hash_key(memory, &inputs);
        let cached = self.cache.lock().ok().and_then(|cache| {
            cache
                .entries
                .iter()
                .find(|e| {
                    e.hash == key
                        && e.definition == d
                        && memories_equal(&e.memory, memory)
                        && inputs_equal(&e.inputs, &inputs)
                })
                .map(|e| e.probabilities.clone())
        });
        let probabilities = if let Some(p) = cached {
            p
        } else {
            let result = memory.predict(&d, &inputs)?;
            if let Ok(mut cache) = self.cache.lock() {
                if cache.entries.len() == CACHE_LIMIT {
                    cache.entries.pop_front();
                }
                cache.entries.push_back(CacheEntry {
                    hash: key,
                    definition: d.clone(),
                    memory: memory.clone(),
                    inputs: inputs.clone(),
                    probabilities: result.clone(),
                });
            }
            result
        };
        let mut result = Vec::with_capacity(frame.actions.len());
        for (action, class) in &frame.actions {
            let index = inputs
                .legal
                .binary_search(class)
                .expect("frame class coverage");
            let count = frame.actions.values().filter(|c| *c == class).count();
            let total = frame
                .suggestions
                .iter()
                .filter(|a| frame.actions[*a] == *class)
                .count();
            let matching = frame.suggestions.iter().filter(|a| *a == action).count();
            let uniform = 1. / count as f64;
            let portion = if total == 0 {
                uniform
            } else {
                0.5 * uniform + 0.5 * matching as f64 / total as f64
            };
            result.push((action.clone(), probabilities[index] * portion));
        }
        let mass: f64 = result.iter().map(|(_, p)| p).sum();
        if mass > 0. {
            for (_, p) in &mut result {
                *p /= mass;
            }
        } else {
            let uniform = 1. / result.len() as f64;
            for (_, p) in &mut result {
                *p = uniform;
            }
        }
        Ok(result)
    }
    /// One real or branch-local observed action per person and world tick.
    pub fn observe(
        &mut self,
        frame: &PublicBehavior,
        actor: usize,
        actual: &str,
        tick: u64,
    ) -> Result<()> {
        if actor >= 10_000
            || actor == self.viewer
            || tick >= (1_u64 << 63)
            || !self.people.contains_key(&actor) && self.people.len() >= 10_000
        {
            return Err(Error::Invalid("bounded observed person and tick required"));
        }
        let inputs = frame.inputs(&self.alphabet, self.model_set)?;
        let class = frame
            .actions
            .get(actual)
            .ok_or(Error::Invalid("actual action must be publicly legal"))?;
        let mut person = self.people.get(&actor).cloned().unwrap_or_else(|| Person {
            definition: definition(self.model_set),
            memory: Memory::default(),
            last_tick: None,
        });
        if person.definition != definition(self.model_set)
            || person.last_tick.is_some_and(|last| tick <= last)
        {
            return Err(Error::Invalid("behavior observation clock must advance"));
        }
        person.memory.observe(&person.definition, &inputs, class)?;
        person.last_tick = Some(tick);
        self.people.insert(actor, person);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame() -> PublicBehavior {
        PublicBehavior {
            alphabet: vec!["advance".into(), "avoid".into(), "support".into()],
            context: "open".into(),
            actions: BTreeMap::from([
                ("a".into(), "advance".into()),
                ("b".into(), "avoid".into()),
                ("c".into(), "support".into()),
                ("d".into(), "support".into()),
            ]),
            suggestions: vec!["a".into(), "a".into(), "a".into()],
        }
    }
    fn new() -> Beliefs {
        Beliefs::new("g/e".into(), 0, frame().alphabet).unwrap()
    }
    fn probability(b: &Beliefs, actor: usize, action: &str) -> f64 {
        b.predict(&frame(), actor)
            .unwrap()
            .into_iter()
            .find(|(a, _)| a == action)
            .unwrap()
            .1
    }
    #[test]
    fn separate_people_learn_and_unmodeled_class_is_retained() {
        let mut b = new();
        let initial = probability(&b, 2, "c") + probability(&b, 2, "d");
        for tick in 0..40 {
            b.observe(&frame(), 1, "a", tick).unwrap();
            b.observe(&frame(), 2, "c", tick).unwrap();
        }
        assert!(probability(&b, 1, "a") > probability(&b, 2, "a"));
        assert!(probability(&b, 2, "c") + probability(&b, 2, "d") > initial);
        let rows = b.predict(&frame(), 2).unwrap();
        assert_eq!(rows.len(), 4);
        assert!((rows.iter().map(|(_, p)| p).sum::<f64>() - 1.).abs() < 1e-12);
        assert!(rows.iter().all(|(_, p)| *p >= 0.));
    }
    #[test]
    fn change_points_clone_and_restore_are_isolated() {
        let mut b = new();
        for tick in 0..40 {
            b.observe(&frame(), 1, "a", tick).unwrap();
        }
        let before = probability(&b, 1, "c");
        let original = b.checkpoint().unwrap();
        let mut branch = b.clone();
        for tick in 40..100 {
            branch.observe(&frame(), 1, "c", tick).unwrap();
        }
        assert!(probability(&branch, 1, "c") > before);
        assert_eq!(b.checkpoint().unwrap(), original);
        assert!(b.observe(&frame(), 1, "a", 39).is_err());
        let scope = "g/e";
        let restored = Beliefs::restore(&original, scope, 0, &frame().alphabet).unwrap();
        assert_eq!(restored.checkpoint().unwrap(), original);
        assert!(Beliefs::restore(&original, scope, 3, &frame().alphabet).is_err());
        assert!(Beliefs::restore(&original, "g/other", 0, &frame().alphabet).is_err());
    }
    #[test]
    fn invalid_frames_and_repeated_person_tick_are_rejected_without_mutation() {
        let mut b = new();
        let mut invalid = frame();
        invalid.suggestions[0] = "missing".into();
        assert!(b.observe(&invalid, 1, "a", 0).is_err());
        b.observe(&frame(), 1, "a", 0).unwrap();
        let before = b.checkpoint().unwrap();
        assert!(b.observe(&frame(), 1, "b", 0).is_err());
        assert_eq!(b.checkpoint().unwrap(), before);
    }

    #[test]
    fn temporarily_unavailable_class_retains_the_same_history_when_it_returns() {
        let mut learned = new();
        for tick in 0..30 {
            learned.observe(&frame(), 1, "a", tick).unwrap();
        }
        let mut restricted = frame();
        restricted.actions.remove("a");
        restricted.suggestions = vec!["b".into(), "b".into(), "b".into()];
        let input = restricted
            .inputs(&restricted.alphabet, ModelSet::Adapter)
            .unwrap();
        assert_eq!(input.legal, frame().alphabet);
        assert!(input.models.iter().all(|row| row[0] == 0.));
        for tick in 30..35 {
            learned.observe(&restricted, 1, "b", tick).unwrap();
        }
        let only_legal = learned.predict(&restricted, 1).unwrap();
        assert!((only_legal.iter().map(|(_, p)| p).sum::<f64>() - 1.).abs() < 1e-12);
        assert!(only_legal.iter().all(|(a, _)| a != "a"));
        assert!(probability(&learned, 1, "a") > probability(&new(), 1, "a"));
        assert!(learned.people[&1]
            .memory
            .history
            .iter()
            .all(|e| e.inputs.legal == frame().alphabet));
    }

    #[test]
    fn cache_is_shared_bounded_and_not_part_of_saved_or_learned_state() {
        let mut b = new();
        for tick in 0..12 {
            b.observe(&frame(), 1, "a", tick).unwrap();
        }
        let saved = b.checkpoint().unwrap();
        let first = b.predict(&frame(), 1).unwrap();
        let branch = b.clone();
        assert!(Arc::ptr_eq(&b.cache, &branch.cache));
        assert_eq!(first, branch.predict(&frame(), 1).unwrap());
        assert_eq!(saved, b.checkpoint().unwrap());
        let restored = Beliefs::restore(&saved, "g/e", 0, &frame().alphabet).unwrap();
        assert!(restored.cache.lock().unwrap().entries.is_empty());
        assert_eq!(first, restored.predict(&frame(), 1).unwrap());
        for i in 0..(CACHE_LIMIT + 5) {
            let mut varied = frame();
            varied.context = format!("situation-{i}");
            b.predict(&varied, 1).unwrap();
        }
        assert_eq!(b.cache.lock().unwrap().entries.len(), CACHE_LIMIT);
        assert_eq!(saved, b.checkpoint().unwrap());
    }

    #[test]
    fn cache_hash_collision_cannot_change_prediction() {
        let mut b = new();
        b.observe(&frame(), 1, "a", 0).unwrap();
        let input = frame().inputs(&b.alphabet, ModelSet::Adapter).unwrap();
        let key = hash_key(&b.people[&1].memory, &input);
        b.cache.lock().unwrap().entries.push_back(CacheEntry {
            hash: key,
            definition: definition(ModelSet::Adapter),
            memory: Memory::default(),
            inputs: input.clone(),
            probabilities: vec![0., 0., 1.],
        });
        let expected =
            Beliefs::restore(&b.checkpoint().unwrap(), "g/e", 0, &frame().alphabet).unwrap();
        assert_eq!(
            b.predict(&frame(), 1).unwrap(),
            expected.predict(&frame(), 1).unwrap()
        );
    }

    #[test]
    fn six_models_use_the_same_class_memory_and_action_distribution() {
        let mut f = frame();
        f.suggestions = vec![
            "a".into(),
            "b".into(),
            "c".into(),
            "d".into(),
            "b".into(),
            "a".into(),
        ];
        let mut b = Beliefs::new_with_models(
            "g/e".into(),
            0,
            f.alphabet.clone(),
            ModelSet::AppraisalMixture,
        )
        .unwrap();
        let mut reference = Memory::default();
        let d = definition(ModelSet::AppraisalMixture);
        let inputs = f.inputs(&f.alphabet, ModelSet::AppraisalMixture).unwrap();
        assert_eq!(inputs.models.len(), 6);
        for tick in 0..20 {
            let actual = if tick % 3 == 0 { "a" } else { "c" };
            b.observe(&f, 1, actual, tick).unwrap();
            reference.observe(&d, &inputs, &f.actions[actual]).unwrap();
        }
        let classes = reference.predict(&d, &inputs).unwrap();
        let rows = b.predict(&f, 1).unwrap();
        for (index, class) in f.alphabet.iter().enumerate() {
            let sum: f64 = rows
                .iter()
                .filter(|(a, _)| &f.actions[a] == class)
                .map(|(_, p)| p)
                .sum();
            assert!((sum - classes[index]).abs() < 1e-12);
        }
        let c = rows.iter().find(|(a, _)| a == "c").unwrap().1;
        let d = rows.iter().find(|(a, _)| a == "d").unwrap().1;
        assert!((c - d).abs() < 1e-12);
        let saved = b.checkpoint().unwrap();
        let restored =
            Beliefs::restore_with_models(&saved, "g/e", 0, &f.alphabet, ModelSet::AppraisalMixture)
                .unwrap();
        assert_eq!(restored.predict(&f, 1).unwrap(), rows);
        assert!(
            Beliefs::restore_with_models(&saved, "g/e", 0, &f.alphabet, ModelSet::Adapter).is_err()
        );
        assert!(new().predict(&f, 1).is_err());
        let mut three = f.clone();
        three.suggestions.truncate(3);
        assert!(b.predict(&three, 1).is_err());
        assert!(b.observe(&f, 1, "a", 19).is_err());
        assert_eq!(b.checkpoint().unwrap(), saved);
    }

    #[test]
    fn adapter_and_repeated_six_preserve_predictions_and_reject_old_schema() {
        let f = frame();
        let mut six = f.clone();
        six.suggestions.extend(f.suggestions.clone());
        let mut adapter = new();
        let mut mixture = Beliefs::new_with_models(
            "g/e".into(),
            0,
            f.alphabet.clone(),
            ModelSet::AppraisalMixture,
        )
        .unwrap();
        assert_eq!(adapter.model_set(), ModelSet::Adapter);
        for tick in 0..20 {
            let actual = if tick % 2 == 0 { "a" } else { "c" };
            adapter.observe(&f, 1, actual, tick).unwrap();
            mixture.observe(&six, 1, actual, tick).unwrap();
        }
        for ((a, p), (b, q)) in adapter
            .predict(&f, 1)
            .unwrap()
            .iter()
            .zip(mixture.predict(&six, 1).unwrap())
        {
            assert_eq!(*a, b);
            assert!((p - q).abs() < 1e-12);
        }
        let mut state: serde_json::Value =
            serde_json::from_slice(&adapter.checkpoint().unwrap()).unwrap();
        state.as_object_mut().unwrap().remove("version");
        assert!(
            Beliefs::restore(&serde_json::to_vec(&state).unwrap(), "g/e", 0, &f.alphabet).is_err()
        );
        let mut state: serde_json::Value =
            serde_json::from_slice(&adapter.checkpoint().unwrap()).unwrap();
        state["people"]["1"]["last_tick"] = (1_u64 << 63).into();
        assert!(
            Beliefs::restore(&serde_json::to_vec(&state).unwrap(), "g/e", 0, &f.alphabet).is_err()
        );
    }
}
