use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const TRAITS: [&str; 5] = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
];
pub const NEEDS: [&str; 5] = ["physiology", "safety", "belonging", "esteem", "growth"];
pub const VALUES: [&str; 10] = [
    "self_direction",
    "stimulation",
    "hedonism",
    "achievement",
    "power",
    "security",
    "conformity",
    "tradition",
    "benevolence",
    "universalism",
];

// This order also matches Python's sorted scope keys in the RNG seed digest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scope {
    pub episode: String,
    pub game: String,
    pub npc: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Need {
    Physiology,
    Safety,
    Belonging,
    Esteem,
    Growth,
}

impl Need {
    pub(crate) fn at(index: usize) -> Self {
        [
            Self::Physiology,
            Self::Safety,
            Self::Belonging,
            Self::Esteem,
            Self::Growth,
        ][index]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Need,
    Principle,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Priority {
    Lexicographic,
    Finite,
    Personal { rigidity: f64 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub priority: Priority,
    pub residual: [f64; 8],
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            priority: Priority::Lexicographic,
            residual: [0.; 8],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub scope: Scope,
    pub seed: u64,
    /// In TRAITS order, each in [0, 1].
    pub personality: [f64; 5],
    /// In VALUES order, each in [0, 1].
    pub values: [f64; 10],
    pub objective: String,
    pub policy: Policy,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub primary_need: Option<Need>,
    pub mode: Option<Mode>,
    pub intent_action: Option<String>,
    pub age: u32,
    pub mode_urgency: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeedInput {
    pub supported: bool,
    pub enabled: bool,
    pub deficit: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    pub p: f64,
    pub objective: f64,
    pub needs: [f64; 5],
    pub values: [f64; 10],
    pub style: [f64; 5],
    pub cost: f64,
}

impl Outcome {
    pub(crate) fn effects(&self) -> [f64; 22] {
        let mut row = [0.; 22];
        row[0] = self.objective;
        row[1..6].copy_from_slice(&self.needs);
        row[6..16].copy_from_slice(&self.values);
        row[16..21].copy_from_slice(&self.style);
        row[21] = self.cost;
        row
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Action {
    pub id: String,
    pub target: Option<String>,
    pub legal: bool,
    pub known_failure: bool,
    pub confidence: f64,
    pub familiarity: f64,
    pub switch_cost: f64,
    pub outcomes: Vec<Outcome>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reading {
    pub confidence: f64,
    pub uncertainty: f64,
    pub evidence_ids: Vec<String>,
    pub threatened: bool,
    pub maintains_advantage: bool,
    pub deltas: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub tick: u64,
    pub needs: [NeedInput; 5],
    pub facts: BTreeMap<String, String>,
    pub actions: Vec<Action>,
    pub opponent: Option<Reading>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ActionScore {
    pub id: String,
    pub score: f64,
    pub features: [f64; 8],
    pub eligible: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Decision {
    pub action_id: String,
    pub target: Option<String>,
    pub reading_used: bool,
    pub request_reading: bool,
    pub next_state: State,
    /// Sorted by action ID, independent of caller order.
    pub scores: Vec<ActionScore>,
    pub rng: [f64; 3],
}
