//! Two small rule adapters. Their effects and legality belong here, never in the runtime.
use reflex_runtime::{immediate::*, *};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Resources,
    Combat,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Body {
    pub health: i32,
    pub energy: i32,
    pub wealth: i32,
    pub ammo: i32,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DemoWorld {
    pub tick: u64,
    pub kind: Kind,
    pub bodies: BTreeMap<String, Body>,
    pub stock: i32,
    pub rng: u64,
}
pub fn session(kind: Kind) -> SessionId {
    SessionId {
        game: format!("demo-{kind:?}"),
        episode: "continuing".into(),
    }
}
pub fn configs(kind: Kind, count: usize) -> Vec<Config> {
    let s = session(kind);
    (0..count)
        .map(|i| Config {
            scope: Scope {
                game: s.game.clone(),
                episode: s.episode.clone(),
                npc: format!("npc-{i:04}"),
            },
            seed: 7831 + i as u64,
            personality: if i % 2 == 0 {
                [0.7, 0.8, 0.3, 0.9, 0.6]
            } else {
                [0.5, 0.4, 0.8, 0.2, 0.3]
            },
            values: if i % 2 == 0 {
                [0.2, 0.1, 0.3, 0.4, 0.1, 0.7, 0.2, 0.2, 0.8, 0.7]
            } else {
                [0.6, 0.7, 0.4, 0.8, 0.8, 0.3, 0.1, 0.1, 0.2, 0.1]
            },
            objective: "make-progress".into(),
            policy: Policy {
                priority: Priority::Finite,
                ..Policy::default()
            },
        })
        .collect()
}
pub fn initial(kind: Kind, configs: &[Config]) -> DemoWorld {
    DemoWorld {
        tick: 0,
        kind,
        stock: 2,
        rng: 198732,
        bodies: configs
            .iter()
            .map(|c| {
                (
                    c.scope.npc.clone(),
                    Body {
                        health: 12,
                        energy: 6,
                        wealth: 0,
                        ammo: 2,
                    },
                )
            })
            .collect(),
    }
}
fn action(
    id: &str,
    target: Option<String>,
    legal: bool,
    objective: f64,
    needs: [f64; 5],
    values: [f64; 10],
) -> Action {
    Action {
        id: id.into(),
        target,
        legal,
        known_failure: false,
        confidence: 1.,
        familiarity: 0.,
        switch_cost: 0.,
        outcomes: vec![Outcome {
            p: 1.,
            objective,
            needs,
            values,
            style: [0.; 5],
            cost: 0.,
        }],
    }
}
impl DemoWorld {
    fn observation(&self, npc: &str) -> reflex_runtime::Result<Option<Observation>> {
        let body = self
            .bodies
            .get(npc)
            .ok_or(reflex_runtime::Error::Invalid("missing body"))?;
        if self.kind == Kind::Combat
            && (body.health <= 0 || self.bodies.values().filter(|b| b.health > 0).count() < 2)
        {
            return Ok(None);
        }
        let actions = match self.kind {
            Kind::Resources => {
                let target = self
                    .bodies
                    .iter()
                    .filter(|(id, _)| id.as_str() != npc)
                    .min_by_key(|(_, b)| b.wealth)
                    .map(|(id, _)| id.clone());
                let mut generosity = [0.; 10];
                generosity[8] = 0.8;
                vec![
                    action(
                        "work",
                        None,
                        body.energy >= 2 && self.stock > 0,
                        0.5,
                        [-0.2, 0., 0., 0.2, 0.3],
                        [0.; 10],
                    ),
                    action(
                        "rest",
                        None,
                        true,
                        0.,
                        [if body.energy < 10 { 0.4 } else { 0. }, 0., 0., 0., 0.],
                        [0.; 10],
                    ),
                    action(
                        "share",
                        target.clone(),
                        body.wealth >= 2 && target.is_some(),
                        -0.2,
                        [0., 0., 0.4, 0., 0.],
                        generosity,
                    ),
                ]
            }
            Kind::Combat => {
                let mut actions = vec![
                    action("cover", None, true, 0., [0., 0.3, 0., 0., 0.], [0.; 10]),
                    action(
                        "reload",
                        None,
                        body.ammo < 3,
                        0.1,
                        [0., 0., 0., 0., 0.15],
                        [0.; 10],
                    ),
                    action(
                        "recover",
                        None,
                        body.health < 12,
                        0.1,
                        [0.25, 0.1, 0., 0., 0.],
                        [0.; 10],
                    ),
                ];
                for (id, other) in &self.bodies {
                    if id != npc && other.health > 0 {
                        let mut shot = action(
                            &format!("fire-{id}"),
                            Some(id.clone()),
                            body.ammo > 0,
                            0.5,
                            [0., 0.2, 0., 0.2, 0.1],
                            [0.; 10],
                        );
                        shot.outcomes[0].p = 0.65;
                        shot.outcomes.push(Outcome {
                            p: 0.35,
                            objective: 0.,
                            needs: [0.; 5],
                            values: [0.; 10],
                            style: [0.; 5],
                            cost: 0.1,
                        });
                        actions.push(shot);
                    }
                }
                actions
            }
        };
        let deficits = match self.kind {
            Kind::Resources => [(10 - body.energy) as f64 / 10., 0.1, 0.4, 0.3, 0.5],
            Kind::Combat => [(12 - body.health) as f64 / 12., 0.5, 0.1, 0.4, 0.3],
        };
        Ok(Some(Observation {
            tick: self.tick,
            needs: std::array::from_fn(|i| NeedInput {
                supported: true,
                enabled: true,
                deficit: Some(deficits[i]),
            }),
            facts: BTreeMap::from([(
                "visible-state".into(),
                format!(
                    "tick={}, health={}, energy={}, stock={}",
                    self.tick, body.health, body.energy, self.stock
                ),
            )]),
            actions,
            opponent: None,
        }))
    }
}
impl World for DemoWorld {
    const SAVE_FORMAT: &'static str = "rust-demo-world-v1";
    fn revision(&self) -> u64 {
        self.tick
    }
    fn observe(&self, config: &Config) -> reflex_runtime::Result<Option<Observation>> {
        self.observation(&config.scope.npc)
    }
    fn advance(&self, choices: &[Choice<'_>]) -> reflex_runtime::Result<(Self, Vec<Receipt>)> {
        let mut next = self.clone();
        let mut receipts = Vec::new();
        let mut ordered: Vec<_> = choices.iter().collect();
        // A rotating simultaneous resource tie-break is a world rule, not NPC logic.
        if !ordered.is_empty() {
            let n = ordered.len();
            ordered.rotate_left(self.tick as usize % n);
        }
        for choice in ordered {
            let original = self
                .bodies
                .get(choice.npc)
                .ok_or(reflex_runtime::Error::Invalid("unknown body"))?;
            let observation =
                self.observation(choice.npc)?
                    .ok_or(reflex_runtime::Error::Invalid(
                        "inactive actor received a choice",
                    ))?;
            if !observation.actions.iter().any(|a| {
                a.id == choice.decision.action_id && a.target == choice.decision.target && a.legal
            }) {
                return Err(reflex_runtime::Error::Invalid("illegal world action"));
            }
            let id = choice.decision.action_id.as_str();
            let mut executed = true;
            match self.kind {
                Kind::Resources => match id {
                    "work" if next.stock > 0 => {
                        next.stock -= 1;
                        let b = next.bodies.get_mut(choice.npc).unwrap();
                        b.wealth += 3;
                        b.energy -= 2;
                    }
                    "work" => executed = false,
                    "rest" => {
                        let b = next.bodies.get_mut(choice.npc).unwrap();
                        b.energy = (b.energy + 3).min(10);
                    }
                    "share" => {
                        next.bodies.get_mut(choice.npc).unwrap().wealth -= 2;
                        next.bodies
                            .get_mut(choice.decision.target.as_ref().unwrap())
                            .unwrap()
                            .wealth += 2;
                    }
                    _ => return Err(reflex_runtime::Error::Invalid("unknown resource action")),
                },
                Kind::Combat => match id {
                    "cover" => {}
                    "reload" => next.bodies.get_mut(choice.npc).unwrap().ammo = 3,
                    "recover" => {
                        let b = next.bodies.get_mut(choice.npc).unwrap();
                        b.health += (original.health + 2).min(12) - original.health;
                    }
                    _ => {
                        next.bodies.get_mut(choice.npc).unwrap().ammo -= 1;
                        next.rng ^= next.rng << 13;
                        next.rng ^= next.rng >> 7;
                        next.rng ^= next.rng << 17;
                        if next.rng % 100 < 65 {
                            let target = choice.decision.target.as_ref().unwrap();
                            let covering = choices
                                .iter()
                                .any(|c| c.npc == target && c.decision.action_id == "cover");
                            next.bodies.get_mut(target).unwrap().health -=
                                if covering { 1 } else { 3 };
                        }
                    }
                },
            }
            receipts.push(Receipt {
                npc: choice.npc.into(),
                execution: if executed {
                    Execution::Executed {
                        action_id: choice.decision.action_id.clone(),
                        target: choice.decision.target.clone(),
                    }
                } else {
                    Execution::NotExecuted {
                        reason: "resource consumed by another actor".into(),
                    }
                },
            });
        }
        for body in next.bodies.values_mut() {
            body.health = body.health.clamp(0, 12);
        }
        next.tick += 1;
        next.stock = 2;
        Ok((next, receipts))
    }
}
