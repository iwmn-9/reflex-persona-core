use crate::{
    social::{Binding as SocialBinding, Event as SocialEvent},
    ArenaGame,
};
use reflex_immediate::{Action, NeedInput, Observation, Outcome};
use reflex_owned::{prediction, progress, Error, Feedback, Input, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn err() -> Error {
    Error::Invalid("invalid arena action or state")
}
fn rnd(seed: u64, n: u64) -> u64 {
    let mut x = seed.wrapping_add(n.wrapping_mul(0x9e3779b97f4a7c15));
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn clamp(x: f64) -> f64 {
    x.clamp(-1., 1.)
}
fn row(o: &Outcome) -> [f64; 22] {
    let mut a = [0.; 22];
    a[0] = o.objective;
    a[1..6].copy_from_slice(&o.needs);
    a[6..16].copy_from_slice(&o.values);
    a[16..21].copy_from_slice(&o.style);
    a[21] = o.cost;
    a
}
trait World: ArenaGame {
    fn legal(&self, p: usize) -> Vec<String>;
    fn route_context(&self, _p: usize) -> Option<&str> {
        None
    }
    fn behavior_alphabet(&self) -> Vec<String>;
    fn deficits(&self, p: usize) -> [Option<f64>; 5];
    fn transition(&self, c: &[String], seed: u64) -> Result<Self>;
    fn effect(&self, next: &Self, p: usize, action: &str) -> Outcome;
    fn model_choice(&self, player: usize, hypothesis: usize) -> String;
    fn facts(&self, p: usize) -> BTreeMap<String, String>;
    fn binding(&self, p: usize, action: &str) -> reflex_owned::experience::Binding;
    fn readiness(&self, p: usize) -> f64 {
        self.purpose(p)
    }
}
fn identified_response(viewer: usize, actual: &[String], models: &[Vec<String>]) -> Option<String> {
    let mut matching = models.iter().enumerate().filter(|(_, choices)| {
        choices.len() == actual.len()
            && choices
                .iter()
                .zip(actual)
                .enumerate()
                .all(|(p, (expected, observed))| p == viewer || expected == observed)
    });
    let (index, _) = matching.next()?;
    if matching.next().is_some() {
        None
    } else {
        Some(index.to_string())
    }
}
fn feedback<W: World>(before: &W, after: &W, c: &[String]) -> Vec<Feedback> {
    (0..before.players())
        .map(|p| {
            let o = before.effect(after, p, &c[p]);
            let pd = before.deficits(p);
            let nd = after.deficits(p);
            let progress = std::array::from_fn(|i| match (pd[i], nd[i]) {
                (Some(a), Some(b)) => Some(clamp(a - b)),
                _ => None,
            });
            let models: Vec<_> = (0..3)
                .map(|h| before.modeled_choices(p, &h.to_string()).unwrap())
                .collect();
            let response = identified_response(p, c, &models);
            Feedback {
                observed: Some(row(&o)),
                response,
                needs: Some(progress::NeedFeedback {
                    progress,
                    maintained: [false; 5],
                    completed: [false; 5],
                }),
                purpose: Some(progress::PurposeFeedback {
                    level: after.purpose(p),
                    readiness: after.readiness(p),
                    maintained: false,
                    completed: after.terminal(),
                }),
            }
        })
        .collect()
}
fn make_input<W: World>(w: &W, p: usize) -> Result<Input> {
    if p >= w.players() || w.terminal() {
        return Err(err());
    }
    let choices = (0..3)
        .map(|h| w.modeled_choices(p, &h.to_string()))
        .collect::<Result<Vec<_>>>()?;
    make_input_scenarios(w, p, &choices)
}
fn make_input_scenarios<W: World>(w: &W, p: usize, choices: &[Vec<String>]) -> Result<Input> {
    if p >= w.players()
        || w.terminal()
        || choices.len() != 3
        || choices.iter().any(|row| {
            row.len() != w.players()
                || row
                    .iter()
                    .enumerate()
                    .any(|(actor, action)| actor != p && !w.legal(actor).contains(action))
        })
    {
        return Err(err());
    }
    let legal = w.legal(p);
    let social = w.social_bindings(p)?;
    let mut actions = Vec::new();
    let mut conditional = BTreeMap::new();
    for a in legal {
        let mut outcomes = Vec::new();
        for (h, scenario) in choices.iter().enumerate() {
            let mut c = scenario.clone();
            c[p] = a.clone();
            let model = w.model_world(p, 0x12345678 + h as u64)?;
            let next = model.transition(&c, 0x12345678 + h as u64)?;
            let mut o = w.effect(&next, p, &a);
            o.p = 1. / 3.;
            outcomes.push(o);
        }
        conditional.insert(a.clone(), outcomes.clone());
        actions.push(Action {
            target: social.get(&a).map(|b| b.target.clone()),
            id: a,
            legal: true,
            known_failure: false,
            confidence: 1.,
            familiarity: 0.,
            switch_cost: 0.,
            outcomes,
        });
    }
    let bindings: BTreeMap<_, _> = actions
        .iter()
        .map(|a| {
            let mut binding = w.binding(p, &a.id);
            if let Some(route) = w.route_context(p) {
                binding.situation.push_str(":route:");
                binding.situation.push_str(route);
            }
            (a.id.clone(), binding)
        })
        .collect();
    let activities = actions
        .iter()
        .map(|a| {
            (
                a.id.clone(),
                progress::Activity {
                    kind: progress::Kind::Attempt,
                    reason: "game action".into(),
                    condition: bindings[&a.id].situation.clone(),
                    patience: 3,
                    release: String::new(),
                    replacement: true,
                },
            )
        })
        .collect();
    let deficits = w.deficits(p);
    Ok(Input {
        observation: Observation {
            tick: w.tick(),
            needs: std::array::from_fn(|i| NeedInput {
                supported: deficits[i].is_some(),
                enabled: deficits[i].is_some(),
                deficit: deficits[i],
            }),
            facts: w.facts(p),
            actions,
            opponent: None,
        },
        bindings,
        // These labels index forecast samples, not world outcomes. Actual play
        // resolves every independently controlled participant's chosen action.
        responses: prediction::Inputs {
            context: vec![],
            legal: vec!["0".into(), "1".into(), "2".into()],
            models: vec![
                vec![0.8, 0.1, 0.1],
                vec![0.1, 0.8, 0.1],
                vec![0.1, 0.1, 0.8],
            ],
        },
        conditional,
        purpose: progress::Purpose {
            level: w.purpose(p),
            readiness: w.readiness(p),
            maintained: false,
            activities,
        },
        threatened: false,
        maintains_advantage: false,
    })
}
fn binding(method: String, situation: String) -> reflex_owned::experience::Binding {
    reflex_owned::experience::Binding {
        method,
        situation,
        estimated: vec![0, 1, 2, 3, 4, 5, 9, 10, 11, 14, 15],
    }
}
fn bucket(value: f64, low: f64, high: f64) -> &'static str {
    if value < low {
        "low"
    } else if value < high {
        "mid"
    } else {
        "high"
    }
}
fn outcome(
    objective: f64,
    needs: [f64; 5],
    values: [f64; 10],
    style: [f64; 5],
    cost: f64,
) -> Outcome {
    Outcome {
        p: 1.,
        objective: clamp(objective),
        needs: needs.map(clamp),
        values: values.map(clamp),
        style: style.map(clamp),
        cost: cost.clamp(0., 1.),
    }
}
fn hypotheses<W: World>(w: &W, viewer: usize, response: &str) -> Result<Vec<String>> {
    if viewer >= w.players() {
        return Err(err());
    }
    let h: usize = response.parse().map_err(|_| err())?;
    if h > 2 {
        return Err(err());
    }
    Ok((0..w.players()).map(|p| w.model_choice(p, h)).collect())
}
macro_rules! common {
    ($t:ty) => {
        fn input(&self, p: usize) -> Result<Input> {
            make_input(self, p)
        }
        fn input_scenarios(&self, p: usize, choices: &[Vec<String>]) -> Result<Input> {
            make_input_scenarios(self, p, choices)
        }
        fn behavior(&self, viewer: usize, actor: usize) -> Result<crate::behavior::PublicBehavior> {
            if viewer >= self.players() || actor >= self.players() || self.terminal() {
                return Err(err());
            }
            let legal = self.legal(actor);
            let actions = legal
                .iter()
                .map(|a| (a.clone(), self.binding(actor, a).method))
                .collect();
            Ok(crate::behavior::PublicBehavior {
                alphabet: self.behavior_alphabet(),
                context: self.binding(actor, &legal[0]).situation,
                actions,
                suggestions: (0..3).map(|h| self.model_choice(actor, h)).collect(),
            })
        }
        fn advance(&self, c: &[String], seed: u64) -> Result<(Self, Vec<Feedback>)> {
            let next = self.transition(c, seed)?;
            let f = feedback(self, &next, c);
            Ok((next, f))
        }
        fn modeled_choices(&self, p: usize, r: &str) -> Result<Vec<String>> {
            hypotheses(self, p, r)
        }
    };
}

#[derive(Clone, Serialize)]
pub struct Hagetaka {
    hands: Vec<Vec<i32>>,
    scores: Vec<i32>,
    prize: i32,
    remaining: Vec<i32>,
    carry: i32,
    round: u64,
    last: Vec<i32>,
    #[serde(skip)]
    deck: Vec<i32>,
}
impl Hagetaka {
    pub fn new(seed: u64) -> Self {
        let mut deck: Vec<i32> = (-5..0).chain(1..=10).collect();
        for i in (1..deck.len()).rev() {
            let j = (rnd(seed, i as u64) % (i as u64 + 1)) as usize;
            deck.swap(i, j)
        }
        let prize = deck.remove(0);
        let mut remaining = deck.clone();
        remaining.sort_unstable();
        Self {
            hands: vec![(1..=15).collect(); 4],
            scores: vec![0; 4],
            prize,
            remaining,
            carry: 0,
            round: 0,
            last: vec![],
            deck,
        }
    }
}
impl ArenaGame for Hagetaka {
    fn partners(&self, _: usize) -> Vec<usize> {
        vec![]
    }
    fn offers(&self, announced: &[String]) -> Result<Vec<crate::coordination::Offer>> {
        if announced.len() != self.players() {
            return Err(err());
        }
        Ok(vec![])
    }
    fn players(&self) -> usize {
        4
    }
    fn tick(&self) -> u64 {
        self.round
    }
    fn terminal(&self) -> bool {
        self.round >= 15
    }
    common!(Hagetaka);
    fn model_world(&self, viewer: usize, seed: u64) -> Result<Self> {
        if viewer >= self.players() {
            return Err(err());
        }
        let mut model = self.clone();
        model.deck = self.remaining.clone();
        for i in (1..model.deck.len()).rev() {
            let j = (rnd(seed, i as u64) % (i as u64 + 1)) as usize;
            model.deck.swap(i, j);
        }
        Ok(model)
    }
    fn purpose(&self, p: usize) -> f64 {
        (0.5 + (self.scores[p] as f64 - self.scores.iter().sum::<i32>() as f64 / 4.) / 140.)
            .clamp(0., 1.)
    }
    fn view(&self) -> Value {
        json!(self)
    }
}
impl World for Hagetaka {
    fn behavior_alphabet(&self) -> Vec<String> {
        ["bid:high", "bid:low", "bid:mid"]
            .map(str::to_owned)
            .to_vec()
    }
    fn model_choice(&self, p: usize, h: usize) -> String {
        let pot = self.prize + self.carry;
        let nominal = if pot >= 0 {
            (pot as f64 * 1.4).round() as i32
        } else {
            15 - pot.abs()
        };
        let observed = self.last.get(p).copied().unwrap_or(nominal);
        let target = match h {
            0 => nominal - 2 + (p as i32 % 3),
            1 => nominal + (p as i32 - 1),
            _ => observed + if pot < 0 { 1 } else { 2 } + (p as i32 % 2),
        };
        let bid = self.hands[p]
            .iter()
            .min_by_key(|&&x| ((x - target).abs(), x))
            .unwrap();
        format!("{bid:02}")
    }
    fn facts(&self, p: usize) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("round".into(), self.round.to_string()),
            ("pot".into(), (self.prize + self.carry).to_string()),
            ("hand".into(), format!("{:?}", self.hands[p])),
            ("scores".into(), format!("{:?}", self.scores)),
        ])
    }
    fn binding(&self, p: usize, action: &str) -> reflex_owned::experience::Binding {
        let bid = action.parse::<f64>().unwrap_or(0.);
        let pot = self.prize + self.carry;
        let situation = format!(
            "pot:{}-{}:cards:{}",
            if pot < 0 { "negative" } else { "nonnegative" },
            bucket(pot.abs() as f64, 4., 9.),
            bucket(self.hands[p].len() as f64, 5., 10.)
        );
        binding(format!("bid:{}", bucket(bid, 6., 11.)), situation)
    }
    fn legal(&self, p: usize) -> Vec<String> {
        self.hands[p].iter().map(|x| format!("{x:02}")).collect()
    }
    fn deficits(&self, _: usize) -> [Option<f64>; 5] {
        [None, Some(0.2), None, Some(0.3), None]
    }
    fn transition(&self, c: &[String], _: u64) -> Result<Self> {
        if self.terminal() || c.len() != 4 {
            return Err(err());
        }
        let mut bids = Vec::new();
        for (p, a) in c.iter().enumerate() {
            if !self.legal(p).contains(a) {
                return Err(err());
            }
            bids.push(a.parse::<i32>().map_err(|_| err())?)
        }
        let mut n = self.clone();
        let pot = self.carry + self.prize;
        let mut unique: Vec<usize> = (0..4)
            .filter(|&i| bids.iter().filter(|&&b| b == bids[i]).count() == 1)
            .collect();
        unique.sort_by_key(|&i| bids[i]);
        let winner = if pot < 0 {
            unique.first()
        } else {
            unique.last()
        };
        if let Some(&i) = winner {
            n.scores[i] += pot;
            n.carry = 0
        } else {
            n.carry = pot
        }
        for (i, &bid) in bids.iter().enumerate() {
            n.hands[i].retain(|&b| b != bid);
        }
        n.last = bids;
        n.round += 1;
        if !n.deck.is_empty() {
            n.prize = n.deck.remove(0);
            n.remaining.retain(|&x| x != n.prize)
        }
        Ok(n)
    }
    fn effect(&self, n: &Self, p: usize, a: &str) -> Outcome {
        let gain = (n.scores[p] - self.scores[p]) as f64 / 70.;
        let relative = self.purpose(p);
        let obj = n.purpose(p) - relative;
        let bid = a.parse::<f64>().unwrap_or(0.);
        let safety = -(bid / 15.) * (self.remaining.len() as f64 / 15.) * 0.12 - (-gain).max(0.);
        let mut v = [0.; 10];
        v[3] = gain;
        v[4] = obj;
        v[5] = safety;
        outcome(
            obj,
            [0., safety, 0., gain, 0.],
            v,
            [bid / 300., -bid / 300., 0., 0., 0.],
            0.,
        )
    }
}

#[derive(Clone, PartialEq, Serialize)]
struct Person {
    energy: f64,
    health: f64,
    stock: f64,
    score: f64,
}
#[derive(Clone, PartialEq, Serialize)]
pub struct Encounters {
    people: Vec<Person>,
    history: Vec<Vec<String>>,
    tick: u64,
    limit: u64,
}
impl Encounters {
    pub fn new(_seed: u64, limit: u64) -> Self {
        Self {
            people: vec![
                Person {
                    energy: 0.8,
                    health: 8.,
                    stock: 3.,
                    score: 0.
                };
                4
            ],
            history: vec![vec![]; 4],
            tick: 0,
            limit,
        }
    }
}
impl ArenaGame for Encounters {
    fn social_bindings(&self, player: usize) -> Result<BTreeMap<String, SocialBinding>> {
        if player >= self.players() || self.terminal() {
            return Err(err());
        }
        let mut bindings = BTreeMap::new();
        for action in self.legal(player) {
            if let Some((kind, target)) = action.split_once(':') {
                if kind == "cooperate" || kind == "claim" {
                    let target: usize = target.parse().map_err(|_| err())?;
                    bindings.insert(
                        action.clone(),
                        SocialBinding {
                            target: target.to_string(),
                            support: if kind == "cooperate" { 1. } else { -1. },
                            distress: (1. - self.people[target].health / 8.).clamp(0., 1.),
                        },
                    );
                }
            }
        }
        Ok(bindings)
    }
    fn social_events(
        &self,
        next: &Self,
        choices: &[String],
        seed: u64,
    ) -> Result<Vec<Vec<SocialEvent>>> {
        let (resolved, events) = self.resolved(choices, seed, true)?;
        if resolved != *next {
            return Err(err());
        }
        Ok(events)
    }

    fn partners(&self, player: usize) -> Vec<usize> {
        if player >= self.players() {
            return vec![];
        }
        (0..self.players()).filter(|&p| p != player).collect()
    }
    fn offers(&self, announced: &[String]) -> Result<Vec<crate::coordination::Offer>> {
        if self.terminal()
            || announced.len() != self.players()
            || announced
                .iter()
                .enumerate()
                .any(|(p, a)| !self.legal(p).contains(a))
        {
            return Err(err());
        }
        let mut offers = Vec::new();
        for a in 0..4 {
            for b in a + 1..4 {
                let left = format!("cooperate:{b}");
                let right = format!("cooperate:{a}");
                if self.legal(a).contains(&left) && self.legal(b).contains(&right) {
                    offers.push(crate::coordination::Offer {
                        id: format!("cooperate:{a}:{b}"),
                        roots: BTreeMap::from([(a, left), (b, right)]),
                    });
                }
            }
        }
        Ok(offers)
    }
    fn players(&self) -> usize {
        4
    }
    fn tick(&self) -> u64 {
        self.tick
    }
    fn terminal(&self) -> bool {
        self.tick >= self.limit
    }
    common!(Encounters);
    fn model_world(&self, viewer: usize, _: u64) -> Result<Self> {
        if viewer >= self.players() {
            return Err(err());
        }
        Ok(self.clone())
    }
    fn purpose(&self, p: usize) -> f64 {
        (0.5 + self.people[p].score / 40.).clamp(0., 1.)
    }
    fn view(&self) -> Value {
        json!(self)
    }
}
impl Encounters {
    fn resolved(
        &self,
        c: &[String],
        seed: u64,
        collect: bool,
    ) -> Result<(Self, Vec<Vec<SocialEvent>>)> {
        if self.terminal() || c.len() != 4 || (0..4).any(|p| !self.legal(p).contains(&c[p])) {
            return Err(err());
        }
        let mut events = if collect {
            vec![vec![]; self.players()]
        } else {
            vec![]
        };
        let mut n = self.clone();
        for p in 0..4 {
            let a = &c[p];
            if a == "finished" {
                continue;
            }
            let (gain, health) = if a == "rest" {
                (0.08, 0.)
            } else if a == "alone" {
                (0.3, 0.06)
            } else {
                let (kind, t) = a.split_once(':').ok_or_else(err)?;
                let target: usize = t.parse().map_err(|_| err())?;
                let other = &c[target];
                let matched = other.ends_with(&format!(":{p}"));
                if !matched {
                    (0.12, 0.)
                } else {
                    let other_coop = other.starts_with("cooperate:");
                    let accident = kind == "cooperate"
                        && other_coop
                        && rnd(
                            seed,
                            self.tick * 16 + p.min(target) as u64 * 4 + p.max(target) as u64,
                        ) % 100
                            < 5;
                    if collect {
                        events[p].push(SocialEvent {
                            target: target.to_string(),
                            event_id: format!("{}:{p}:{target}", self.tick),
                            tick: self.tick,
                            agency: !accident,
                            confidence: 1.,
                            benefit: if accident {
                                0.
                            } else if other_coop {
                                0.8
                            } else if kind == "cooperate" {
                                -1.
                            } else {
                                0.
                            },
                        });
                    }
                    if accident {
                        (-0.1, -0.15)
                    } else {
                        match (kind, other_coop) {
                            ("cooperate", true) => (0.55, 0.12),
                            ("cooperate", false) => (-0.15, -0.1),
                            (_, true) => (0.8, 0.08),
                            _ => (0.05, -0.02),
                        }
                    }
                }
            };
            let u = &mut n.people[p];
            u.energy = if a == "rest" {
                (u.energy + 0.4).min(1.)
            } else {
                (u.energy - 0.18).max(0.)
            };
            u.energy = (u.energy - 0.025).max(0.);
            u.stock = (u.stock + gain - 0.08).max(0.);
            u.health = ((u.health + health).clamp(0., 8.) - 0.04).max(0.);
            u.score += gain;
        }
        for (p, choice) in c.iter().enumerate() {
            n.history[p].push(choice.clone());
            if n.history[p].len() > 8 {
                n.history[p].remove(0);
            }
        }
        n.tick += 1;
        Ok((n, events))
    }
}
impl World for Encounters {
    fn behavior_alphabet(&self) -> Vec<String> {
        ["alone", "claim", "cooperate", "finished", "rest"]
            .map(str::to_owned)
            .to_vec()
    }
    fn model_choice(&self, p: usize, h: usize) -> String {
        let legal = self.legal(p);
        if h == 2 {
            return if legal.contains(&"alone".to_string()) && self.people[p].energy >= 0.5 {
                "alone".into()
            } else if legal.contains(&"rest".to_string()) {
                "rest".into()
            } else {
                legal[0].clone()
            };
        }
        // Last publicly initiated contact, otherwise deterministic rotating pairs.
        let prior = self.history[p]
            .last()
            .and_then(|a| a.split_once(':'))
            .and_then(|(_, v)| v.parse::<usize>().ok())
            .filter(|&q| q != p && self.people[q].health > 0.);
        let pairings = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]];
        let partner = prior.unwrap_or_else(|| {
            let pairs = pairings[(self.tick as usize) % 3];
            pairs
                .iter()
                .find_map(|&(a, b)| {
                    if a == p {
                        Some(b)
                    } else if b == p {
                        Some(a)
                    } else {
                        None
                    }
                })
                .unwrap()
        });
        let candidate = format!("{}:{partner}", if h == 0 { "cooperate" } else { "claim" });
        if legal.contains(&candidate) {
            candidate
        } else if legal.contains(&"rest".into()) {
            "rest".into()
        } else {
            legal[0].clone()
        }
    }
    fn facts(&self, p: usize) -> BTreeMap<String, String> {
        let u = &self.people[p];
        BTreeMap::from([
            ("tick".into(), self.tick.to_string()),
            ("energy".into(), format!("{:.2}", u.energy)),
            ("health".into(), format!("{:.2}", u.health)),
            ("stock".into(), format!("{:.2}", u.stock)),
            (
                "last_choices".into(),
                format!(
                    "{:?}",
                    self.history
                        .iter()
                        .map(|h| h.last().map(String::as_str).unwrap_or("none"))
                        .collect::<Vec<_>>()
                ),
            ),
        ])
    }
    fn binding(&self, p: usize, action: &str) -> reflex_owned::experience::Binding {
        let u = &self.people[p];
        let mut parts = action.split(':');
        let method = parts.next().unwrap_or(action);
        let partner = parts
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .and_then(|i| self.history[i].last())
            .map(|a| a.split(':').next().unwrap_or("none"))
            .unwrap_or("none");
        binding(
            method.into(),
            format!(
                "energy:{}:health:{}:stock:{}:partner:{}",
                bucket(u.energy, 0.3, 0.7),
                bucket(u.health, 3., 6.),
                bucket(u.stock, 1., 3.),
                partner
            ),
        )
    }
    fn legal(&self, p: usize) -> Vec<String> {
        let u = &self.people[p];
        if u.health <= 0. {
            return vec!["finished".into()];
        }
        let mut l = vec!["rest".into()];
        if u.energy >= 0.18 {
            l.push("alone".into());
            for i in 0..4 {
                if i != p && self.people[i].health > 0. {
                    l.push(format!("cooperate:{i}"));
                    l.push(format!("claim:{i}"));
                }
            }
        }
        l.sort();
        l
    }
    fn deficits(&self, p: usize) -> [Option<f64>; 5] {
        let u = &self.people[p];
        [
            Some(1. - u.energy),
            Some((1. - u.health / 8.).max(((2. - u.stock) / 2.).max(0.))),
            Some(0.3),
            Some(0.2),
            Some(0.3),
        ]
    }
    fn transition(&self, c: &[String], seed: u64) -> Result<Self> {
        self.resolved(c, seed, false).map(|(world, _)| world)
    }
    fn effect(&self, n: &Self, p: usize, a: &str) -> Outcome {
        let old = &self.people[p];
        let u = &n.people[p];
        let gain = u.score - old.score;
        // Safety means observed health/reserve adequacy, regardless of method.
        // Use the same deficit as factual need feedback, not a bonus for working alone.
        let safety = self.deficits(p)[1].unwrap() - n.deficits(p)[1].unwrap();
        let mut needs = [(u.energy - old.energy), safety, 0., 0., 0.];
        let mut v = [0.; 10];
        v[5] = safety;
        let mut style = [0.; 5];
        if let Some(target) = a.strip_prefix("cooperate:") {
            let t: usize = target.parse().unwrap();
            let matched = n.history[t].last().unwrap().ends_with(&format!(":{p}"));
            if matched {
                needs[2] = 0.3;
                v[8] = 0.7;
                v[9] = 0.3;
            }
            style[3] = 0.6;
        } else if let Some(target) = a.strip_prefix("claim:") {
            let t: usize = target.parse().unwrap();
            let other = n.history[t].last().unwrap();
            if other.ends_with(&format!(":{p}")) {
                needs[3] = 0.2;
                v[4] = 0.9;
                v[3] = 0.35;
                if other.starts_with("cooperate:") {
                    v[8] = -0.6;
                }
            }
            style[2] = 0.5;
        } else if a == "alone" {
            needs[4] = 0.3;
            v[0] = 0.3;
            v[3] = 0.4;
            style[1] = 0.4;
        }
        outcome(
            gain,
            needs,
            v,
            style,
            if a == "rest" || a == "finished" {
                0.
            } else {
                0.05
            },
        )
    }
}
#[derive(Clone, PartialEq, Serialize)]
struct Unit {
    team: usize,
    x: i32,
    y: i32,
    hp: i32,
    ammo: i32,
    kit: i32,
    guard: bool,
}
#[derive(Clone, PartialEq, Serialize)]
pub struct Combat {
    units: Vec<Unit>,
    walls: Vec<(i32, i32)>,
    covers: Vec<(i32, i32)>,
    goal: String,
    hold: [u32; 2],
    secured: [bool; 2],
    tick: u64,
    limit: u64,
    #[serde(skip)]
    routes: [Option<String>; 6],
}
impl Combat {
    pub fn new(_: u64, map: &str, goal: &str, limit: u64) -> Self {
        let (walls, covers) = match map {
            "cover" => (vec![(4, 1), (4, 3)], vec![(3, 1), (3, 3), (5, 1), (5, 3)]),
            "choke" => (vec![(4, 0), (4, 1), (4, 3), (4, 4)], vec![(3, 2), (5, 2)]),
            _ => (vec![], vec![]),
        };
        let units = (0..2)
            .flat_map(|team| {
                [0, 2, 4].map(move |y| Unit {
                    team,
                    x: if team == 0 { 1 } else { 7 },
                    y,
                    hp: 9,
                    ammo: 3,
                    kit: 1,
                    guard: false,
                })
            })
            .collect();
        Self {
            units,
            walls,
            covers,
            goal: goal.into(),
            hold: [0; 2],
            secured: [false; 2],
            tick: 0,
            limit,
            routes: std::array::from_fn(|_| None),
        }
    }
    fn alive(&self, t: usize) -> bool {
        self.units.iter().any(|u| u.team == t && u.hp > 0)
    }
    fn winner(&self, t: usize) -> bool {
        let k = !self.alive(1 - t);
        let c = self.secured[t];
        self.alive(t)
            && match self.goal.as_str() {
                "eliminate" => k,
                "secure" => c,
                "both" => k && c,
                _ => k || c,
            }
    }
    fn los(&self, a: (i32, i32), b: (i32, i32)) -> bool {
        for &(x, y) in &self.walls {
            let (mut lo, mut hi) = (0f64, 1f64);
            for (s, e, c) in [(a.0, b.0, x), (a.1, b.1, y)] {
                let d = (e - s) as f64;
                if d == 0. {
                    if (s as f64) < c as f64 - 0.5 || (s as f64) > c as f64 + 0.5 {
                        lo = 1.;
                        hi = 0.;
                        break;
                    }
                } else {
                    let f = (c as f64 - 0.5 - s as f64) / d;
                    let l = (c as f64 + 0.5 - s as f64) / d;
                    lo = lo.max(f.min(l));
                    hi = hi.min(f.max(l));
                }
            }
            if lo <= hi {
                return false;
            }
        }
        true
    }
    // Static public terrain only: as in combat.py, moving occupancy and ammunition
    // do not alter the geometric path to a zone or firing position.
    fn terrain_distance(&self, start: (i32, i32), goal: impl Fn((i32, i32)) -> bool) -> u32 {
        let mut queue = std::collections::VecDeque::from([(start, 0u32)]);
        let mut seen = [[false; 5]; 9];
        seen[start.0 as usize][start.1 as usize] = true;
        while let Some(((x, y), distance)) = queue.pop_front() {
            if goal((x, y)) {
                return distance;
            }
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (nx, ny) = (x + dx, y + dy);
                if (0..9).contains(&nx)
                    && (0..5).contains(&ny)
                    && !self.walls.contains(&(nx, ny))
                    && !seen[nx as usize][ny as usize]
                {
                    seen[nx as usize][ny as usize] = true;
                    queue.push_back(((nx, ny), distance + 1));
                }
            }
        }
        45
    }
    fn zone_distance(&self, pos: (i32, i32)) -> u32 {
        self.terrain_distance(pos, |(x, y)| x == 4 && (1..=3).contains(&y))
    }
    fn firing_distance(&self, pos: (i32, i32), enemies: &[(i32, i32)]) -> u32 {
        self.terrain_distance(pos, |(x, y)| {
            enemies
                .iter()
                .any(|&(a, b)| (x - a).abs() + (y - b).abs() <= 4 && self.los((x, y), (a, b)))
        })
    }
    fn preparation(&self, team: usize) -> f64 {
        self.preparation_for_route(team, None)
    }
    fn preparation_for_route(&self, team: usize, route: Option<&str>) -> f64 {
        if self.winner(team) {
            return 1.;
        }
        if self.winner(1 - team) || !self.alive(team) {
            return -1.;
        }
        let own: Vec<_> = self
            .units
            .iter()
            .filter(|u| u.team == team && u.hp > 0)
            .collect();
        let eliminate = self.route_progress(team, "eliminate");
        let secure = self.route_progress(team, "secure");
        let mission = match self.goal.as_str() {
            "eliminate" => eliminate,
            "secure" => secure,
            "both" => eliminate.min(secure),
            _ => eliminate.max(secure),
        };
        let fitness = own.iter().map(|u| u.hp).sum::<i32>() as f64 / 27.;
        let ammunition = own.iter().map(|u| u.ammo).sum::<i32>() as f64 / 9.;
        let route_mission = match route {
            Some("eliminate") => 0.6 * eliminate + 0.2 * mission,
            Some("secure") => 0.6 * secure + 0.2 * mission,
            _ => 0.8 * mission,
        };
        0.6 * (route_mission + 0.15 * fitness + 0.05 * ammunition)
    }
    fn route_progress(&self, team: usize, route: &str) -> f64 {
        let own: Vec<_> = self
            .units
            .iter()
            .filter(|u| u.team == team && u.hp > 0)
            .collect();
        match route {
            "eliminate" => {
                let enemies: Vec<_> = self
                    .units
                    .iter()
                    .filter(|u| u.team != team && u.hp > 0)
                    .collect();
                if enemies.is_empty() {
                    return 1.;
                }
                let positions: Vec<_> = enemies.iter().map(|u| (u.x, u.y)).collect();
                let damage = 1. - enemies.iter().map(|u| u.hp).sum::<i32>() as f64 / 27.;
                let access = own
                    .iter()
                    .map(|u| {
                        (1. - self.firing_distance((u.x, u.y), &positions) as f64 / 4.).max(0.)
                    })
                    .sum::<f64>()
                    / 3.;
                0.65 * damage + 0.35 * access
            }
            "secure" => {
                if self.secured[team] {
                    return 1.;
                }
                let proximity = own
                    .iter()
                    .map(|u| 1. - (self.zone_distance((u.x, u.y)) as f64 / 8.).min(1.))
                    .fold(0f64, f64::max);
                0.5 * proximity + 0.5 * self.hold[team] as f64 / 3.
            }
            _ => unreachable!("route is validated"),
        }
    }
    fn route_allowed(&self, team: usize, route: &str) -> bool {
        matches!(route, "eliminate" | "secure")
            && (self.goal == "either" || self.goal == "both" || self.goal == route)
            && (self.goal != "both" || self.route_progress(team, route) < 1.)
    }
    fn chance(&self, i: usize, j: usize) -> f64 {
        let a = &self.units[i];
        let b = &self.units[j];
        let d = (a.x - b.x).abs() + (a.y - b.y).abs();
        if a.hp <= 0
            || b.hp <= 0
            || a.team == b.team
            || a.ammo <= 0
            || d > 4
            || !self.los((a.x, a.y), (b.x, b.y))
        {
            return 0.;
        }
        (0.9 - 0.08 * d as f64
            - if self.covers.contains(&(b.x, b.y)) {
                0.22
            } else {
                0.
            }
            - if b.guard { 0.2 } else { 0. })
        .max(0.12)
    }
}
impl ArenaGame for Combat {
    fn social_bindings(&self, player: usize) -> Result<BTreeMap<String, SocialBinding>> {
        if player >= self.players() || self.terminal() {
            return Err(err());
        }
        let mut bindings = BTreeMap::new();
        for action in self.legal(player) {
            if let Some((kind, target)) = action.split_once(':') {
                if kind == "heal" || kind == "shoot" {
                    let target: usize = target.parse().map_err(|_| err())?;
                    bindings.insert(
                        action.clone(),
                        SocialBinding {
                            target: target.to_string(),
                            support: if kind == "heal" { 1. } else { -1. },
                            distress: (1. - self.units[target].hp as f64 / 9.).clamp(0., 1.),
                        },
                    );
                }
            }
        }
        Ok(bindings)
    }
    fn social_events(
        &self,
        next: &Self,
        choices: &[String],
        seed: u64,
    ) -> Result<Vec<Vec<SocialEvent>>> {
        let (resolved, events) = self.resolved(choices, seed, true)?;
        if resolved != *next {
            return Err(err());
        }
        Ok(events)
    }

    fn current_route(&self, p: usize) -> Option<&str> {
        self.routes.get(p).and_then(Option::as_deref)
    }
    fn route_input(&self, p: usize) -> Result<Option<Observation>> {
        if p >= self.players() || self.terminal() {
            return Err(err());
        }
        if self.units[p].hp <= 0 {
            return Ok(None);
        }
        let team = self.units[p].team;
        let actions = ["eliminate", "secure"]
            .into_iter()
            .map(|route| {
                let quality = 0.5 + 0.5 * self.route_progress(team, route);
                let mut values = [0.; 10];
                values[3] = quality;
                if route == "eliminate" {
                    values[4] = 0.6;
                    values[5] = -0.2;
                } else {
                    values[6] = 0.5;
                    values[5] = 0.35;
                }
                let mut needs = [0.; 5];
                needs[3] = quality;
                Action {
                    id: route.into(),
                    target: None,
                    legal: self.route_allowed(team, route),
                    known_failure: false,
                    confidence: 1.,
                    familiarity: 0.,
                    switch_cost: 0.,
                    outcomes: vec![outcome(quality, needs, values, [0.; 5], 0.)],
                }
            })
            .collect();
        let deficits = self.deficits(p);
        Ok(Some(Observation {
            tick: self.tick,
            needs: std::array::from_fn(|i| NeedInput {
                supported: deficits[i].is_some(),
                enabled: deficits[i].is_some(),
                deficit: deficits[i],
            }),
            facts: self.facts(p),
            actions,
            opponent: None,
        }))
    }
    fn with_route(&self, p: usize, route: Option<&str>) -> Result<Self> {
        if p >= self.players() {
            return Err(err());
        }
        if let Some(route) = route {
            if self.terminal()
                || self.units[p].hp <= 0
                || !self.route_allowed(self.units[p].team, route)
            {
                return Err(err());
            }
        }
        let mut clone = self.clone();
        clone.routes[p] = route.map(str::to_owned);
        Ok(clone)
    }
    fn agreement_compatible(&self, known: &BTreeMap<usize, String>) -> Result<bool> {
        let mut destinations = std::collections::BTreeSet::new();
        for (&actor, action) in known {
            if actor >= self.players() {
                return Err(err());
            }
            if let Some(destination) = action.strip_prefix("move:") {
                let mut parts = destination.split(':');
                let x = parts
                    .next()
                    .and_then(|s| s.parse::<i32>().ok())
                    .ok_or_else(err)?;
                let y = parts
                    .next()
                    .and_then(|s| s.parse::<i32>().ok())
                    .ok_or_else(err)?;
                if parts.next().is_some() {
                    return Err(err());
                }
                if !destinations.insert((self.units[actor].team, x, y)) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn partners(&self, player: usize) -> Vec<usize> {
        if player >= self.players() {
            return vec![];
        }
        let team = self.units[player].team;
        (0..self.players())
            .filter(|&p| p != player && self.units[p].team == team)
            .collect()
    }
    fn offers(&self, announced: &[String]) -> Result<Vec<crate::coordination::Offer>> {
        if self.terminal() || announced.len() != self.players() {
            return Err(err());
        }
        fn destination(a: &str) -> Option<(i32, i32)> {
            let mut s = a.split(':');
            if s.next() != Some("move") {
                return None;
            }
            Some((s.next()?.parse().ok()?, s.next()?.parse().ok()?))
        }
        fn enumerate(
            game: &Combat,
            members: &[usize],
            index: usize,
            roots: &mut BTreeMap<usize, String>,
            fixed: &[(i32, i32)],
            out: &mut Vec<crate::coordination::Offer>,
            team: usize,
        ) {
            if out.len() >= 32 {
                return;
            }
            if index == members.len() {
                let id = format!("team-{team}-offer-{}", out.len());
                out.push(crate::coordination::Offer {
                    id,
                    roots: roots.clone(),
                });
                return;
            }
            let p = members[index];
            for action in game
                .legal(p)
                .into_iter()
                .filter(|a| a == "guard" || a.starts_with("move:"))
            {
                let dest = destination(&action);
                if dest.is_some_and(|d| {
                    fixed.contains(&d) || roots.values().any(|a| destination(a) == Some(d))
                }) {
                    continue;
                }
                roots.insert(p, action);
                enumerate(game, members, index + 1, roots, fixed, out, team);
                roots.remove(&p);
                if out.len() >= 32 {
                    break;
                }
            }
        }
        let mut all = Vec::new();
        for team in 0..2 {
            let team_players: Vec<usize> = (0..6).filter(|&p| self.units[p].team == team).collect();
            if team_players
                .iter()
                .any(|&p| !self.legal(p).contains(&announced[p]))
            {
                continue;
            }
            let members: Vec<usize> = team_players
                .iter()
                .copied()
                .filter(|&p| {
                    destination(&announced[p]).is_some_and(|d| {
                        team_players
                            .iter()
                            .any(|&q| q != p && destination(&announced[q]) == Some(d))
                    })
                })
                .collect();
            if members.len() < 2 {
                continue;
            }
            let fixed: Vec<_> = team_players
                .iter()
                .filter(|p| !members.contains(p))
                .filter_map(|&p| destination(&announced[p]))
                .collect();
            let mut offers = Vec::new();
            enumerate(
                self,
                &members,
                0,
                &mut BTreeMap::new(),
                &fixed,
                &mut offers,
                team,
            );
            all.extend(offers);
        }
        Ok(all)
    }
    fn players(&self) -> usize {
        6
    }
    fn tick(&self) -> u64 {
        self.tick
    }
    fn terminal(&self) -> bool {
        self.tick >= self.limit
            || self.winner(0)
            || self.winner(1)
            || (!self.alive(0) && !self.alive(1))
    }
    common!(Combat);
    fn model_world(&self, viewer: usize, _: u64) -> Result<Self> {
        if viewer >= self.players() {
            return Err(err());
        }
        let mut model = self.clone();
        for (p, route) in model.routes.iter_mut().enumerate() {
            if p != viewer {
                *route = None;
            }
        }
        Ok(model)
    }
    fn purpose(&self, p: usize) -> f64 {
        let t = self.units[p].team;
        if self.winner(t) {
            return 1.;
        }
        if self.winner(1 - t) || !self.alive(t) {
            return 0.;
        }
        let damage = 1.
            - self
                .units
                .iter()
                .filter(|u| u.team != t)
                .map(|u| u.hp)
                .sum::<i32>() as f64
                / 27.;
        let capture = if self.secured[t] {
            1.
        } else {
            self.hold[t] as f64 / 3.
        };
        let mission = match self.goal.as_str() {
            "eliminate" => damage,
            "secure" => capture,
            "both" => damage.min(capture),
            _ => damage.max(capture),
        };
        let fitness = self
            .units
            .iter()
            .filter(|u| u.team == t)
            .map(|u| u.hp)
            .sum::<i32>() as f64
            / 27.;
        (0.2 + 0.6 * mission + 0.2 * fitness).clamp(0., 1.)
    }
    fn view(&self) -> Value {
        json!(self)
    }
}
impl Combat {
    fn resolved(
        &self,
        c: &[String],
        seed: u64,
        collect: bool,
    ) -> Result<(Self, Vec<Vec<SocialEvent>>)> {
        if self.terminal() || c.len() != 6 || (0..6).any(|p| !self.legal(p).contains(&c[p])) {
            return Err(err());
        }
        let mut events = if collect {
            vec![vec![]; self.players()]
        } else {
            vec![]
        };
        let mut n = self.clone();
        for (p, choice) in c.iter().enumerate() {
            n.units[p].guard = choice == "guard";
        }
        let moves: Vec<Option<(i32, i32)>> = c
            .iter()
            .map(|a| {
                let mut s = a.split(':');
                if s.next() == Some("move") {
                    Some((s.next()?.parse().ok()?, s.next()?.parse().ok()?))
                } else {
                    None
                }
            })
            .collect();
        for p in 0..6 {
            if let Some(q) = moves[p] {
                if moves.iter().filter(|&&x| x == Some(q)).count() == 1 {
                    n.units[p].x = q.0;
                    n.units[p].y = q.1;
                }
            }
        }
        let mut healing = [0; 6];
        for (p, choice) in c.iter().enumerate() {
            if choice == "reload" {
                n.units[p].ammo = 3;
            } else if let Some(t) = choice.strip_prefix("heal:") {
                let j: usize = t.parse().map_err(|_| err())?;
                healing[j] += 4;
                n.units[p].kit -= 1;
            }
        }
        if collect {
            // Split only the health actually restored; capped duplicate healing earns no extra credit.
            for (source, choice) in c.iter().enumerate() {
                if let Some(t) = choice.strip_prefix("heal:") {
                    let target: usize = t.parse().map_err(|_| err())?;
                    if source != target {
                        let restored = healing[target].min(9 - n.units[target].hp);
                        events[target].push(SocialEvent {
                            target: source.to_string(),
                            event_id: format!("{}:aid:{target}:{source}", self.tick),
                            tick: self.tick,
                            agency: true,
                            confidence: 1.,
                            benefit: (restored as f64 * 4. / healing[target] as f64) / 9.,
                        });
                    }
                }
            }
        }
        for (p, h) in healing.iter().enumerate() {
            n.units[p].hp = (n.units[p].hp + h).min(9);
        }
        let mut losses = [0; 6];
        let mut hits = vec![];
        for (p, choice) in c.iter().enumerate() {
            if let Some(t) = choice.strip_prefix("shoot:") {
                let j: usize = t.parse().map_err(|_| err())?;
                if (rnd(seed, self.tick * 6 + p as u64) % 1_000_000) as f64 / 1_000_000.
                    < n.chance(p, j)
                {
                    losses[j] += 3;
                    if collect {
                        hits.push((p, j));
                    }
                }
                n.units[p].ammo -= 1;
            }
        }
        for (source, target) in hits {
            let lost = losses[target].min(n.units[target].hp);
            events[target].push(SocialEvent {
                target: source.to_string(),
                event_id: format!("{}:damage:{target}:{source}", self.tick),
                tick: self.tick,
                agency: true,
                confidence: 1.,
                benefit: -(lost as f64 * 3. / losses[target] as f64) / 9.,
            });
        }
        for (p, d) in losses.iter().enumerate() {
            n.units[p].hp = (n.units[p].hp - d).max(0);
        }
        let mut count = [0; 2];
        for u in &n.units {
            if u.hp > 0 && u.x == 4 && (1..=3).contains(&u.y) {
                count[u.team] += 1;
            }
        }
        for t in 0..2 {
            n.hold[t] = if count[t] > count[1 - t] && count[t] > 0 {
                (n.hold[t] + 1).min(3)
            } else {
                0
            };
            n.secured[t] |= n.hold[t] >= 3;
        }
        n.tick += 1;
        Ok((n, events))
    }
}
impl World for Combat {
    fn route_context(&self, p: usize) -> Option<&str> {
        self.routes[p].as_deref()
    }
    fn readiness(&self, p: usize) -> f64 {
        self.preparation(self.units[p].team)
    }
    fn behavior_alphabet(&self) -> Vec<String> {
        [
            "finished",
            "guard",
            "heal",
            "move:other",
            "move:toward-zone",
            "reload",
            "shoot",
        ]
        .map(str::to_owned)
        .to_vec()
    }
    fn model_choice(&self, p: usize, h: usize) -> String {
        let legal = self.legal(p);
        let u = &self.units[p];
        if h == 0 {
            return legal
                .iter()
                .find(|a| a.starts_with("heal:"))
                .or_else(|| legal.iter().find(|a| a.as_str() == "guard"))
                .unwrap_or(&legal[0])
                .clone();
        }
        if h == 2 {
            if let Some(a) = legal
                .iter()
                .filter(|a| a.starts_with("shoot:"))
                .max_by(|a, b| {
                    let x = a[6..].parse::<usize>().unwrap();
                    let y = b[6..].parse::<usize>().unwrap();
                    self.chance(p, x).total_cmp(&self.chance(p, y))
                })
            {
                return a.clone();
            }
            if legal.contains(&"reload".into()) {
                return "reload".into();
            }
        }
        let current_distance = self.zone_distance((u.x, u.y));
        let best = legal
            .iter()
            .filter_map(|a| {
                let mut parts = a.split(':');
                if parts.next() != Some("move") {
                    return None;
                }
                let x = parts.next()?.parse::<i32>().ok()?;
                let y = parts.next()?.parse::<i32>().ok()?;
                let distance = self.zone_distance((x, y));
                (distance < current_distance).then_some((distance, a))
            })
            .min_by_key(|(d, _)| *d);
        best.map(|(_, a)| a.clone()).unwrap_or_else(|| {
            if u.ammo < 3 && legal.contains(&"reload".into()) {
                "reload".into()
            } else {
                legal[0].clone()
            }
        })
    }
    fn facts(&self, p: usize) -> BTreeMap<String, String> {
        let u = &self.units[p];
        BTreeMap::from([
            ("tick".into(), self.tick.to_string()),
            ("goal".into(), self.goal.clone()),
            (
                "unit".into(),
                format!(
                    "team:{} pos:{},{} hp:{} ammo:{} kit:{}",
                    u.team, u.x, u.y, u.hp, u.ammo, u.kit
                ),
            ),
            ("hold".into(), format!("{:?}", self.hold)),
        ])
    }
    fn binding(&self, p: usize, action: &str) -> reflex_owned::experience::Binding {
        let u = &self.units[p];
        let mut parts = action.split(':');
        let kind = parts.next().unwrap_or(action);
        let enemy = self
            .units
            .iter()
            .enumerate()
            .filter(|(_, v)| v.team != u.team && v.hp > 0)
            .map(|(i, v)| {
                (
                    (u.x - v.x).abs() + (u.y - v.y).abs(),
                    self.chance(p, i) > 0.,
                )
            })
            .min_by_key(|v| v.0);
        let method = if kind == "move" {
            let x = parts
                .next()
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(u.x);
            let y = parts
                .next()
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(u.y);
            let toward = (4 - x).abs() + (2 - y).abs() < (4 - u.x).abs() + (2 - u.y).abs();
            format!("move:{}", if toward { "toward-zone" } else { "other" })
        } else {
            kind.into()
        };
        let situation = format!(
            "goal:{}:hp:{}:ammo:{}:enemy:{}:cover:{}",
            self.goal,
            bucket(u.hp as f64, 4., 7.),
            bucket(u.ammo as f64, 1., 3.),
            match enemy {
                None => "none",
                Some((_, true)) => "visible",
                Some((d, false)) if d <= 4 => "blocked",
                _ => "distant",
            },
            self.covers.contains(&(u.x, u.y))
        );
        binding(method, situation)
    }
    fn legal(&self, p: usize) -> Vec<String> {
        let u = &self.units[p];
        if u.hp <= 0 {
            return vec!["finished".into()];
        }
        let mut l = vec!["guard".into()];
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let q = (u.x + dx, u.y + dy);
            if (0..9).contains(&q.0)
                && (0..5).contains(&q.1)
                && !self.walls.contains(&q)
                && !self.units.iter().any(|v| v.hp > 0 && (v.x, v.y) == q)
            {
                l.push(format!("move:{}:{}", q.0, q.1));
            }
        }
        if u.ammo < 3 {
            l.push("reload".into())
        }
        for (j, v) in self.units.iter().enumerate() {
            if self.chance(p, j) > 0. {
                l.push(format!("shoot:{j}"));
            }
            if u.kit > 0
                && v.team == u.team
                && v.hp > 0
                && v.hp < 9
                && (u.x - v.x).abs() + (u.y - v.y).abs() <= 1
            {
                l.push(format!("heal:{j}"));
            }
        }
        l.sort();
        l
    }
    fn deficits(&self, p: usize) -> [Option<f64>; 5] {
        let u = &self.units[p];
        [
            Some(1. - u.hp as f64 / 9.),
            Some((1. - u.hp as f64 / 9. + 0.15).min(1.)),
            Some(
                1. - self
                    .units
                    .iter()
                    .filter(|v| v.team == u.team)
                    .map(|v| v.hp)
                    .sum::<i32>() as f64
                    / 27.,
            ),
            Some(0.35),
            Some(0.25),
        ]
    }
    fn transition(&self, c: &[String], seed: u64) -> Result<Self> {
        self.resolved(c, seed, false).map(|(world, _)| world)
    }
    fn effect(&self, n: &Self, p: usize, a: &str) -> Outcome {
        let u = &self.units[p];
        let v = &n.units[p];
        let dh = (v.hp - u.hp) as f64 / 9.;
        let allies = (n
            .units
            .iter()
            .filter(|x| x.team == u.team)
            .map(|x| x.hp)
            .sum::<i32>()
            - self
                .units
                .iter()
                .filter(|x| x.team == u.team)
                .map(|x| x.hp)
                .sum::<i32>()) as f64
            / 27.;
        let route = self.routes[p].as_deref();
        let obj = clamp(
            n.preparation_for_route(u.team, route) - self.preparation_for_route(u.team, route),
        );
        let mut values = [0.; 10];
        values[3] = obj;
        values[5] = dh;
        values[8] = allies;
        if a.starts_with("shoot:") {
            values[4] = obj
        }
        let mut style = [0.; 5];
        style[1] = 0.15 * ((v.ammo - u.ammo) as f64 / 3.).max(0.);
        outcome(obj, [dh, dh, allies, obj, 0.], values, style, 0.)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encounter_events_separate_choice_from_shared_bad_luck() {
        let game = Encounters::new(1, 8);
        let choices = vec![
            "cooperate:1".into(),
            "cooperate:0".into(),
            "rest".into(),
            "rest".into(),
        ];
        let accident = (0..1000).find(|&s| rnd(s, 1) % 100 < 5).unwrap();
        let next = game.transition(&choices, accident).unwrap();
        let events = game.social_events(&next, &choices, accident).unwrap();
        assert!(!events[0][0].agency && !events[1][0].agency);
        assert_eq!(events[0][0].benefit, 0.);
        assert!(events[2].is_empty());
        let mut exploit = choices.clone();
        exploit[1] = "claim:0".into();
        let next = game.transition(&exploit, accident).unwrap();
        let events = game.social_events(&next, &exploit, accident).unwrap();
        assert_eq!(events[0][0].target, "1");
        assert_eq!(events[0][0].benefit, -1.);
        assert!(events[0][0].agency);
        assert_eq!(events[1][0].benefit, 0.8);
        assert!(game.social_events(&game, &exploit, accident).is_err());
        let input = game.input(0).unwrap();
        assert_eq!(
            input
                .observation
                .actions
                .iter()
                .find(|a| a.id == "cooperate:1")
                .unwrap()
                .target
                .as_deref(),
            Some("1")
        );
    }
    #[test]
    fn combat_events_credit_only_delivered_help_and_damage() {
        let mut game = Combat::new(1, "open", "eliminate", 10);
        for (p, xy) in [(0, (3, 1)), (1, (3, 2)), (2, (2, 1)), (3, (4, 1))] {
            game.units[p].x = xy.0;
            game.units[p].y = xy.1;
        }
        game.units[0].hp = 8;
        let choices: Vec<String> = ["guard", "heal:0", "heal:0", "shoot:0", "guard", "guard"]
            .map(str::to_owned)
            .to_vec();
        let seed = (0..1000)
            .find(|&s| {
                let next = game.transition(&choices, s).unwrap();
                next.units[0].hp == 6
            })
            .unwrap();
        let next = game.transition(&choices, seed).unwrap();
        let events = game.social_events(&next, &choices, seed).unwrap();
        assert_eq!(events[0].len(), 3);
        let total: f64 = events[0].iter().map(|e| e.benefit).sum();
        assert!((total - (next.units[0].hp - game.units[0].hp) as f64 / 9.).abs() < 1e-12);
        for target in ["1", "2"] {
            assert!(
                (events[0]
                    .iter()
                    .find(|e| e.target == target)
                    .unwrap()
                    .benefit
                    - 0.5 / 9.)
                    .abs()
                    < 1e-12
            );
        }
        assert!(events[1].is_empty());
        assert!(game.social_events(&game, &choices, seed).is_err());
    }
    #[test]
    fn fair_competition_does_not_create_betrayal_memory() {
        let game = Hagetaka::new(1);
        let choices = (0..4).map(|p| game.legal(p)[0].clone()).collect::<Vec<_>>();
        let next = game.transition(&choices, 1).unwrap();
        assert!(game.social_bindings(0).unwrap().is_empty());
        assert!(game
            .social_events(&next, &choices, 1)
            .unwrap()
            .iter()
            .all(Vec::is_empty));
    }
    fn verify_public_behavior<G: ArenaGame>(game: &G) {
        let first = game.behavior(0, 1).unwrap();
        assert!(!first.context.is_empty());
        assert_eq!(first.suggestions.len(), 3);
        assert!(first
            .actions
            .values()
            .all(|class| first.alphabet.contains(class)));
        assert!(first
            .suggestions
            .iter()
            .all(|action| first.actions.contains_key(action)));
        assert!(game.behavior(game.players(), 1).is_err());
        assert!(game.behavior(0, game.players()).is_err());
        let scenarios = (0..3)
            .map(|h| game.modeled_choices(0, &h.to_string()).unwrap())
            .collect::<Vec<_>>();
        let baseline = game.input(0).unwrap();
        let derived = game.input_scenarios(0, &scenarios).unwrap();
        assert_eq!(baseline.conditional, derived.conditional);
        assert!(game.input_scenarios(0, &scenarios[..2]).is_err());
        let mut invalid = scenarios.clone();
        invalid[0][1] = "not-legal".into();
        assert!(game.input_scenarios(0, &invalid).is_err());
        let mut ignored_own = scenarios;
        ignored_own[0][0] = "ignored".into();
        assert!(game.input_scenarios(0, &ignored_own).is_ok());
    }
    #[test]
    fn agreement_uses_only_known_teammates_without_banning_independent_collisions() {
        let mut game = Combat::new(1, "open", "secure", 10);
        game.units[0].x = 3;
        game.units[0].y = 1;
        game.units[1].x = 5;
        game.units[1].y = 1;
        let mut choices = vec!["guard".to_string(); 6];
        let offer = crate::coordination::Offer {
            id: "conflict".into(),
            roots: BTreeMap::from([(0, "move:4:1".into()), (1, "move:4:1".into())]),
        };
        let known = crate::known_offer_intentions(&game, &offer, &choices).unwrap();
        assert_eq!(known.keys().copied().collect::<Vec<_>>(), vec![0, 1, 2]);
        assert!(!crate::offer_compatible(&game, &offer, &choices).unwrap());
        choices[3] = "unrevealed-enemy-action".into();
        assert_eq!(
            known,
            crate::known_offer_intentions(&game, &offer, &choices).unwrap()
        );
        assert!(!crate::offer_compatible(&game, &offer, &choices).unwrap());
        let safe = crate::coordination::Offer {
            id: "safe".into(),
            roots: BTreeMap::from([(0, "move:4:1".into()), (1, "guard".into())]),
        };
        assert!(crate::offer_compatible(&game, &safe, &choices).unwrap());
        let invalid = crate::coordination::Offer {
            id: "wrong-team".into(),
            roots: BTreeMap::from([(0, "guard".into()), (3, "guard".into())]),
        };
        assert!(crate::offer_compatible(&game, &invalid, &choices).is_err());
        choices[3] = "guard".into();
        choices[0] = "move:4:1".into();
        choices[1] = "move:4:1".into();
        let (next, _) = game.advance(&choices, 1).unwrap();
        assert_eq!((next.units[0].x, next.units[1].x), (3, 5));
    }
    #[test]
    fn voluntary_offers_are_legal_scoped_and_ignore_opponents() {
        let hagetaka = Hagetaka::new(1);
        assert!(hagetaka.partners(0).is_empty());
        assert!(hagetaka
            .offers(&hagetaka.modeled_choices(0, "0").unwrap())
            .unwrap()
            .is_empty());
        let encounters = Encounters::new(1, 5);
        assert_eq!(encounters.partners(0), vec![1, 2, 3]);
        let announced = vec!["alone".to_string(); 4];
        let offers = encounters.offers(&announced).unwrap();
        assert_eq!(offers.len(), 6);
        for offer in &offers {
            assert_eq!(offer.roots.len(), 2);
            for (&actor, action) in &offer.roots {
                assert!(encounters.legal(actor).contains(action));
                assert!(action.starts_with("cooperate:"));
            }
        }
        let mut combat = Combat::new(1, "open", "secure", 10);
        combat.units[0].x = 3;
        combat.units[0].y = 1;
        combat.units[1].x = 5;
        combat.units[1].y = 1;
        assert_eq!(combat.partners(0), vec![1, 2]);
        let mut choices = vec!["guard".to_string(); 6];
        assert!(combat.offers(&choices).unwrap().is_empty());
        choices[0] = "move:4:1".into();
        choices[1] = "move:4:1".into();
        let offers = combat.offers(&choices).unwrap();
        assert!(!offers.is_empty());
        assert!(offers.len() <= 32);
        for offer in &offers {
            assert_eq!(offer.roots.keys().copied().collect::<Vec<_>>(), vec![0, 1]);
            for (&p, a) in &offer.roots {
                assert!(combat.legal(p).contains(a));
            }
            let mut c = choices.clone();
            for (&p, a) in &offer.roots {
                c[p] = a.clone();
            }
            let moved: Vec<_> = c[..3].iter().filter(|a| a.starts_with("move:")).collect();
            assert!(moved
                .iter()
                .enumerate()
                .all(|(i, a)| !moved[..i].contains(a)));
        }
        let other_move = combat
            .legal(3)
            .into_iter()
            .find(|a| a.starts_with("move:"))
            .unwrap();
        choices[3] = other_move;
        assert_eq!(combat.offers(&choices).unwrap(), offers);
        let (_, feedback) = combat
            .advance(
                &[
                    "move:4:1".into(),
                    "move:4:1".into(),
                    "guard".into(),
                    "guard".into(),
                    "guard".into(),
                    "guard".into(),
                ],
                1,
            )
            .unwrap();
        assert_eq!(feedback[0].observed.unwrap()[16], 0.);
    }
    #[test]
    fn individual_behavior_and_scenarios_cover_all_games() {
        verify_public_behavior(&Hagetaka::new(11));
        verify_public_behavior(&Encounters::new(11, 10));
        verify_public_behavior(&Combat::new(11, "cover", "either", 10));
    }
    #[test]
    fn responses_require_one_complete_match() {
        let strings = |xs: &[&str]| xs.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let actual = strings(&["own", "a", "b"]);
        let first = strings(&["different own", "a", "b"]);
        let second = strings(&["own", "a", "c"]);
        let third = strings(&["own", "d", "b"]);
        assert_eq!(
            identified_response(0, &actual, &[first.clone(), second.clone(), third.clone()]),
            Some("0".into())
        );
        assert_eq!(
            identified_response(0, &actual, &[first.clone(), first, third.clone()]),
            None
        );
        assert_eq!(
            identified_response(0, &actual, &[second.clone(), third, second]),
            None
        );
    }
    #[test]
    fn missing_response_still_learns_observed_effects() {
        use reflex_immediate::{Config, Policy, Scope};
        use reflex_owned::{prediction::Definition, Owner, OwnerConfig};
        let config = OwnerConfig {
            actor: Config {
                scope: Scope {
                    game: "test".into(),
                    episode: "1".into(),
                    npc: "0".into(),
                },
                seed: 1,
                personality: [0.5; 5],
                values: [0.5; 10],
                objective: "gain resources".into(),
                policy: Policy::default(),
            },
            predictor: Definition {
                model_id: "test".into(),
                factors: vec![],
                names: vec!["0".into(), "1".into(), "2".into()],
                window: 8,
            },
            contract: "test".into(),
            purpose_target: "score".into(),
        };
        let world = Encounters::new(1, 3);
        let mut owner = Owner::new(config).unwrap();
        let stage = owner
            .prepare_estimated(&world.input(0).unwrap(), false)
            .unwrap();
        let action = stage.decision().action_id.clone();
        let choices = vec![action.clone(), "rest".into(), "rest".into(), "rest".into()];
        let (_, feedback) = world.advance(&choices, 1).unwrap();
        assert_eq!(feedback[0].response, None);
        let ticket = owner.commit(stage, &action).unwrap();
        owner.resolve(ticket, &feedback[0]).unwrap();
        let state = owner.state().unwrap();
        assert_eq!(state.prediction.observations, 0);
        assert_eq!(state.experience.entries.len(), 1);
        assert_eq!(state.experience.entries[0].count, 1);
        assert!(state.progress.peak.is_some());
    }
    #[test]
    fn public_models_cover_cooperation_claim_and_independence() {
        let mut w = Encounters::new(7, 10);
        for (h, expected) in [("0", "cooperate:"), ("1", "claim:"), ("2", "alone")] {
            let choices = w.modeled_choices(0, h).unwrap();
            assert!(choices.iter().all(|c| c.starts_with(expected)));
            assert!(choices
                .iter()
                .enumerate()
                .all(|(p, c)| w.legal(p).contains(c)));
        }
        w.history[1].push("claim:2".into());
        assert_eq!(w.modeled_choices(0, "0").unwrap()[1], "cooperate:2");
        assert_eq!(w.modeled_choices(3, "0").unwrap()[1], "cooperate:2");
        let combat = Combat::new(1, "cover", "either", 10);
        assert_eq!(combat.modeled_choices(0, "0").unwrap()[0], "guard");
        assert!(combat.modeled_choices(0, "1").unwrap()[0].starts_with("move:"));
        let contest = Hagetaka::new(7);
        assert!((0..3).any(|h| {
            let c = contest.modeled_choices(0, &h.to_string()).unwrap();
            c.iter().any(|x| x != &c[0])
        }));
    }
    #[test]
    fn combat_approach_models_hold_every_zone_cell_and_keep_existing_fallbacks() {
        let mut game = Combat::new(1, "open", "secure", 10);
        // Block enemy sight without changing the zone or movement around it.
        game.walls = (0..5).map(|y| (6, y)).collect();
        for (i, y) in [0, 2, 4].into_iter().enumerate() {
            game.units[i + 3].x = 8;
            game.units[i + 3].y = y;
        }
        for y in 1..=3 {
            game.units[0].x = 4;
            game.units[0].y = y;
            for h in [1, 2] {
                let action = game.model_choice(0, h);
                assert_eq!(action, "guard");
                assert!(game.legal(0).contains(&action));
            }
        }
        let mut visible = game.clone();
        visible.walls.clear();
        visible.units[0].y = 2;
        visible.units[3].x = 5;
        visible.units[3].y = 2;
        assert_eq!(visible.model_choice(0, 1), "guard");
        let shot = visible.model_choice(0, 2);
        assert!(shot.starts_with("shoot:"));
        assert!(visible.legal(0).contains(&shot));
        game.units[0].ammo = 0;
        for h in [1, 2] {
            let action = game.model_choice(0, h);
            assert_eq!(action, "reload");
            assert!(game.legal(0).contains(&action));
        }
        game.units[0].hp = 0;
        for h in 0..3 {
            let action = game.model_choice(0, h);
            assert_eq!(action, "finished");
            assert!(game.legal(0).contains(&action));
        }
    }
    #[test]
    fn combat_approach_models_follow_shorter_terrain_path_around_wall() {
        let mut game = Combat::new(1, "cover", "secure", 10);
        game.units[0].x = 3;
        game.units[0].y = 1;
        assert!(game.walls.contains(&(4, 1)));
        let current = game.zone_distance((3, 1));
        for h in [1, 2] {
            let action = game.model_choice(0, h);
            assert_eq!(action, "move:3:2");
            assert!(game.legal(0).contains(&action));
            assert!(game.zone_distance((3, 2)) < current);
        }
        game.units[1].x = 3;
        game.units[1].y = 2;
        for h in [1, 2] {
            let action = game.model_choice(0, h);
            assert_eq!(action, "guard");
            assert!(game.legal(0).contains(&action));
        }
    }
    #[test]
    fn model_deck_ignores_hidden_order() {
        let a = Hagetaka::new(11);
        let mut b = a.clone();
        b.deck.reverse();
        let mut a = a.model_world(0, 42).unwrap();
        let mut b = b.model_world(0, 42).unwrap();
        for _ in 0..15 {
            assert_eq!(a.view(), b.view());
            let ca = a.modeled_choices(0, "1").unwrap();
            let cb = b.modeled_choices(0, "1").unwrap();
            assert_eq!(ca, cb);
            a = a.advance(&ca, 9).unwrap().0;
            b = b.advance(&cb, 9).unwrap().0;
        }
    }
    #[test]
    fn hagetaka_ties_carry_and_negative_low_unique() {
        let mut g = Hagetaka::new(7);
        g.prize = 5;
        g.carry = 0;
        let (g, _) = g
            .advance(&["01".into(), "01".into(), "02".into(), "02".into()], 1)
            .unwrap();
        assert_eq!(g.carry, 5);
        assert_eq!(g.scores, [0, 0, 0, 0]);
        let mut g = g;
        g.prize = -7;
        let (g, _) = g
            .advance(&["03".into(), "04".into(), "05".into(), "05".into()], 2)
            .unwrap();
        assert_eq!(g.scores[0], -2);
        assert_eq!(g.carry, 0);
    }
    #[test]
    fn encounters_match_and_public_history() {
        let g = Encounters::new(1, 3);
        let c = [
            "cooperate:1".into(),
            "claim:0".into(),
            "alone".into(),
            "rest".into(),
        ];
        let (n, f) = g.advance(&c, 5).unwrap();
        assert!((n.people[0].score + 0.15).abs() < 1e-9);
        assert!((n.people[1].score - 0.8).abs() < 1e-9);
        assert_eq!(n.history[0][0], c[0]);
        assert_eq!(f.len(), 4);
    }
    #[test]
    fn combat_route_candidates_follow_progress_and_constraints() {
        let mut game = Combat::new(1, "open", "both", 20);
        let frame = game.route_input(0).unwrap().unwrap();
        assert!(frame.actions.iter().all(|a| a.legal));
        for a in &frame.actions {
            let progress = game.route_progress(0, &a.id);
            let quality = 0.5 + 0.5 * progress;
            assert!((a.outcomes[0].objective - quality).abs() < 1e-12);
            assert_eq!(a.outcomes[0].needs[3], quality);
            assert_eq!(a.outcomes[0].values[3], quality);
        }
        game.secured[0] = true;
        let frame = game.route_input(0).unwrap().unwrap();
        assert!(
            !frame
                .actions
                .iter()
                .find(|a| a.id == "secure")
                .unwrap()
                .legal
        );
        assert!(
            frame
                .actions
                .iter()
                .find(|a| a.id == "eliminate")
                .unwrap()
                .legal
        );
        assert!(game.with_route(0, Some("secure")).is_err());
        assert!(game.with_route(0, Some("unknown")).is_err());
    }
    #[test]
    fn combat_route_is_private_and_cannot_create_phantom_progress() {
        let game = Combat::new(1, "cover", "either", 20);
        let routed = game
            .with_route(0, Some("eliminate"))
            .unwrap()
            .with_route(1, Some("secure"))
            .unwrap();
        assert_eq!(game.preparation(0), game.preparation_for_route(0, None));
        assert_eq!(game.view(), routed.view());
        let model = routed.model_world(0, 77).unwrap();
        assert_eq!(model.routes[0].as_deref(), Some("eliminate"));
        assert_eq!(model.routes[1], None);
        assert_eq!(
            routed
                .effect(&routed.with_route(0, Some("secure")).unwrap(), 0, "guard")
                .objective,
            0.
        );
        let own = routed.input(0).unwrap();
        let other = routed.input(1).unwrap();
        assert!(own
            .bindings
            .values()
            .all(|b| b.situation.ends_with(":route:eliminate")));
        assert!(own
            .purpose
            .activities
            .iter()
            .all(|(id, a)| a.condition == own.bindings[id].situation));
        assert!(other
            .bindings
            .values()
            .all(|b| b.situation.ends_with(":route:secure")));
        assert_eq!(
            game.behavior(0, 1).unwrap().context,
            routed.behavior(0, 1).unwrap().context
        );
        assert!(routed.with_route(0, None).unwrap().routes[0].is_none());
    }
    #[test]
    fn combat_preparation_uses_terrain_and_cancels_round_trips() {
        let choke = Combat::new(1, "choke", "eliminate", 10);
        let open = Combat::new(1, "open", "eliminate", 10);
        assert!(choke.firing_distance((3, 0), &[(5, 0)]) > open.firing_distance((3, 0), &[(5, 0)]));
        assert!(choke.zone_distance((3, 0)) > open.zone_distance((3, 0)));
        let mut toward = choke.clone();
        toward.units[0].x = 2;
        let mut back = toward.clone();
        back.units[0].x = 1;
        let forward = choke.effect(&toward, 0, "move:2:0").objective;
        let reverse = toward.effect(&back, 0, "move:1:0").objective;
        assert!(forward > 0.);
        assert!((forward + reverse).abs() < 1e-12);
        assert_eq!(choke.purpose(0), toward.purpose(0));
        let mut secure = Combat::new(1, "open", "secure", 10);
        let before = secure.preparation(0);
        secure.units[0].x = 3;
        secure.units[0].y = 1;
        assert!(secure.preparation(0) > before);
        let before_hold = secure.preparation(0);
        secure.hold[0] = 1;
        assert!(secure.preparation(0) > before_hold);
    }
    #[test]
    fn actual_position_progress_is_the_same_readiness_in_input_and_feedback() {
        let game = Combat::new(1, "choke", "eliminate", 10);
        let before = game.input(0).unwrap();
        let mut choices = vec!["guard".to_string(); game.players()];
        choices[0] = "move:2:0".into();
        let (next, feedback) = game.advance(&choices, 7).unwrap();
        let after = next.input(0).unwrap();
        assert_eq!(before.purpose.level, after.purpose.level);
        assert!(after.purpose.readiness > before.purpose.readiness);
        assert_eq!(before.purpose.readiness, game.preparation(0));
        assert_eq!(
            feedback[0].purpose.as_ref().unwrap().readiness,
            after.purpose.readiness
        );
        assert_eq!(after.purpose.readiness, next.preparation(0));
    }
    #[test]
    fn failures_are_scoped_to_the_public_situation_used_by_experience() {
        let game = Combat::new(1, "open", "eliminate", 10);
        let before = game.input(0).unwrap();
        for action in &before.observation.actions {
            assert_eq!(
                before.purpose.activities[&action.id].condition,
                before.bindings[&action.id].situation
            );
        }
        let mut progress = progress::Progress::default();
        let feedback = progress::PurposeFeedback {
            level: before.purpose.level,
            readiness: before.purpose.readiness,
            maintained: false,
            completed: false,
        };
        for _ in 0..3 {
            progress
                .observe("guard", &before.purpose, Some(&feedback))
                .unwrap();
        }
        assert!(!progress
            .allowed(&before.observation, &before.purpose)
            .unwrap()
            .contains("guard"));
        let mut changed = game.clone();
        changed.units[0].ammo = 1;
        let later = changed.input(0).unwrap();
        assert_ne!(
            before.bindings["guard"].situation,
            later.bindings["guard"].situation
        );
        assert!(progress
            .allowed(&later.observation, &later.purpose)
            .unwrap()
            .contains("guard"));
    }
    #[test]
    fn noncombat_readiness_uses_purpose_before_and_after_the_transition() {
        fn check<G: World>(game: G) {
            let input = game.input(0).unwrap();
            assert_eq!(input.purpose.readiness, game.purpose(0));
            let choices = game.modeled_choices(0, "0").unwrap();
            let (next, feedback) = game.advance(&choices, 7).unwrap();
            assert_eq!(
                feedback[0].purpose.as_ref().unwrap().readiness,
                next.purpose(0)
            );
        }
        check(Hagetaka::new(1));
        check(Encounters::new(1, 10));
    }
    #[test]
    fn combat_style_tracks_only_ammunition_recovery() {
        let mut game = Combat::new(1, "open", "either", 10);
        let mut choices = vec!["guard".to_string(); 6];
        let (next, feedback) = game.advance(&choices, 1).unwrap();
        assert_eq!(game.effect(&next, 0, "guard").style, [0.; 5]);
        assert_eq!(&feedback[0].observed.unwrap()[16..21], &[0.; 5]);
        for (before, expected) in [(0, 0.15), (1, 0.1), (2, 0.05)] {
            game.units[0].ammo = before;
            choices[0] = "reload".into();
            let (next, _) = game.advance(&choices, 1).unwrap();
            let style = game.effect(&next, 0, "reload").style;
            assert!((style[1] - expected).abs() < 1e-12);
            assert_eq!([style[0], style[2], style[3], style[4]], [0.; 4]);
        }
        let mut spent = game.clone();
        spent.units[0].ammo = 0;
        assert_eq!(game.effect(&spent, 0, "shoot:3").style, [0.; 5]);
    }

    #[test]
    fn combat_healing_preserves_actual_health_effects_without_style_bonus() {
        let mut game = Combat::new(1, "open", "either", 10);
        game.units[0].hp = 5;
        let mut choices = vec!["guard".to_string(); 6];
        choices[0] = "heal:0".into();
        let (next, _) = game.advance(&choices, 1).unwrap();
        let effect = game.effect(&next, 0, "heal:0");
        assert_eq!(effect.style, [0.; 5]);
        assert!((effect.needs[0] - 4. / 9.).abs() < 1e-12);
        assert!((effect.values[5] - 4. / 9.).abs() < 1e-12);
        assert!((effect.values[8] - 4. / 27.).abs() < 1e-12);
        game.units[1].x = 2;
        game.units[1].y = 0;
        game.units[1].hp = 5;
        choices[0] = "heal:1".into();
        let (next, _) = game.advance(&choices, 1).unwrap();
        let effect = game.effect(&next, 0, "heal:1");
        assert_eq!(effect.style, [0.; 5]);
        assert!((effect.values[8] - 4. / 27.).abs() < 1e-12);
    }

    #[test]
    fn combat_move_without_physical_progress_has_no_openness_reward() {
        let battle = Combat::new(1, "choke", "eliminate", 10);
        let mut choices = vec!["guard".to_string(); 6];
        choices[0] = "move:0:0".into();
        let (next, feedback) = battle.advance(&choices, 1).unwrap();
        assert_eq!((next.units[0].x, next.units[0].y), (0, 0));
        assert_eq!(battle.purpose(0), next.purpose(0));
        assert_eq!(feedback[0].observed.unwrap()[16], 0.);
    }
    #[test]
    fn combat_simultaneous_collision_and_capture() {
        let mut g = Combat::new(1, "open", "secure", 10);
        g.units[0].x = 3;
        g.units[0].y = 1;
        g.units[1].x = 5;
        g.units[1].y = 1;
        let mut c = vec!["guard".to_string(); 6];
        c[0] = "move:4:1".into();
        c[1] = "move:4:1".into();
        let (n, _) = g.advance(&c, 1).unwrap();
        assert_eq!(n.units[0].x, 3);
        assert_eq!(n.units[1].x, 5);
        c[1] = "guard".into();
        let (mut n, _) = g.advance(&c, 1).unwrap();
        assert_eq!(n.hold[0], 1);
        for _ in 0..2 {
            let c = vec!["guard".to_string(); 6];
            n = n.advance(&c, 1).unwrap().0;
        }
        assert!(n.winner(0));
    }
    #[test]
    fn combat_both_keeps_captured_purpose_after_leaving_but_tracks_unsecured_hold() {
        let mut game = Combat::new(1, "open", "both", 10);
        game.units[3].hp = 3;
        game.units[4].hp = 3;
        game.secured[0] = true;
        game.hold[0] = 3;
        let held = game.purpose(0);
        game.hold[0] = 0;
        assert_eq!(game.purpose(0), held);
        assert!(!game.winner(0));

        game.secured[0] = false;
        game.hold[0] = 0;
        let no_hold = game.purpose(0);
        game.hold[0] = 1;
        let partial = game.purpose(0);
        game.hold[0] = 2;
        assert!(no_hold < partial && partial < game.purpose(0));

        for unit in &mut game.units[3..] {
            unit.hp = 0;
        }
        assert!(!game.winner(0), "both still requires permanent capture");
        game.secured[0] = true;
        assert!(game.winner(0));
        assert_eq!(game.purpose(0), 1.);
        assert_eq!(game.purpose(3), 0.);
        let mut opposite = Combat::new(1, "open", "both", 10);
        for unit in &mut opposite.units[..3] {
            unit.hp = 0;
        }
        assert!(!opposite.winner(1));
        opposite.secured[1] = true;
        assert!(opposite.winner(1));
        assert_eq!(opposite.purpose(3), 1.);
        assert_eq!(opposite.purpose(0), 0.);
    }
    #[test]
    fn encounter_conserving_model_rests_before_work_becomes_illegal() {
        let mut g = Encounters::new(1, 32);
        for energy in [0.18, 0.185, 0.39, 0.499] {
            g.people[3].energy = energy;
            assert!(g.legal(3).contains(&"alone".to_string()));
            assert_eq!(g.model_choice(3, 2), "rest");
            assert_eq!(g.behavior(0, 3).unwrap().suggestions[2], "rest");
        }
        g.people[3].energy = 0.5;
        assert_eq!(g.model_choice(3, 2), "alone");
        g.people[3].health = 0.;
        assert_eq!(g.model_choice(3, 2), "finished");
    }
    #[test]
    fn encounter_safety_tracks_state_change_not_action_name() {
        for (stock, health, next_stock, next_health) in
            [(0.3, 4., 2., 6.), (2., 6., 0.3, 4.), (3., 8., 3., 8.)]
        {
            let mut g = Encounters::new(1, 32);
            g.people[0].stock = stock;
            g.people[0].health = health;
            let mut n = g.clone();
            n.people[0].stock = next_stock;
            n.people[0].health = next_health;
            n.history[1].push("cooperate:0".into());
            let expected = g.deficits(0)[1].unwrap() - n.deficits(0)[1].unwrap();
            for action in ["alone", "rest", "cooperate:1", "claim:1"] {
                let effect = g.effect(&n, 0, action);
                assert_eq!(effect.needs[1], expected);
                assert_eq!(effect.values[5], expected);
            }
        }
    }
    #[test]
    fn inputs_cover_actions_and_use_three_public_models() {
        let gs: Vec<Box<dyn Fn() -> Input>> = vec![
            Box::new(|| Hagetaka::new(1).input(0).unwrap()),
            Box::new(|| Encounters::new(1, 4).input(0).unwrap()),
            Box::new(|| Combat::new(1, "cover", "either", 4).input(0).unwrap()),
        ];
        for make in gs {
            let i = make();
            assert_eq!(i.responses.models.len(), 3);
            assert_eq!(i.conditional.len(), i.observation.actions.len());
            assert_eq!(i.purpose.activities.len(), i.observation.actions.len());
        }
    }
}
