//! Experimental multi-actor arena and bounded planner; not the frozen H32 owner.
use reflex_owned::{Feedback, Input, Result};
use std::collections::{BTreeMap, BTreeSet};
pub mod behavior;
pub mod coordination;
pub mod games;
pub mod planner;
pub mod robustness;
pub mod routes;

/// Current, non-binding intentions disclosed by all permitted partners.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Intentions {
    pub tick: u64,
    pub choices: BTreeMap<usize, String>,
}

/// All model callbacks read public state only. External effects are forbidden.
pub trait ArenaGame: Clone {
    fn players(&self) -> usize;
    fn tick(&self) -> u64;
    fn terminal(&self) -> bool;
    fn input(&self, player: usize) -> Result<Input>;
    fn behavior(&self, viewer: usize, actor: usize) -> Result<behavior::PublicBehavior>;
    fn input_scenarios(&self, player: usize, choices: &[Vec<String>]) -> Result<Input>;
    /// Build a private hypothetical world from public knowledge only.
    fn model_world(&self, viewer: usize, seed: u64) -> Result<Self>;
    fn advance(&self, choices: &[String], seed: u64) -> Result<(Self, Vec<Feedback>)>;
    fn modeled_choices(&self, viewer: usize, response: &str) -> Result<Vec<String>>;
    fn purpose(&self, player: usize) -> f64;
    fn view(&self) -> serde_json::Value;
    /// Only these participants may receive this actor's current announced intent.
    fn partners(&self, player: usize) -> Vec<usize>;
    fn offers(&self, announced: &[String]) -> Result<Vec<coordination::Offer>>;
    /// Game-proposed public subgoals; the same person's policy chooses among them.
    fn route_input(&self, _player: usize) -> Result<Option<reflex_immediate::Observation>> {
        Ok(None)
    }
    /// Context for this actor's appraisal, never a world transition or extra reward.
    fn with_route(&self, _player: usize, route: Option<&str>) -> Result<Self> {
        if route.is_some() {
            return Err(reflex_owned::Error::Invalid("game has no route context"));
        }
        Ok(self.clone())
    }
    fn current_route(&self, _player: usize) -> Option<&str> {
        None
    }
    /// Compatibility of already disclosed joint intentions; independent actions stay legal.
    fn agreement_compatible(&self, _known: &BTreeMap<usize, String>) -> Result<bool> {
        Ok(true)
    }
}

/// Validate an offer using only intentions disclosed to all of its participants.
/// Roots are applied over the current partial negotiation, never over hidden rivals.
pub fn offer_compatible<G: ArenaGame>(
    game: &G,
    offer: &coordination::Offer,
    choices: &[String],
) -> Result<bool> {
    let known = known_offer_intentions(game, offer, choices)?;
    game.agreement_compatible(&known)
}

fn known_offer_intentions<G: ArenaGame>(
    game: &G,
    offer: &coordination::Offer,
    choices: &[String],
) -> Result<BTreeMap<usize, String>> {
    let invalid =
        || reflex_owned::Error::Invalid("valid mutually shared offer and intentions required");
    if choices.len() != game.players() || !(2..=8).contains(&offer.roots.len()) || game.terminal() {
        return Err(invalid());
    }
    let members: BTreeSet<usize> = offer.roots.keys().copied().collect();
    if members.iter().any(|&p| p >= game.players()) {
        return Err(invalid());
    }
    let mut partner_sets = Vec::new();
    for &member in &members {
        let partners = game.partners(member);
        let set: BTreeSet<_> = partners.iter().copied().collect();
        if set.len() != partners.len()
            || set.iter().any(|&p| p >= game.players() || p == member)
            || members.iter().any(|&p| p != member && !set.contains(&p))
        {
            return Err(invalid());
        }
        partner_sets.push((member, set));
    }
    let viewer = *members.iter().next().ok_or_else(invalid)?;
    let mut known = BTreeMap::new();
    for (person, current) in choices.iter().enumerate() {
        if members.contains(&person)
            || partner_sets
                .iter()
                .all(|(member, set)| *member == person || set.contains(&person))
        {
            let action = offer.roots.get(&person).unwrap_or(current);
            if !game.behavior(viewer, person)?.actions.contains_key(action) {
                return Err(invalid());
            }
            known.insert(person, action.clone());
        }
    }
    Ok(known)
}

/// Independent public choices, sampled with common random numbers across roots.
pub fn public_scenarios<G: ArenaGame>(
    game: &G,
    viewer: usize,
    beliefs: &behavior::Beliefs,
    seed: u64,
) -> Result<Vec<Vec<String>>> {
    let mut scenarios = vec![Vec::new(); 3];
    for person in 0..game.players() {
        let frame = game.behavior(viewer, person)?;
        let probabilities = if person == viewer {
            vec![(
                frame
                    .actions
                    .keys()
                    .next()
                    .ok_or(reflex_owned::Error::Invalid("legal placeholder required"))?
                    .clone(),
                1.,
            )]
        } else {
            beliefs.predict(&frame, person)?
        };
        for (sample, choices) in scenarios.iter_mut().enumerate() {
            let mut bits = seed
                ^ game.tick().wrapping_mul(0x9e3779b97f4a7c15)
                ^ (person as u64).wrapping_mul(0xbf58476d1ce4e5b9)
                ^ (sample as u64).wrapping_mul(0x94d049bb133111eb);
            bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
            let u = ((bits ^ (bits >> 31)) >> 11) as f64 / ((1_u64 << 53) as f64);
            let mut sum = 0.;
            let mut selected = &probabilities
                .last()
                .ok_or(reflex_owned::Error::Invalid("nonempty public model"))?
                .0;
            for (action, probability) in &probabilities {
                sum += probability;
                if u < sum {
                    selected = action;
                    break;
                }
            }
            choices.push(selected.clone());
        }
    }
    Ok(scenarios)
}

pub fn public_input<G: ArenaGame>(
    game: &G,
    viewer: usize,
    beliefs: &behavior::Beliefs,
    seed: u64,
) -> Result<Input> {
    game.input_scenarios(viewer, &public_scenarios(game, viewer, beliefs, seed)?)
}

/// Exactly the viewer's permitted partners may be fixed, for this tick only.
pub(crate) fn public_scenarios_with_intentions<G: ArenaGame>(
    game: &G,
    viewer: usize,
    beliefs: &behavior::Beliefs,
    seed: u64,
    intentions: &Intentions,
) -> Result<Vec<Vec<String>>> {
    if viewer >= game.players() || intentions.tick != game.tick() {
        return Err(reflex_owned::Error::Invalid(
            "current valid viewer required",
        ));
    }
    let partners = game.partners(viewer);
    let unique: BTreeSet<_> = partners.iter().copied().collect();
    if unique.len() != partners.len()
        || unique.iter().any(|&p| p >= game.players() || p == viewer)
        || intentions.choices.keys().copied().collect::<BTreeSet<_>>() != unique
    {
        return Err(reflex_owned::Error::Invalid(
            "current complete partner intentions required",
        ));
    }
    for (&person, action) in &intentions.choices {
        if !game.behavior(viewer, person)?.actions.contains_key(action) {
            return Err(reflex_owned::Error::Invalid(
                "legal public partner intention required",
            ));
        }
    }
    let mut scenarios = public_scenarios(game, viewer, beliefs, seed)?;
    for row in &mut scenarios {
        for (&person, action) in &intentions.choices {
            row[person] = action.clone();
        }
    }
    Ok(scenarios)
}

pub fn public_input_with_intentions<G: ArenaGame>(
    game: &G,
    viewer: usize,
    beliefs: &behavior::Beliefs,
    seed: u64,
    intentions: &Intentions,
) -> Result<Input> {
    game.input_scenarios(
        viewer,
        &public_scenarios_with_intentions(game, viewer, beliefs, seed, intentions)?,
    )
}

/// Call after actions are revealed. The game supplies only viewer-visible frames.
pub fn observe_public<G: ArenaGame>(
    game: &G,
    viewer: usize,
    beliefs: &mut behavior::Beliefs,
    choices: &[String],
) -> Result<()> {
    if choices.len() != game.players() {
        return Err(reflex_owned::Error::Invalid(
            "complete public choices required",
        ));
    }
    let mut next = beliefs.clone();
    for (person, choice) in choices.iter().enumerate() {
        if person != viewer {
            next.observe(&game.behavior(viewer, person)?, person, choice, game.tick())?;
        }
    }
    *beliefs = next;
    Ok(())
}
