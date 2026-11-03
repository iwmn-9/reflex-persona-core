//! Bounded public-model planning. Separate research profile, not fixed H32.
use crate::{behavior::Beliefs, observe_public, public_input, public_scenarios, ArenaGame};
use reflex_immediate::{Action, Continuation, ContinuationPlan, Outcome, ReturnUnits};
use reflex_owned::{Error, Feedback, LearnedState, Owner, OwnerConfig, Result, Stage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub horizon: u32,
    pub max_regret: f64,
    pub model_seed: u64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            horizon: 8,
            max_regret: 0.15,
            model_seed: 701,
        }
    }
}
impl Config {
    fn validate(&self) -> Result<()> {
        if !(1..=32).contains(&self.horizon)
            || !self.max_regret.is_finite()
            || !(0.0..=1.0).contains(&self.max_regret)
        {
            return Err(Error::Invalid("bounded planner settings required"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Milestone {
    pub remaining: Vec<String>,
    pub value: f64,
    pub deadline: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Planner {
    config: Config,
    owner: OwnerConfig,
    milestone: Option<Milestone>,
    last_tick: Option<u64>,
    owner_state_sha256: String,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlannerSave {
    version: String,
    planner: Planner,
    sha256: String,
}
#[derive(Clone)]
struct Path {
    outcome: Outcome,
    purpose: f64,
    actions: Vec<String>,
    goals: Vec<f64>,
}
struct Candidate {
    root: String,
    paths: Vec<Path>,
    purpose: f64,
    ordinary: bool,
}
/// Proposal contains a detached actor and a single-use stage; discard to cancel.
pub struct Proposal {
    actor: Owner,
    stage: Stage,
    before: LearnedState,
    owner: OwnerConfig,
    tick: u64,
    action: String,
    next_milestone: Option<Milestone>,
    planner_before: Vec<u8>,
    pub audit: Audit,
}
pub struct Consent {
    pub ranked: Vec<String>,
    pub proposals: BTreeMap<String, Proposal>,
    pub simulated_nodes: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct Audit {
    pub reason: String,
    pub modeled_purpose: f64,
    pub milestone: Option<Milestone>,
    pub simulated_nodes: usize,
}
fn owner_state_hash(owner: &Owner) -> Result<String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(owner.state()?)?)
    ))
}
impl Proposal {
    pub fn action(&self) -> &str {
        &self.action
    }
}

impl Planner {
    pub fn new(owner: &Owner, config: Config) -> Result<Self> {
        config.validate()?;
        owner.state()?;
        if !owner.config().contract.contains("estimated-model") {
            return Err(Error::Invalid("estimated-model owner contract required"));
        }
        Ok(Self {
            config,
            owner: owner.config().clone(),
            milestone: None,
            last_tick: owner.state()?.last_tick,
            owner_state_sha256: owner_state_hash(owner)?,
        })
    }
    pub fn milestone(&self) -> Option<&Milestone> {
        self.milestone.as_ref()
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    /// Restore only against an independently supplied owner; personality is checked.
    pub fn restore(owner: &Owner, bytes: &[u8]) -> Result<Self> {
        let save: PlannerSave = serde_json::from_slice(bytes)?;
        if save.version != "arena-planner-v2"
            || save.sha256 != format!("{:x}", Sha256::digest(serde_json::to_vec(&save.planner)?))
        {
            return Err(Error::Invalid("planner checkpoint integrity mismatch"));
        }
        let p = save.planner;
        p.config.validate()?;
        owner.state()?;
        if p.owner != *owner.config()
            || !p.owner.contract.contains("estimated-model")
            || p.last_tick != owner.state()?.last_tick
            || p.owner_state_sha256 != owner_state_hash(owner)?
        {
            return Err(Error::Invalid("planner owner mismatch"));
        }
        if let Some(m) = &p.milestone {
            if m.remaining.is_empty()
                || m.remaining.len() > 32
                || !m.value.is_finite()
                || !(-1.0..=1.0).contains(&m.value)
                || m.remaining.iter().any(|s| s.is_empty())
                || owner.state()?.last_tick.is_some_and(|t| m.deadline <= t)
            {
                return Err(Error::Invalid("invalid milestone"));
            }
        }
        Ok(p)
    }
    pub fn checkpoint(&self) -> Result<Vec<u8>> {
        let save = PlannerSave {
            version: "arena-planner-v2".into(),
            planner: self.clone(),
            sha256: format!("{:x}", Sha256::digest(serde_json::to_vec(self)?)),
        };
        Ok(serde_json::to_vec(&save)?)
    }
    /// Model uses three public response scenarios and the owner's own feedback loop.
    pub fn propose<G: ArenaGame>(
        &self,
        owner: &Owner,
        game: &G,
        player: usize,
    ) -> Result<Proposal> {
        self.propose_inner(owner, game, player, None)
    }
    pub fn propose_with_beliefs<G: ArenaGame>(
        &self,
        owner: &Owner,
        game: &G,
        player: usize,
        beliefs: &Beliefs,
    ) -> Result<Proposal> {
        let scope = serde_json::to_string(&(
            &owner.config().actor.scope.game,
            &owner.config().actor.scope.episode,
        ))?;
        beliefs.validate(&scope, player, &game.behavior(player, player)?.alphabet)?;
        self.propose_inner(owner, game, player, Some(beliefs))
    }
    fn propose_inner<G: ArenaGame>(
        &self,
        owner: &Owner,
        game: &G,
        player: usize,
        beliefs: Option<&Beliefs>,
    ) -> Result<Proposal> {
        self.config.validate()?;
        if owner.config() != &self.owner
            || owner.state()?.last_tick != self.last_tick
            || owner_state_hash(owner)? != self.owner_state_sha256
            || player >= game.players()
            || game.terminal()
        {
            return Err(Error::Invalid("live matching arena owner required"));
        }
        let before = owner.state()?.clone();
        let input = match beliefs {
            Some(b) => public_input(game, player, b, self.config.model_seed)?,
            None => game.input(player)?,
        };
        if input.observation.tick != game.tick() || input.responses.legal != ["0", "1", "2"] {
            return Err(Error::Invalid(
                "matching clock and public response alphabet required",
            ));
        }
        let actor = owner.fork_model()?;
        let stage = actor.prepare_estimated(&input, false)?;
        let normal = stage.decision().action_id.clone();
        let allowed = stage.allowed().clone();
        let live = self.milestone.as_ref().filter(|m| {
            game.tick() < m.deadline
                && game.purpose(player) < m.value - 1e-12
                && !m.remaining.is_empty()
                && allowed.contains(&m.remaining[0])
        });
        let mut roots: Vec<String> = if let Some(m) = live {
            vec![m.remaining[0].clone(), normal]
        } else {
            allowed.iter().cloned().collect()
        };
        roots.sort();
        roots.dedup();
        let probabilities = if beliefs.is_some() {
            vec![1. / 3.; 3]
        } else {
            actor.predict(&input.responses)?
        };
        if probabilities.len() != 3 {
            return Err(Error::Invalid("three public responses required"));
        }
        let mut candidates = Vec::new();
        for root in roots {
            let mut schedules: Vec<Option<Vec<String>>> = vec![None];
            if let Some(m) = live.filter(|m| m.remaining[0] == root) {
                schedules.insert(0, Some(m.remaining.clone()));
            } else if live.is_none() && self.config.horizon > 1 {
                // Enumerate a bounded second move from a public response scenario.
                // Other scenarios still check legality and fall back independently.
                let model = game.model_world(player, self.config.model_seed)?;
                let mut choices = match beliefs {
                    Some(b) => {
                        public_scenarios(&model, player, b, self.config.model_seed)?.remove(0)
                    }
                    None => model.modeled_choices(player, "0")?,
                };
                if choices.len() != game.players() {
                    return Err(Error::Invalid("modeled choices require all players"));
                }
                choices[player] = root.clone();
                let (next, _) = model.advance(&choices, self.config.model_seed)?;
                if !next.terminal() {
                    for a in next
                        .input(player)?
                        .observation
                        .actions
                        .into_iter()
                        .filter(|a| a.legal && !a.known_failure)
                        .take(8)
                    {
                        schedules.push(Some(vec![root.clone(), a.id]));
                    }
                }
            }
            for schedule in schedules {
                if candidates.len() >= 256 {
                    return Err(Error::Invalid("bounded plan count exceeded"));
                }
                let mut paths = Vec::new();
                for scenario in 0..3 {
                    paths.push(simulate(
                        owner,
                        game,
                        player,
                        &root,
                        scenario,
                        self.config.horizon,
                        self.config.model_seed,
                        schedule.as_deref(),
                        beliefs,
                        None,
                        false,
                    )?);
                }
                let purpose = paths
                    .iter()
                    .zip(&probabilities)
                    .map(|(p, w)| p.purpose * w)
                    .sum::<f64>()
                    .clamp(-1., 1.);
                candidates.push(Candidate {
                    root: root.clone(),
                    paths,
                    purpose,
                    ordinary: schedule.is_none(),
                });
            }
        }
        let plans = plans(&candidates, &probabilities);
        let forecast = Continuation {
            owner: owner.config().actor.clone(),
            state: before.intent.clone(),
            tick: game.tick(),
            target: owner.config().purpose_target.clone(),
            horizon: self.config.horizon,
            max_regret: 2.0,
            units: ReturnUnits {
                unit: "arena-turn".into(),
                effects: [self.config.horizon as f64; 22],
                root_terms: self.config.horizon as f64,
            },
            plans,
        };
        // First establish the person's tier over the complete forecast, before
        // purpose can discard any candidates.
        let appraisal = actor.with_continuation(stage, &forecast)?;
        let tier: std::collections::BTreeSet<usize> = appraisal
            .decision()
            .scores
            .iter()
            .filter(|s| s.eligible)
            .filter_map(|s| s.id.strip_prefix("candidate-").and_then(|v| v.parse().ok()))
            .collect();
        let mut bounded = forecast.clone();
        bounded.plans.retain(|p| {
            p.action
                .id
                .strip_prefix("candidate-")
                .and_then(|v| v.parse::<usize>().ok())
                .is_some_and(|i| tier.contains(&i))
        });
        bounded.max_regret = bounded_regret(&candidates, &tier, self.config.max_regret);
        let evaluated =
            actor.with_continuation(actor.prepare_estimated(&input, false)?, &bounded)?;
        let mut filtered = bounded.clone();
        if let Some(m) = live {
            if let Some(inc) = candidates.iter().position(|c| c.root == m.remaining[0]) {
                let eligible = |i: usize| {
                    evaluated
                        .decision()
                        .scores
                        .iter()
                        .any(|s| s.id == format!("candidate-{i:03}") && s.eligible)
                };
                let arrival = |c: &Candidate| {
                    c.paths
                        .iter()
                        .map(|p| {
                            p.goals
                                .iter()
                                .position(|v| *v >= m.value - 1e-12)
                                .map(|i| i + 1)
                        })
                        .collect::<Vec<_>>()
                };
                let incumbent = arrival(&candidates[inc]);
                if !eligible(inc) || incumbent.iter().any(Option::is_none) {
                    let mut fresh = self.clone();
                    fresh.milestone = None;
                    let mut proposal = fresh.propose_inner(owner, game, player, beliefs)?;
                    proposal.audit.reason = "withdraw".into();
                    proposal.planner_before = self.checkpoint()?;
                    return Ok(proposal);
                }
                if eligible(inc) && incumbent.iter().all(Option::is_some) {
                    let latest = incumbent.iter().flatten().copied().max().unwrap();
                    filtered.plans.clear();
                    for plan in &bounded.plans {
                        if keep_retained_plan(plan, &candidates, inc, latest, m.value, &eligible)? {
                            filtered.plans.push(plan.clone());
                        }
                    }
                }
            }
        }
        let stage = if filtered.plans.len() == bounded.plans.len() {
            evaluated
        } else {
            actor.with_continuation(actor.prepare_estimated(&input, false)?, &filtered)?
        };
        let action = stage.decision().action_id.clone();
        let selected = stage
            .selected_plan()
            .and_then(|s| s.strip_prefix("candidate-"))
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or(Error::Invalid("selected plan identity required"))?;
        let selected_candidate = candidates
            .get(selected)
            .ok_or(Error::Invalid("selected plan outside candidate set"))?;
        if selected_candidate.root != action {
            return Err(Error::Invalid("selected plan root mismatch"));
        }
        let next_milestone = if let Some(m) = live.filter(|m| {
            m.remaining[0] == action
                && candidates
                    .iter()
                    .position(|c| std::ptr::eq(c, selected_candidate))
                    == candidates.iter().position(|c| c.root == m.remaining[0])
        }) {
            Some(m.clone())
        } else {
            new_milestone(
                &candidates,
                selected,
                game.purpose(player),
                game.tick(),
                self.config.horizon,
            )
        };
        Ok(Proposal {
            actor,
            stage,
            before,
            owner: self.owner.clone(),
            tick: game.tick(),
            action,
            audit: Audit {
                reason: if live.is_some()
                    && next_milestone
                        .as_ref()
                        .is_some_and(|m| live.is_some_and(|old| old.deadline == m.deadline))
                {
                    "keep"
                } else if self.milestone.is_some()
                    && game.purpose(player) >= self.milestone.as_ref().unwrap().value - 1e-12
                {
                    "reached"
                } else if self.milestone.is_some() {
                    "withdraw"
                } else {
                    "replan"
                }
                .into(),
                modeled_purpose: selected_candidate.purpose,
                milestone: next_milestone.clone(),
                simulated_nodes: candidates
                    .iter()
                    .flat_map(|c| &c.paths)
                    .map(|p| p.actions.len())
                    .sum(),
            },
            next_milestone,
            planner_before: self.checkpoint()?,
        })
    }
    /// The engine executes all chosen actions together, then supplies actual feedback.
    /// A detached replacement is validated before the factual owner is swapped.
    pub fn resolve(
        &mut self,
        owner: &mut Owner,
        mut proposal: Proposal,
        actual_action: &str,
        feedback: &Feedback,
        actual_goal: f64,
    ) -> Result<()> {
        if self.checkpoint()? != proposal.planner_before
            || owner.config() != &proposal.owner
            || owner.state()? != &proposal.before
            || actual_action != proposal.action
            || !actual_goal.is_finite()
            || !(-1.0..=1.0).contains(&actual_goal)
        {
            return Err(Error::Invalid("stale proposal or mismatched execution"));
        }
        let ticket = proposal.actor.commit(proposal.stage, actual_action)?;
        proposal.actor.resolve(ticket, feedback)?;
        let mut next = proposal.next_milestone;
        if let Some(m) = next.as_mut() {
            if m.remaining.first().is_some_and(|x| x == actual_action) {
                m.remaining.remove(0);
            }
        }
        if next.as_ref().is_some_and(|m| {
            m.remaining.is_empty()
                || actual_goal >= m.value - 1e-12
                || proposal.tick + 1 >= m.deadline
        }) {
            next = None
        }
        let last_tick = proposal.actor.state()?.last_tick;
        let owner_state_sha256 = owner_state_hash(&proposal.actor)?;
        *owner = proposal.actor;
        self.milestone = next;
        self.last_tick = last_tick;
        self.owner_state_sha256 = owner_state_sha256;
        Ok(())
    }

    /// Each person privately evaluates joint offers against its own outside action.
    /// Only current intentions from permitted communication partners are used.
    pub fn consent<G: ArenaGame>(
        &self,
        owner: &Owner,
        game: &G,
        player: usize,
        beliefs: &Beliefs,
        announced: &[String],
        offers: &[crate::coordination::Offer],
    ) -> Result<Consent> {
        let scope = serde_json::to_string(&(
            &owner.config().actor.scope.game,
            &owner.config().actor.scope.episode,
        ))?;
        beliefs.validate(&scope, player, &game.behavior(player, player)?.alphabet)?;
        if announced.len() != game.players()
            || owner.config() != &self.owner
            || owner.state()?.last_tick != self.last_tick
            || owner_state_hash(owner)? != self.owner_state_sha256
        {
            return Err(Error::Invalid("current actor and intentions required"));
        }
        let input = public_input(game, player, beliefs, self.config.model_seed)?;
        let actor = owner.fork_model()?;
        let base = actor.prepare_estimated(&input, false)?;
        let allowed = base.allowed().clone();
        let partners = game.partners(player);
        if partners
            .iter()
            .any(|p| *p >= game.players() || *p == player)
        {
            return Err(Error::Invalid("valid communication partners required"));
        }
        let public: BTreeMap<_, _> = partners
            .iter()
            .copied()
            .map(|p| (p, announced[p].clone()))
            .collect();
        let mut intents = vec![(None, announced[player].clone(), public.clone())];
        for offer in offers {
            if let Some(root) = offer.roots.get(&player) {
                if allowed.contains(root)
                    && offer
                        .roots
                        .keys()
                        .all(|p| *p == player || partners.contains(p))
                {
                    let mut first = public.clone();
                    first.extend(offer.roots.clone());
                    intents.push((Some(offer.id.clone()), root.clone(), first));
                }
            }
        }
        if intents.len() == 1 {
            return Ok(Consent {
                ranked: vec![],
                proposals: BTreeMap::new(),
                simulated_nodes: 0,
            });
        }
        let mut candidates = Vec::new();
        for (id, root, first) in &intents {
            let schedule = if id.is_none() {
                self.milestone
                    .as_ref()
                    .filter(|m| m.remaining.first() == Some(root))
                    .map(|m| m.remaining.as_slice())
            } else {
                None
            };
            let paths = (0..3)
                .map(|scenario| {
                    simulate(
                        owner,
                        game,
                        player,
                        root,
                        scenario,
                        self.config.horizon,
                        self.config.model_seed,
                        schedule,
                        Some(beliefs),
                        Some(first),
                        id.is_some(),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let purpose = paths.iter().map(|p| p.purpose).sum::<f64>() / 3.;
            candidates.push(Candidate {
                root: root.clone(),
                paths,
                purpose,
                ordinary: true,
            });
        }
        let forecast = Continuation {
            owner: owner.config().actor.clone(),
            state: owner.state()?.intent.clone(),
            tick: game.tick(),
            target: owner.config().purpose_target.clone(),
            horizon: self.config.horizon,
            max_regret: 2.,
            units: ReturnUnits {
                unit: "arena-turn".into(),
                effects: [self.config.horizon as f64; 22],
                root_terms: self.config.horizon as f64,
            },
            plans: plans(&candidates, &[1. / 3.; 3]),
        };
        let tier_appraisal = actor.with_continuation(base, &forecast)?;
        let tier = tier_appraisal
            .decision()
            .scores
            .iter()
            .filter(|s| s.eligible)
            .filter_map(|s| {
                s.id.strip_prefix("candidate-")
                    .and_then(|x| x.parse::<usize>().ok())
            })
            .collect();
        let mut bounded = forecast.clone();
        bounded.plans.retain(|p| {
            p.action
                .id
                .strip_prefix("candidate-")
                .and_then(|s| s.parse::<usize>().ok())
                .is_some_and(|i| std::collections::BTreeSet::contains(&tier, &i))
        });
        bounded.max_regret = bounded_regret(&candidates, &tier, self.config.max_regret);
        let appraisal =
            actor.with_continuation(actor.prepare_estimated(&input, false)?, &bounded)?;
        let outside = tier_appraisal
            .decision()
            .scores
            .iter()
            .find(|s| s.id == "candidate-000")
            .ok_or(Error::Invalid("outside score required"))?;
        let mut choices: Vec<_> = appraisal
            .decision()
            .scores
            .iter()
            .filter_map(|s| {
                let i = s.id.strip_prefix("candidate-")?.parse::<usize>().ok()?;
                if i == 0 || !s.eligible {
                    return None;
                }
                // Respect personality tiers first, then compare within the same tier.
                if outside.eligible && s.score <= outside.score + 1e-9 {
                    return None;
                }
                Some((i, s.score))
            })
            .collect();
        choices.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut result = Consent {
            ranked: vec![],
            proposals: BTreeMap::new(),
            simulated_nodes: candidates
                .iter()
                .flat_map(|c| &c.paths)
                .map(|p| p.actions.len())
                .sum(),
        };
        for (i, _) in choices {
            let (id, root, first) = &intents[i];
            let id = id.as_ref().unwrap();
            let mut scenarios = public_scenarios(game, player, beliefs, self.config.model_seed)?;
            for row in &mut scenarios {
                for (p, a) in first {
                    row[*p] = a.clone();
                }
            }
            let mut signed = game.input_scenarios(player, &scenarios)?;
            signed_context(&mut signed);
            let executing = owner.fork_model()?;
            let prepared = executing.prepare_estimated(&signed, false)?;
            if !prepared.allowed().contains(root) {
                continue;
            }
            let mut single = forecast.clone();
            single.plans = vec![forecast.plans[i].clone()];
            let stage = executing.with_continuation(prepared, &single)?;
            let milestone = new_milestone(
                &candidates,
                i,
                game.purpose(player),
                game.tick(),
                self.config.horizon,
            );
            result.ranked.push(id.clone());
            result.proposals.insert(
                id.clone(),
                Proposal {
                    actor: executing,
                    stage,
                    before: owner.state()?.clone(),
                    owner: owner.config().clone(),
                    tick: game.tick(),
                    action: root.clone(),
                    next_milestone: milestone.clone(),
                    planner_before: self.checkpoint()?,
                    audit: Audit {
                        reason: "agreement".into(),
                        modeled_purpose: candidates[i].purpose,
                        milestone,
                        simulated_nodes: candidates
                            .iter()
                            .flat_map(|c| &c.paths)
                            .map(|p| p.actions.len())
                            .sum(),
                    },
                },
            );
        }
        Ok(result)
    }
}
fn signed_context(input: &mut reflex_owned::Input) {
    let context = "accepted-current-agreement".to_string();
    input
        .observation
        .facts
        .insert("agreement".into(), context.clone());
    for binding in input.bindings.values_mut() {
        binding.situation = format!("{}|{context}", binding.situation);
    }
}
fn uniform(mut x: u64) -> f64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((x ^ (x >> 31)) >> 11) as f64 / ((1_u64 << 53) as f64)
}
#[allow(clippy::too_many_arguments)]
fn simulate<G: ArenaGame>(
    owner: &Owner,
    world: &G,
    player: usize,
    root: &str,
    scenario: usize,
    horizon: u32,
    seed: u64,
    schedule: Option<&[String]>,
    beliefs: Option<&Beliefs>,
    first: Option<&BTreeMap<usize, String>>,
    signed: bool,
) -> Result<Path> {
    let mut actor = owner.model()?;
    let mut public = beliefs.cloned();
    let mut game = world.model_world(player, seed.wrapping_add(scenario as u64))?;
    let mut effects = [0.; 22];
    let mut actions = Vec::new();
    let mut goals = Vec::new();
    for t in 0..horizon {
        if game.terminal() {
            break;
        }
        let mut sampled = public
            .as_ref()
            .map(|b| public_scenarios(&game, player, b, seed))
            .transpose()?;
        if t == 0 {
            if let (Some(rows), Some(first)) = (&mut sampled, first) {
                for row in rows {
                    for (p, a) in first {
                        row[*p] = a.clone();
                    }
                }
            }
        }
        let mut input = match &sampled {
            Some(choices) => game.input_scenarios(player, choices)?,
            None => game.input(player)?,
        };
        if t == 0 && signed && first.is_some() {
            signed_context(&mut input);
        }
        if input.observation.tick != game.tick() || input.responses.legal != ["0", "1", "2"] {
            return Err(Error::Invalid("matching model observation required"));
        }
        let normal = actor.prepare_estimated(&input)?;
        let proposed = if t == 0 {
            Some(root)
        } else {
            schedule.and_then(|s| s.get(t as usize)).map(String::as_str)
        };
        let stage = if let Some(action) = proposed {
            if t == 0
                || normal.allowed().contains(action)
                    && input
                        .observation
                        .actions
                        .iter()
                        .any(|a| a.id == action && a.legal && !a.known_failure)
            {
                actor.prepare_root_estimated(&input, action)?
            } else {
                normal
            }
        } else {
            normal
        };
        let action = stage.decision().action_id.clone();
        let probs = actor
            .state()?
            .prediction
            .predict(&actor.config().predictor, &input.responses)?;
        let index = if t == 0 {
            scenario
        } else {
            let u =
                uniform(seed ^ game.tick().wrapping_mul(0x9e3779b97f4a7c15) ^ (scenario as u64));
            let mut sum = 0.;
            let mut picked = 2;
            for (i, p) in probs.iter().enumerate() {
                sum += p;
                if u < sum {
                    picked = i;
                    break;
                }
            }
            picked
        };
        let response = &input.responses.legal[index];
        let mut choices = match sampled {
            Some(choices) => choices[index].clone(),
            None => game.modeled_choices(player, response)?,
        };
        if choices.len() != game.players() {
            return Err(Error::Invalid("modeled choices require all players"));
        }
        choices[player] = action.clone();
        let (next, feedback) = game.advance(
            &choices,
            seed.wrapping_add((scenario as u64) * 37)
                .wrapping_add(t as u64),
        )?;
        if next.tick() <= game.tick() || feedback.len() != game.players() {
            return Err(Error::Invalid("modeled transition feedback mismatch"));
        }
        let row = feedback[player]
            .observed
            .ok_or(Error::Invalid("modeled effect observation required"))?;
        for (total, value) in effects.iter_mut().zip(row) {
            *total += value;
        }
        let ticket = actor.commit(stage, &action)?;
        let mut observed = feedback[player].clone();
        if let Some(b) = &mut public {
            observe_public(&game, player, b, &choices)?;
            observed.response = None; // A sample number is not an observed behavior class.
        }
        actor.resolve(ticket, &observed)?;
        let goal = next.purpose(player);
        if !goal.is_finite() || !(-1.0..=1.0).contains(&goal) {
            return Err(Error::Invalid("bounded public purpose required"));
        }
        actions.push(action);
        goals.push(goal);
        game = next;
    }
    let scale = horizon as f64;
    for e in &mut effects {
        *e = (*e / scale).clamp(-1., 1.)
    }
    effects[21] = effects[21].max(0.);
    let mut needs = [0.; 5];
    needs.copy_from_slice(&effects[1..6]);
    let mut values = [0.; 10];
    values.copy_from_slice(&effects[6..16]);
    let mut style = [0.; 5];
    style.copy_from_slice(&effects[16..21]);
    Ok(Path {
        outcome: Outcome {
            p: 0.,
            objective: effects[0],
            needs,
            values,
            style,
            cost: effects[21],
        },
        purpose: game.purpose(player),
        actions,
        goals,
    })
}
fn keep_retained_plan(
    plan: &ContinuationPlan,
    candidates: &[Candidate],
    incumbent: usize,
    latest: usize,
    target: f64,
    eligible: &impl Fn(usize) -> bool,
) -> Result<bool> {
    let index = plan
        .action
        .id
        .strip_prefix("candidate-")
        .and_then(|v| v.parse::<usize>().ok())
        .ok_or(Error::Invalid("invalid candidate id"))?;
    let candidate = candidates
        .get(index)
        .ok_or(Error::Invalid("candidate id out of range"))?;
    let retained = candidates
        .get(incumbent)
        .ok_or(Error::Invalid("incumbent id out of range"))?;
    if plan.root != candidate.root {
        return Err(Error::Invalid("candidate root mismatch"));
    }
    if !eligible(index) {
        return Ok(false);
    }
    if index == incumbent {
        return Ok(true);
    }
    let arrivals: Vec<_> = candidate
        .paths
        .iter()
        .map(|p| {
            p.goals
                .iter()
                .position(|v| *v >= target - 1e-12)
                .map(|i| i + 1)
        })
        .collect();
    Ok(!(candidate.purpose <= retained.purpose + 1e-12
        && arrivals.iter().all(Option::is_some)
        && arrivals.iter().flatten().all(|x| latest < *x)))
}

fn bounded_regret(
    candidates: &[Candidate],
    tier: &std::collections::BTreeSet<usize>,
    fraction: f64,
) -> f64 {
    fn spread(
        candidates: &[Candidate],
        tier: &std::collections::BTreeSet<usize>,
        ordinary: bool,
    ) -> f64 {
        let mut roots = std::collections::BTreeMap::<&str, f64>::new();
        for (i, candidate) in candidates.iter().enumerate() {
            if tier.contains(&i) && (!ordinary || candidate.ordinary) {
                roots
                    .entry(&candidate.root)
                    .and_modify(|best| *best = best.max(candidate.purpose))
                    .or_insert(candidate.purpose);
            }
        }
        let low = roots.values().copied().fold(f64::INFINITY, f64::min);
        let high = roots.values().copied().fold(f64::NEG_INFINITY, f64::max);
        if roots.is_empty() {
            0.
        } else {
            high - low
        }
    }
    fraction * spread(candidates, tier, false).min(spread(candidates, tier, true))
}

fn plans(candidates: &[Candidate], probabilities: &[f64]) -> Vec<ContinuationPlan> {
    candidates
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let outcomes = c
                .paths
                .iter()
                .zip(probabilities)
                .map(|(path, p)| {
                    let mut o = path.outcome.clone();
                    o.p = *p;
                    o
                })
                .collect();
            ContinuationPlan {
                root: c.root.clone(),
                action: Action {
                    id: format!("candidate-{i:03}"),
                    target: None,
                    legal: true,
                    known_failure: false,
                    confidence: 1.,
                    familiarity: 0.,
                    switch_cost: 0.,
                    outcomes,
                },
                purpose: c.purpose,
                correction: None,
            }
        })
        .collect()
}
fn new_milestone(
    candidates: &[Candidate],
    selected: usize,
    origin: f64,
    tick: u64,
    horizon: u32,
) -> Option<Milestone> {
    let chosen = candidates.get(selected)?;
    let other = candidates
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != selected)
        .map(|(_, c)| c)
        .map(|c| c.purpose)
        .fold(origin, f64::max);
    for i in 1..horizon as usize {
        if chosen
            .paths
            .iter()
            .any(|p| p.actions.len() <= i || p.goals.len() <= i)
        {
            break;
        }
        let prefix = &chosen.paths[0].actions[..=i];
        if chosen.paths.iter().any(|p| &p.actions[..=i] != prefix) {
            break;
        }
        let floor = chosen
            .paths
            .iter()
            .map(|p| p.goals[i])
            .fold(f64::INFINITY, f64::min);
        if floor > origin + 1e-12 && floor > other + 1e-12 {
            return Some(Milestone {
                remaining: prefix.to_vec(),
                value: floor,
                deadline: tick + i as u64 + 1,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn candidate(root: &str, purpose: f64, ordinary: bool) -> Candidate {
        Candidate {
            root: root.into(),
            paths: vec![],
            purpose,
            ordinary,
        }
    }
    #[test]
    fn purpose_band_is_limited_by_tier_and_ordinary_root_spreads() {
        let rows = vec![
            candidate("a", 0.2, true),
            candidate("b", 0.4, true),
            candidate("a", 0.9, false),
            candidate("b", 0.8, false),
            candidate("c", 1.0, false),
        ];
        let tier = BTreeSet::from([0, 1, 2, 3]);
        assert!((bounded_regret(&rows, &tier, 0.15) - 0.015).abs() < 1e-12);
        let tier = BTreeSet::from([0, 1, 2, 3, 4]);
        assert!((bounded_regret(&rows, &tier, 0.15) - 0.03).abs() < 1e-12);
        let tier = BTreeSet::from([2, 3]);
        assert_eq!(bounded_regret(&rows, &tier, 0.15), 0.);
    }
    #[test]
    fn retained_filter_uses_original_plan_ids_after_tier_removes_zero_and_two() {
        let mut rows = vec![
            candidate("excluded-zero", 0.9, true),
            candidate("retained", 0.6, false),
            candidate("excluded-two", 0.9, true),
            candidate("slow", 0.5, true),
            candidate("fast", 0.5, true),
        ];
        let outcome = Outcome {
            p: 1.,
            objective: 0.,
            needs: [0.; 5],
            values: [0.; 10],
            style: [0.; 5],
            cost: 0.,
        };
        for (i, goals) in [(1, vec![0.6]), (3, vec![0., 0.6]), (4, vec![0.6])] {
            rows[i].paths.push(Path {
                outcome: outcome.clone(),
                purpose: 0.6,
                actions: vec![],
                goals,
            });
        }
        let all = plans(&rows, &[1.]);
        let eligible = |i| [1, 3, 4].contains(&i);
        let kept: Vec<_> = [1, 3, 4]
            .into_iter()
            .filter(|i| keep_retained_plan(&all[*i], &rows, 1, 1, 0.6, &eligible).unwrap())
            .collect();
        assert_eq!(kept, vec![1, 4]);
        let mut wrong_root = all[4].clone();
        wrong_root.root = "wrong".into();
        assert!(keep_retained_plan(&wrong_root, &rows, 1, 1, 0.6, &eligible).is_err());
    }
}
