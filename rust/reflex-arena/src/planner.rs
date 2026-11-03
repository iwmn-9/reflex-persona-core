//! Bounded public-model planning. Separate research profile, not fixed H32.
use crate::{
    behavior::Beliefs, observe_public, public_input, public_input_with_intentions,
    public_scenarios, public_scenarios_with_intentions, ArenaGame, Intentions,
};
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
    /// Number of explicit second-action candidates per root; zero keeps only ordinary tails.
    pub second_moves: usize,
    /// Independent three-scenario batches, using common seeds across candidates.
    pub scenario_batches: usize,
    /// Re-evaluate the same person's public route options in future model states.
    pub route_selection: bool,
    /// Require improvement over the ordinary continuation across leave-one-batch-out appraisals.
    pub forecast_guard: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            horizon: 8,
            max_regret: 0.15,
            model_seed: 701,
            second_moves: 8,
            scenario_batches: 1,
            route_selection: false,
            forecast_guard: false,
        }
    }
}
impl Config {
    fn validate(&self) -> Result<()> {
        if !(1..=32).contains(&self.horizon)
            || !self.max_regret.is_finite()
            || !(0.0..=1.0).contains(&self.max_regret)
            || self.second_moves > 8
            || !(1..=4).contains(&self.scenario_batches)
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
    pub pre_filter_action: String,
    pub pre_filter_purpose: f64,
    pub forecast_support: Option<crate::robustness::Support>,
    pub milestone: Option<Milestone>,
    pub simulated_nodes: usize,
}
fn selected_candidate<'a>(
    stage: &Stage,
    candidates: &'a [Candidate],
) -> Result<(usize, &'a Candidate)> {
    let index = stage
        .selected_plan()
        .and_then(|s| s.strip_prefix("candidate-"))
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or(Error::Invalid("selected plan identity required"))?;
    let candidate = candidates
        .get(index)
        .ok_or(Error::Invalid("selected plan outside candidate set"))?;
    if candidate.root.is_empty()
        || candidate.root != stage.decision().action_id
        || !candidate.purpose.is_finite()
    {
        return Err(Error::Invalid("selected plan root or purpose mismatch"));
    }
    Ok((index, candidate))
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
        if save.version != "arena-planner-v6"
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
            version: "arena-planner-v6".into(),
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
        self.propose_inner(owner, game, player, None, None)
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
        self.propose_inner(owner, game, player, Some(beliefs), None)
    }
    pub fn propose_with_intentions<G: ArenaGame>(
        &self,
        owner: &Owner,
        game: &G,
        player: usize,
        beliefs: &Beliefs,
        intentions: &Intentions,
    ) -> Result<Proposal> {
        let scope = serde_json::to_string(&(
            &owner.config().actor.scope.game,
            &owner.config().actor.scope.episode,
        ))?;
        beliefs.validate(&scope, player, &game.behavior(player, player)?.alphabet)?;
        self.propose_inner(owner, game, player, Some(beliefs), Some(intentions))
    }
    fn propose_inner<G: ArenaGame>(
        &self,
        owner: &Owner,
        game: &G,
        player: usize,
        beliefs: Option<&Beliefs>,
        intentions: Option<&Intentions>,
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
            Some(b) => match intentions {
                Some(frame) => {
                    public_input_with_intentions(game, player, b, self.config.model_seed, frame)?
                }
                None => public_input(game, player, b, self.config.model_seed)?,
            },
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
            vec![m.remaining[0].clone(), normal.clone()]
        } else {
            allowed.iter().cloned().collect()
        };
        roots.sort();
        roots.dedup();
        let base_probabilities = if beliefs.is_some() {
            vec![1. / 3.; 3]
        } else {
            actor.predict(&input.responses)?
        };
        if base_probabilities.len() != 3 {
            return Err(Error::Invalid("three public responses required"));
        }
        let probabilities: Vec<_> = (0..self.config.scenario_batches)
            .flat_map(|_| {
                base_probabilities
                    .iter()
                    .map(|p| p / self.config.scenario_batches as f64)
            })
            .collect();
        let mut candidates = Vec::new();
        for root in roots {
            let mut schedules: Vec<Option<Vec<String>>> = vec![None];
            if let Some(m) = live.filter(|m| m.remaining[0] == root) {
                schedules.insert(0, Some(m.remaining.clone()));
            } else if live.is_none() && self.config.horizon > 1 && self.config.second_moves > 0 {
                // Enumerate a bounded second move from a public response scenario.
                // Other scenarios still check legality and fall back independently.
                let model = game.model_world(player, self.config.model_seed)?;
                let mut choices = match beliefs {
                    Some(b) => match intentions {
                        Some(frame) => public_scenarios_with_intentions(
                            &model,
                            player,
                            b,
                            self.config.model_seed,
                            frame,
                        )?
                        .remove(0),
                        None => {
                            public_scenarios(&model, player, b, self.config.model_seed)?.remove(0)
                        }
                    },
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
                        .take(self.config.second_moves)
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
                for batch in 0..self.config.scenario_batches {
                    for scenario in 0..3 {
                        paths.push(simulate(
                            owner,
                            game,
                            player,
                            &root,
                            scenario,
                            self.config.horizon,
                            batch_seed(self.config.model_seed, batch),
                            schedule.as_deref(),
                            beliefs,
                            intentions.map(|frame| &frame.choices),
                            false,
                            self.config.route_selection,
                        )?);
                    }
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
        let (_, pre_filter) = selected_candidate(&appraisal, &candidates)?;
        let pre_filter_action = pre_filter.root.clone();
        let pre_filter_purpose = pre_filter.purpose;
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
                    let mut proposal =
                        fresh.propose_inner(owner, game, player, beliefs, intentions)?;
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
        let mut stage = if filtered.plans.len() == bounded.plans.len() {
            evaluated
        } else {
            actor.with_continuation(actor.prepare_estimated(&input, false)?, &filtered)?
        };
        let mut forecast_support = None;
        let mut fallback = false;
        if self.config.forecast_guard && self.config.horizon > 1 && live.is_none() {
            let (chosen, candidate) = selected_candidate(&stage, &candidates)?;
            if candidate.root != normal {
                if let Some(baseline) = candidates
                    .iter()
                    .position(|c| c.root == normal && c.ordinary)
                {
                    let baseline_id = format!("candidate-{baseline:03}");
                    let baseline_eligible = stage
                        .decision()
                        .scores
                        .iter()
                        .any(|s| s.id == baseline_id && s.eligible);
                    if baseline_eligible {
                        let support = crate::robustness::check(
                            &owner.config().actor,
                            &before.intent,
                            before.last_tick,
                            stage.context(),
                            stage.allowed(),
                            &filtered,
                            &format!("candidate-{chosen:03}"),
                            &baseline_id,
                            3,
                        )?;
                        if !support.supported {
                            let mut conservative = filtered.clone();
                            conservative.plans.retain(|p| p.action.id == baseline_id);
                            stage = actor.with_continuation(
                                actor.prepare_estimated(&input, false)?,
                                &conservative,
                            )?;
                            fallback = true;
                        }
                        forecast_support = Some(support);
                    }
                }
            }
        }
        let action = stage.decision().action_id.clone();
        let (selected, selected_candidate) = selected_candidate(&stage, &candidates)?;
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
                reason: if fallback {
                    "uncertain_forecast"
                } else if live.is_some()
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
                pre_filter_action,
                pre_filter_purpose,
                forecast_support,
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
        self.consent_inner(owner, game, player, beliefs, announced, offers, false)
    }

    /// Also consent to proposals with the same personal score as the outside.
    pub fn consent_allow_indifference<G: ArenaGame>(
        &self,
        owner: &Owner,
        game: &G,
        player: usize,
        beliefs: &Beliefs,
        announced: &[String],
        offers: &[crate::coordination::Offer],
    ) -> Result<Consent> {
        self.consent_inner(owner, game, player, beliefs, announced, offers, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn consent_inner<G: ArenaGame>(
        &self,
        owner: &Owner,
        game: &G,
        player: usize,
        beliefs: &Beliefs,
        announced: &[String],
        offers: &[crate::coordination::Offer],
        allow_indifference: bool,
    ) -> Result<Consent> {
        let scope = serde_json::to_string(&(
            &owner.config().actor.scope.game,
            &owner.config().actor.scope.episode,
        ))?;
        beliefs.validate(&scope, player, &game.behavior(player, player)?.alphabet)?;
        if player >= game.players()
            || announced.len() != game.players()
            || owner.config() != &self.owner
            || owner.state()?.last_tick != self.last_tick
            || owner_state_hash(owner)? != self.owner_state_sha256
        {
            return Err(Error::Invalid("current actor and intentions required"));
        }
        let input = public_input(game, player, beliefs, self.config.model_seed)?;
        let actor = owner.fork_model()?;
        let base = actor.prepare_estimated(&input, false)?;
        if !base.allowed().contains(&announced[player]) {
            return Err(Error::Invalid(
                "outside action must be allowed without agreement",
            ));
        }
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
                if offer
                    .roots
                    .keys()
                    .all(|p| *p == player || partners.contains(p))
                {
                    let mut first = public.clone();
                    first.extend(offer.roots.clone());
                    let mut scenarios =
                        public_scenarios(game, player, beliefs, self.config.model_seed)?;
                    for row in &mut scenarios {
                        for (p, action) in &first {
                            row[*p] = action.clone();
                        }
                    }
                    let mut signed = game.input_scenarios(player, &scenarios)?;
                    signed_context(&mut signed);
                    let candidate = actor.prepare_estimated(&signed, false)?;
                    if candidate.allowed().contains(root) {
                        intents.push((Some(offer.id.clone()), root.clone(), first));
                    }
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
            let mut paths = Vec::with_capacity(3 * self.config.scenario_batches);
            for batch in 0..self.config.scenario_batches {
                for scenario in 0..3 {
                    paths.push(simulate(
                        owner,
                        game,
                        player,
                        root,
                        scenario,
                        self.config.horizon,
                        batch_seed(self.config.model_seed, batch),
                        schedule,
                        Some(beliefs),
                        Some(first),
                        id.is_some(),
                        self.config.route_selection,
                    )?);
                }
            }
            let purpose = paths.iter().map(|p| p.purpose).sum::<f64>() / paths.len() as f64;
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
            plans: plans(
                &candidates,
                &vec![
                    1. / (3 * self.config.scenario_batches) as f64;
                    3 * self.config.scenario_batches
                ],
            ),
        };
        // Each root was admitted in its own signed or unsigned context. This
        // detached appraisal compares those candidates without committing it.
        let allowed_roots = intents.iter().map(|(_, root, _)| root.clone()).collect();
        let comparison = reflex_immediate::Agent::from_state(
            owner.config().actor.clone(),
            owner.state()?.intent.clone(),
            owner.state()?.last_tick,
        )?;
        let tier_appraisal =
            comparison.preview_continuation(base.context(), &forecast, &allowed_roots)?;
        let pre_filter_index = tier_appraisal
            .selected_plan()
            .and_then(|s| s.strip_prefix("candidate-"))
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or(Error::Invalid("selected plan identity required"))?;
        let pre_filter = candidates
            .get(pre_filter_index)
            .ok_or(Error::Invalid("selected plan outside candidate set"))?;
        if pre_filter.root != tier_appraisal.decision().action_id {
            return Err(Error::Invalid("selected plan root mismatch"));
        }
        let pre_filter_action = pre_filter.root.clone();
        let pre_filter_purpose = pre_filter.purpose;
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
            comparison.preview_continuation(base.context(), &bounded, &allowed_roots)?;
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
                if outside.eligible
                    && if allow_indifference {
                        s.score < outside.score - 1e-9
                    } else {
                        s.score <= outside.score + 1e-9
                    }
                {
                    return None;
                }
                Some((i, s.score))
            })
            .collect();
        let mut supports = BTreeMap::new();
        if self.config.forecast_guard && self.config.horizon > 1 && outside.eligible {
            let mut stable = Vec::new();
            for choice in choices {
                let support = crate::robustness::check(
                    &owner.config().actor,
                    &owner.state()?.intent,
                    owner.state()?.last_tick,
                    base.context(),
                    &allowed_roots,
                    &forecast,
                    &format!("candidate-{:03}", choice.0),
                    "candidate-000",
                    3,
                )?;
                if support.supported {
                    supports.insert(choice.0, support);
                    stable.push(choice);
                }
            }
            choices = stable;
        }
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
            let (selected, _) = selected_candidate(&stage, &candidates)?;
            if selected != i {
                return Err(Error::Invalid("agreement plan identity mismatch"));
            }
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
                        pre_filter_action: pre_filter_action.clone(),
                        pre_filter_purpose,
                        forecast_support: supports.remove(&i),
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
    for activity in input.purpose.activities.values_mut() {
        activity.condition = format!("{}|{context}", activity.condition);
    }
}
fn batch_seed(seed: u64, batch: usize) -> u64 {
    seed.wrapping_add(0x9e3779b97f4a7c15_u64.wrapping_mul(batch as u64))
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
    route_selection: bool,
) -> Result<Path> {
    let mut actor = owner.model()?;
    let mut public = beliefs.cloned();
    let mut game = world.model_world(player, seed.wrapping_add(scenario as u64))?;
    let mut route = crate::routes::RouteState {
        chosen: game.current_route(player).map(str::to_owned),
        switches: 0,
    };
    let mut follow_schedule = true;
    let mut effects = [0.; 22];
    let mut actions = Vec::new();
    let mut goals = Vec::new();
    for t in 0..horizon {
        if game.terminal() {
            break;
        }
        if route_selection && t > 0 {
            if let Some(frame) = game.route_input(player)? {
                let choice = crate::routes::choose(
                    &actor.config().actor,
                    &actor.state()?.intent,
                    &frame,
                    &route,
                    0.,
                    0.06,
                )?;
                if choice.state.chosen != route.chosen {
                    follow_schedule = false;
                }
                route = choice.state;
                game = game.with_route(player, route.chosen.as_deref())?;
            } else {
                game = game.with_route(player, None)?;
            }
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
        } else if follow_schedule {
            schedule.and_then(|s| s.get(t as usize)).map(String::as_str)
        } else {
            None
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

    #[allow(dead_code)]
    mod fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../reflex-owned/tests/support/mod.rs"
        ));
    }

    #[derive(Clone)]
    struct AgreementWorld {
        tick: u64,
    }
    impl ArenaGame for AgreementWorld {
        fn players(&self) -> usize {
            2
        }
        fn tick(&self) -> u64 {
            self.tick
        }
        fn terminal(&self) -> bool {
            self.tick >= 5
        }
        fn input(&self, _: usize) -> Result<reflex_owned::Input> {
            let mut input = fixture::input(self.tick);
            input.responses.legal = vec!["0".into(), "1".into(), "2".into()];
            input.responses.models = vec![vec![1. / 3.; 3]; 3];
            let mut act = input.observation.actions[0].clone();
            act.outcomes = vec![Outcome {
                p: 1.,
                objective: 0.9,
                needs: [0.; 5],
                values: [0.; 10],
                style: [0.; 5],
                cost: 0.,
            }];
            let mut next = act.clone();
            next.id = "next".into();
            next.outcomes[0].objective = 0.;
            input.observation.actions = vec![act.clone(), next.clone()];
            let mut binding = input.bindings["act"].clone();
            binding.method = "next".into();
            input.bindings.insert("next".into(), binding);
            input
                .conditional
                .insert("act".into(), vec![act.outcomes[0].clone(); 3]);
            input
                .conditional
                .insert("next".into(), vec![next.outcomes[0].clone(); 3]);
            let mut activity = input.purpose.activities["act"].clone();
            activity.kind = reflex_owned::progress::Kind::Attempt;
            input
                .purpose
                .activities
                .insert("act".into(), activity.clone());
            input.purpose.activities.insert("next".into(), activity);
            input.threatened = false;
            Ok(input)
        }
        fn behavior(&self, _: usize, _: usize) -> Result<crate::behavior::PublicBehavior> {
            Ok(crate::behavior::PublicBehavior {
                alphabet: vec!["act".into(), "next".into()],
                context: "public".into(),
                actions: BTreeMap::from([
                    ("act".into(), "act".into()),
                    ("next".into(), "next".into()),
                ]),
                suggestions: vec!["act".into(), "next".into(), "next".into()],
            })
        }
        fn input_scenarios(&self, player: usize, _: &[Vec<String>]) -> Result<reflex_owned::Input> {
            self.input(player)
        }
        fn model_world(&self, _: usize, _: u64) -> Result<Self> {
            Ok(self.clone())
        }
        fn advance(&self, choices: &[String], _: u64) -> Result<(Self, Vec<Feedback>)> {
            let mut feedback = fixture::feedback();
            feedback.observed = Some(if choices[0] == "act" && choices[1] == "act" {
                let mut v = [0.; 22];
                v[0] = 0.9;
                v
            } else {
                [0.; 22]
            });
            feedback.response = None;
            Ok((
                Self {
                    tick: self.tick + 1,
                },
                vec![feedback.clone(), feedback],
            ))
        }
        fn modeled_choices(&self, _: usize, _: &str) -> Result<Vec<String>> {
            Ok(vec!["next".into(), "next".into()])
        }
        fn purpose(&self, _: usize) -> f64 {
            0.
        }
        fn view(&self) -> serde_json::Value {
            serde_json::json!({"tick":self.tick})
        }
        fn partners(&self, p: usize) -> Vec<usize> {
            vec![1 - p]
        }
        fn offers(&self, _: &[String]) -> Result<Vec<crate::coordination::Offer>> {
            Ok(vec![])
        }
    }

    #[test]
    fn consent_evaluates_signed_context_without_reviving_unsigned_failures() {
        let mut config = fixture::config("person");
        config.contract = "arena-estimated-model-v1".into();
        config.predictor.names = vec!["0".into(), "1".into(), "2".into()];
        let world = AgreementWorld { tick: 3 };
        let input = world.input(0).unwrap();
        let mut progress = reflex_owned::progress::Progress::default();
        let stalled = reflex_owned::progress::PurposeFeedback {
            level: 0.,
            readiness: 0.,
            maintained: false,
            completed: false,
        };
        for _ in 0..3 {
            progress
                .observe("act", &input.purpose, Some(&stalled))
                .unwrap();
        }
        let state = LearnedState {
            last_tick: Some(2),
            progress,
            ..Default::default()
        };
        let mut owner = Owner::from_state(config.clone(), state).unwrap();
        let before = owner.state().unwrap().clone();
        assert!(!owner
            .prepare_estimated(&input, false)
            .unwrap()
            .allowed()
            .contains("act"));
        let mut signed = input.clone();
        signed_context(&mut signed);
        assert!(owner
            .prepare_estimated(&signed, false)
            .unwrap()
            .allowed()
            .contains("act"));
        assert_eq!(
            signed.bindings["act"].situation,
            signed.purpose.activities["act"].condition
        );
        let scope = serde_json::to_string(&(&config.actor.scope.game, &config.actor.scope.episode))
            .unwrap();
        let beliefs = Beliefs::new(scope, 0, vec!["act".into(), "next".into()]).unwrap();
        let guarded = Planner::new(
            &owner,
            Config {
                horizon: 2,
                second_moves: 0,
                scenario_batches: 2,
                forecast_guard: true,
                ..Config::default()
            },
        )
        .unwrap();
        let guarded_offer = crate::coordination::Offer {
            id: "guarded".into(),
            roots: BTreeMap::from([(0, "act".into()), (1, "act".into())]),
        };
        let guarded_consent = guarded
            .consent(
                &owner,
                &world,
                0,
                &beliefs,
                &["next".into(), "next".into()],
                &[guarded_offer],
            )
            .unwrap();
        assert_eq!(guarded_consent.ranked, vec!["guarded"]);
        assert!(
            guarded_consent.proposals["guarded"]
                .audit
                .forecast_support
                .as_ref()
                .unwrap()
                .supported
        );
        assert_eq!(owner.state().unwrap(), &before);
        let mut planner = Planner::new(
            &owner,
            Config {
                horizon: 1,
                second_moves: 0,
                ..Config::default()
            },
        )
        .unwrap();
        let offer = crate::coordination::Offer {
            id: "joint".into(),
            roots: BTreeMap::from([(0, "act".into()), (1, "act".into())]),
        };
        let outside = vec!["next".into(), "next".into()];
        assert!(planner
            .consent(&owner, &world, 0, &beliefs, &outside, &[])
            .unwrap()
            .ranked
            .is_empty());
        assert_eq!(owner.state().unwrap(), &before);
        assert!(planner
            .consent(
                &owner,
                &world,
                0,
                &beliefs,
                &["act".into(), "next".into()],
                std::slice::from_ref(&offer)
            )
            .is_err());
        let mut consent = planner
            .consent(
                &owner,
                &world,
                0,
                &beliefs,
                &outside,
                std::slice::from_ref(&offer),
            )
            .unwrap();
        assert_eq!(consent.ranked, vec!["joint"]);
        assert_eq!(owner.state().unwrap(), &before);
        let proposal = consent.proposals.remove("joint").unwrap();
        let (_, feedback) = world.advance(&["act".into(), "act".into()], 1).unwrap();
        planner
            .resolve(&mut owner, proposal, "act", &feedback[0], 0.)
            .unwrap();
        let learned = owner.state().unwrap();
        assert_eq!(learned.progress.failures.len(), 2);
        assert!(learned
            .progress
            .failures
            .contains(&before.progress.failures[0]));
        assert!(learned
            .experience
            .entries
            .iter()
            .any(|e| e.binding.situation == "condition|accepted-current-agreement"));
        assert!(!learned
            .progress
            .allowed(
                &world.input(0).unwrap().observation,
                &world.input(0).unwrap().purpose
            )
            .unwrap()
            .contains("act"));
        assert!(learned
            .progress
            .allowed(&signed.observation, &signed.purpose)
            .unwrap()
            .contains("act"));
    }

    #[derive(Clone, Debug, PartialEq)]
    struct RouteWorld {
        tick: u64,
        route: String,
    }

    impl ArenaGame for RouteWorld {
        fn players(&self) -> usize {
            1
        }
        fn tick(&self) -> u64 {
            self.tick
        }
        fn terminal(&self) -> bool {
            self.tick >= 2
        }
        fn input(&self, _: usize) -> Result<reflex_owned::Input> {
            let mut input = fixture::input(self.tick);
            let mut act = input.observation.actions[0].clone();
            let mut next = act.clone();
            next.id = "next".into();
            for outcome in &mut act.outcomes {
                outcome.objective = if self.route == "first" { 0.9 } else { 0.0 };
            }
            for outcome in &mut next.outcomes {
                outcome.objective = if self.route == "second" { 0.9 } else { 0.0 };
            }
            input.observation.actions = vec![act.clone(), next.clone()];
            let mut binding = input.bindings["act"].clone();
            binding.method = "next".into();
            input.bindings.insert("next".into(), binding);
            input
                .conditional
                .insert("act".into(), vec![act.outcomes[0].clone(); 3]);
            input
                .conditional
                .insert("next".into(), vec![next.outcomes[0].clone(); 3]);
            input
                .purpose
                .activities
                .insert("next".into(), input.purpose.activities["act"].clone());
            input.responses.legal = vec!["0".into(), "1".into(), "2".into()];
            input.responses.models = vec![vec![1. / 3.; 3]; 3];
            Ok(input)
        }
        fn behavior(&self, _: usize, _: usize) -> Result<crate::behavior::PublicBehavior> {
            unreachable!()
        }
        fn input_scenarios(&self, player: usize, _: &[Vec<String>]) -> Result<reflex_owned::Input> {
            self.input(player)
        }
        fn model_world(&self, _: usize, _: u64) -> Result<Self> {
            Ok(self.clone())
        }
        fn advance(&self, _: &[String], _: u64) -> Result<(Self, Vec<Feedback>)> {
            let mut feedback = fixture::feedback();
            feedback.response = Some("0".into());
            Ok((
                Self {
                    tick: self.tick + 1,
                    route: self.route.clone(),
                },
                vec![feedback],
            ))
        }
        fn modeled_choices(&self, _: usize, _: &str) -> Result<Vec<String>> {
            Ok(vec!["act".into()])
        }
        fn purpose(&self, _: usize) -> f64 {
            0.
        }
        fn view(&self) -> serde_json::Value {
            serde_json::json!({"tick": self.tick})
        }
        fn partners(&self, _: usize) -> Vec<usize> {
            vec![]
        }
        fn offers(&self, _: &[String]) -> Result<Vec<crate::coordination::Offer>> {
            Ok(vec![])
        }
        fn route_input(&self, _: usize) -> Result<Option<reflex_immediate::Observation>> {
            let mut obs = self.input(0)?.observation;
            obs.actions[0].id = "first".into();
            obs.actions[0].legal = self.tick == 0;
            obs.actions[1].id = "second".into();
            Ok(Some(obs))
        }
        fn with_route(&self, _: usize, route: Option<&str>) -> Result<Self> {
            Ok(Self {
                tick: self.tick,
                route: route.unwrap_or("first").into(),
            })
        }
        fn current_route(&self, _: usize) -> Option<&str> {
            Some(&self.route)
        }
    }

    #[test]
    fn model_changes_finished_route_and_stops_following_its_schedule() {
        let mut config = fixture::config("person");
        config.contract = "arena-estimated-model-v1".into();
        config.predictor.names = vec!["0".into(), "1".into(), "2".into()];
        let owner = Owner::new(config).unwrap();
        let before = owner.state().unwrap().clone();
        let world = RouteWorld {
            tick: 0,
            route: "first".into(),
        };
        let untouched = world.clone();
        let schedule = vec!["act".into(), "act".into()];
        let simulate_mode = |enabled| {
            simulate(
                &owner,
                &world,
                0,
                "act",
                0,
                2,
                1,
                Some(&schedule),
                None,
                None,
                false,
                enabled,
            )
            .unwrap()
        };
        let old = simulate_mode(false);
        let switched = simulate_mode(true);
        assert_eq!(old.actions, ["act", "act"]);
        assert_eq!(switched.actions, ["act", "next"]);
        assert_eq!(owner.state().unwrap(), &before);
        assert_eq!(world, untouched);
    }

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

    #[test]
    fn ordinary_tail_uses_the_modeled_owners_current_immediate_decision() {
        let game = crate::games::Hagetaka::new(7);
        let mut config = fixture::config("a");
        config.contract = "arena-estimated-model-v1".into();
        config.predictor.names = vec!["0".into(), "1".into(), "2".into()];
        let owner = Owner::new(config).unwrap();
        let root = game.input(0).unwrap().observation.actions[0].id.clone();
        let path = simulate(
            &owner, &game, 0, &root, 0, 3, 701, None, None, None, false, false,
        )
        .unwrap();
        let model_world = game.model_world(0, 701).unwrap();
        let mut modeled = owner.model().unwrap();
        let input = model_world.input(0).unwrap();
        let stage = modeled.prepare_root_estimated(&input, &root).unwrap();
        let mut actions = model_world.modeled_choices(0, "0").unwrap();
        actions[0] = root.clone();
        let (next, feedback) = model_world.advance(&actions, 701).unwrap();
        let ticket = modeled.commit(stage, &root).unwrap();
        modeled.resolve(ticket, &feedback[0]).unwrap();
        let immediate = modeled.prepare_estimated(&next.input(0).unwrap()).unwrap();
        assert_eq!(path.actions[0], root);
        assert_eq!(path.actions[1], immediate.decision().action_id);
    }
}
