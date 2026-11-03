//! Bounded public-model planning. Separate research profile, not fixed H32.
use crate::{
    behavior::Beliefs, observe_public, public_input, public_input_with_intentions,
    public_scenarios, public_scenarios_with_intentions, social, ArenaGame, Intentions,
};
use reflex_immediate::{Action, Continuation, ContinuationPlan, Outcome, ReturnUnits};
use reflex_owned::{Error, Feedback, Input, LearnedState, Owner, OwnerConfig, Result, Stage};
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
    terminal: bool,
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

/// Apply social preference after the owner's existing immediate judgment.
pub fn prepare_social(
    owner: &Owner,
    input: &Input,
    bindings: &BTreeMap<String, social::Binding>,
    memory: Option<&social::Memory>,
) -> Result<Stage> {
    let stage = owner.prepare_estimated(input, false)?;
    let Some(memory) = memory else {
        return Ok(stage);
    };
    memory.validate_for(&owner.config().actor, owner.state()?.last_tick)?;
    let bias = social_bias(memory, stage.context(), bindings)?;
    let purpose = stage
        .context()
        .actions
        .iter()
        .map(|a| {
            (
                a.id.clone(),
                a.outcomes
                    .iter()
                    .map(|o| o.p * o.objective)
                    .sum::<f64>()
                    .clamp(-1., 1.),
            )
        })
        .collect();
    let selected = social::choose(
        stage.decision(),
        &stage.decision().action_id,
        &purpose,
        &bias,
        0.15,
    )?;
    if selected == stage.decision().action_id {
        return Ok(stage);
    }
    let action = stage
        .context()
        .actions
        .iter()
        .find(|a| a.id == selected)
        .ok_or(Error::Invalid("selected social root required"))?
        .clone();
    let forecast = Continuation {
        owner: owner.config().actor.clone(),
        state: owner.state()?.intent.clone(),
        tick: input.observation.tick,
        target: owner.config().purpose_target.clone(),
        horizon: 1,
        max_regret: 2.,
        units: ReturnUnits {
            unit: "arena-turn".into(),
            effects: [1.; 22],
            root_terms: 1.,
        },
        plans: vec![ContinuationPlan {
            root: selected,
            action,
            purpose: 0.,
            correction: None,
        }],
    };
    owner.with_continuation(stage, &forecast)
}
fn social_bias(
    memory: &social::Memory,
    observation: &reflex_immediate::Observation,
    bindings: &BTreeMap<String, social::Binding>,
) -> Result<BTreeMap<String, f64>> {
    Ok(memory
        .scores(observation, bindings)?
        .into_iter()
        .map(|(id, score)| (id, score.adjustment))
        .collect())
}
fn plan_bias(forecast: &Continuation, roots: &BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    forecast
        .plans
        .iter()
        .map(|p| (p.action.id.clone(), *roots.get(&p.root).unwrap_or(&0.)))
        .collect()
}
fn plan_purpose(forecast: &Continuation) -> BTreeMap<String, f64> {
    forecast
        .plans
        .iter()
        .map(|p| (p.action.id.clone(), p.purpose))
        .collect()
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
        if save.version != "arena-planner-v7"
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
            if !m.value.is_finite()
                || !(-1.0..=1.0).contains(&m.value)
                || !p.last_tick.is_some_and(|t| {
                    t.checked_add(p.config.horizon as u64)
                        .is_some_and(|limit| m.deadline > t && m.deadline <= limit)
                })
            {
                return Err(Error::Invalid("invalid milestone"));
            }
        }
        Ok(p)
    }
    pub fn checkpoint(&self) -> Result<Vec<u8>> {
        let save = PlannerSave {
            version: "arena-planner-v7".into(),
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
        self.propose_inner(owner, game, player, None, None, None)
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
        self.propose_inner(owner, game, player, Some(beliefs), None, None)
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
        self.propose_inner(owner, game, player, Some(beliefs), Some(intentions), None)
    }
    pub fn propose_social<G: ArenaGame>(
        &self,
        owner: &Owner,
        game: &G,
        player: usize,
        beliefs: Option<&Beliefs>,
        intentions: Option<&Intentions>,
        memory: Option<&social::Memory>,
    ) -> Result<Proposal> {
        if intentions.is_some() && beliefs.is_none() {
            return Err(Error::Invalid("intentions require public beliefs"));
        }
        if let Some(b) = beliefs {
            let scope = serde_json::to_string(&(
                &owner.config().actor.scope.game,
                &owner.config().actor.scope.episode,
            ))?;
            b.validate(&scope, player, &game.behavior(player, player)?.alphabet)?;
        }
        self.propose_inner(owner, game, player, beliefs, intentions, memory)
    }
    fn propose_inner<G: ArenaGame>(
        &self,
        owner: &Owner,
        game: &G,
        player: usize,
        beliefs: Option<&Beliefs>,
        intentions: Option<&Intentions>,
        memory: Option<&social::Memory>,
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
        let (input, bindings) = if memory.is_some() {
            let frame =
                crate::decision_frame(game, player, beliefs, self.config.model_seed, intentions)?;
            (frame.input, frame.social)
        } else {
            let input = match beliefs {
                Some(b) => match intentions {
                    Some(frame) => public_input_with_intentions(
                        game,
                        player,
                        b,
                        self.config.model_seed,
                        frame,
                    )?,
                    None => public_input(game, player, b, self.config.model_seed)?,
                },
                None => game.input(player)?,
            };
            (input, BTreeMap::new())
        };
        if input.observation.tick != game.tick() || input.responses.legal != ["0", "1", "2"] {
            return Err(Error::Invalid(
                "matching clock and public response alphabet required",
            ));
        }
        let actor = owner.fork_model()?;
        let stage = prepare_social(&actor, &input, &bindings, memory)?;
        let root_bias = memory
            .map(|m| social_bias(m, stage.context(), &bindings))
            .transpose()?
            .unwrap_or_default();
        let normal = stage.decision().action_id.clone();
        let allowed = stage.allowed().clone();
        let live = self
            .milestone
            .as_ref()
            .filter(|m| game.tick() < m.deadline && game.purpose(player) < m.value - 1e-12);
        let roots: Vec<String> = allowed.iter().cloned().collect();
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
            if self.config.horizon > 1 && self.config.second_moves > 0 {
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
                            memory,
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
        let mut retained = live.cloned();
        let mut reachable = std::collections::BTreeSet::new();
        if let Some(m) = live {
            let eligible: std::collections::BTreeSet<usize> = evaluated
                .decision()
                .scores
                .iter()
                .filter(|s| s.eligible)
                .filter_map(|s| s.id.strip_prefix("candidate-")?.parse().ok())
                .collect();
            let remaining = (m.deadline - game.tick()) as usize;
            reachable = eligible
                .iter()
                .copied()
                .filter(|&i| arrival(&candidates[i], m.value, remaining) == Arrival::Reached)
                .collect();
            if !reachable.is_empty() {
                let permitted = goal_options(&candidates, &eligible, &reachable);
                filtered.plans.retain(|p| {
                    p.action
                        .id
                        .strip_prefix("candidate-")
                        .and_then(|s| s.parse::<usize>().ok())
                        .is_some_and(|i| permitted.contains(&i))
                });
            } else if eligible
                .iter()
                .all(|&i| arrival(&candidates[i], m.value, remaining) == Arrival::Impossible)
            {
                retained = None;
            }
        }
        let mut stage = if filtered.plans.len() == bounded.plans.len() {
            evaluated
        } else {
            actor.with_continuation(actor.prepare_estimated(&input, false)?, &filtered)?
        };
        let selected_before = stage
            .selected_plan()
            .ok_or(Error::Invalid("selected plan required"))?
            .to_owned();
        let bias = plan_bias(&filtered, &root_bias);
        let purpose = plan_purpose(&filtered);
        let mut chosen_id = if memory.is_some() {
            social::choose(stage.decision(), &selected_before, &purpose, &bias, 0.15)?
        } else {
            selected_before.clone()
        };
        let mut forecast_support = None;
        let mut fallback = false;
        if self.config.forecast_guard && self.config.horizon > 1 && reachable.is_empty() {
            let chosen = chosen_id
                .strip_prefix("candidate-")
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or(Error::Invalid("selected candidate required"))?;
            if candidates[chosen].root != normal {
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
                        let support = crate::robustness::check_with_appraisal(
                            &owner.config().actor,
                            &before.intent,
                            before.last_tick,
                            stage.context(),
                            stage.allowed(),
                            &filtered,
                            &chosen_id,
                            &baseline_id,
                            3,
                            &bias,
                        )?;
                        if !support.supported {
                            chosen_id = baseline_id;
                            fallback = true;
                        }
                        forecast_support = Some(support);
                    }
                }
            }
        }
        if chosen_id != selected_before {
            let mut single = filtered.clone();
            single.plans.retain(|p| p.action.id == chosen_id);
            stage = actor.with_continuation(actor.prepare_estimated(&input, false)?, &single)?;
        }
        let action = stage.decision().action_id.clone();
        let (selected, selected_candidate) = selected_candidate(&stage, &candidates)?;
        let baseline = candidates
            .iter()
            .position(|c| c.root == normal && c.ordinary);
        let switching = !reachable.is_empty() && !reachable.contains(&selected);
        let next_milestone = if live.is_some() && !switching {
            retained
        } else {
            baseline.and_then(|baseline| {
                new_milestone(
                    &candidates,
                    selected,
                    baseline,
                    &probabilities,
                    game.purpose(player),
                    game.tick(),
                    self.config.horizon,
                )
            })
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
                    && !switching
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
        if next
            .as_ref()
            .is_some_and(|m| actual_goal >= m.value - 1e-12 || proposal.tick + 1 >= m.deadline)
        {
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
        self.consent_inner(owner, game, player, beliefs, announced, offers, false, None)
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
        self.consent_inner(owner, game, player, beliefs, announced, offers, true, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn consent_social<G: ArenaGame>(
        &self,
        owner: &Owner,
        game: &G,
        player: usize,
        beliefs: &Beliefs,
        announced: &[String],
        offers: &[crate::coordination::Offer],
        memory: Option<&social::Memory>,
        allow_indifference: bool,
    ) -> Result<Consent> {
        self.consent_inner(
            owner,
            game,
            player,
            beliefs,
            announced,
            offers,
            allow_indifference,
            memory,
        )
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
        memory: Option<&social::Memory>,
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
        let actor = owner.fork_model()?;
        let (input, outside_bias) = if let Some(memory) = memory {
            let mut scenarios = public_scenarios(game, player, beliefs, self.config.model_seed)?;
            for row in &mut scenarios {
                for (p, action) in &public {
                    row[*p] = action.clone();
                }
            }
            let frame = game.decision_frame(player, &scenarios)?;
            memory.validate_for(&owner.config().actor, owner.state()?.last_tick)?;
            let scores = social_bias(memory, &frame.input.observation, &frame.social)?;
            let bias = *scores.get(&announced[player]).unwrap_or(&0.);
            (frame.input, bias)
        } else {
            (
                public_input(game, player, beliefs, self.config.model_seed)?,
                0.,
            )
        };
        let base = actor.prepare_estimated(&input, false)?;
        if !base.allowed().contains(&announced[player]) {
            return Err(Error::Invalid(
                "outside action must be allowed without agreement",
            ));
        }
        // A candidate retains the input and appraisal from the exact first-step commitments
        // used to admit it; the cached signed input is reused if this offer executes.
        let mut intents = vec![(
            None,
            announced[player].clone(),
            public.clone(),
            None,
            outside_bias,
        )];
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
                    let (mut signed, bias) = if let Some(memory) = memory {
                        let frame = game.decision_frame(player, &scenarios)?;
                        let scores = social_bias(memory, &frame.input.observation, &frame.social)?;
                        let bias = *scores.get(root).unwrap_or(&0.);
                        (frame.input, bias)
                    } else {
                        (game.input_scenarios(player, &scenarios)?, 0.)
                    };
                    signed_context(&mut signed);
                    let candidate = actor.prepare_estimated(&signed, false)?;
                    if candidate.allowed().contains(root) {
                        intents.push((
                            Some(offer.id.clone()),
                            root.clone(),
                            first,
                            memory.map(|_| signed),
                            bias,
                        ));
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
        for (id, root, first, _, _) in &intents {
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
                        None,
                        Some(beliefs),
                        Some(first),
                        id.is_some(),
                        self.config.route_selection,
                        memory,
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
        let allowed_roots = intents
            .iter()
            .map(|(_, root, _, _, _)| root.clone())
            .collect();
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
        let live = self
            .milestone
            .as_ref()
            .filter(|m| game.tick() < m.deadline && game.purpose(player) < m.value - 1e-12);
        let eligible: std::collections::BTreeSet<usize> = appraisal
            .decision()
            .scores
            .iter()
            .filter(|s| s.eligible)
            .filter_map(|s| s.id.strip_prefix("candidate-")?.parse().ok())
            .collect();
        let reachable: std::collections::BTreeSet<usize> = live
            .map(|m| {
                eligible
                    .iter()
                    .copied()
                    .filter(|&i| {
                        arrival(&candidates[i], m.value, (m.deadline - game.tick()) as usize)
                            == Arrival::Reached
                    })
                    .collect()
            })
            .unwrap_or_default();
        let permitted = goal_options(&candidates, &eligible, &reachable);
        // Offers are only a subset of legal roots; they cannot prove the goal impossible.
        let consent_milestone = live.cloned();
        let outside = tier_appraisal
            .decision()
            .scores
            .iter()
            .find(|s| s.id == "candidate-000")
            .ok_or(Error::Invalid("outside score required"))?;
        let biases: BTreeMap<String, f64> = intents
            .iter()
            .enumerate()
            .map(|(i, (_, _, _, _, bias))| (format!("candidate-{i:03}"), *bias))
            .collect();
        let has_bias = biases.values().any(|v| *v != 0.);
        let best_purpose = appraisal
            .decision()
            .scores
            .iter()
            .filter(|s| s.eligible)
            .filter_map(|s| s.id.strip_prefix("candidate-")?.parse::<usize>().ok())
            .map(|i| candidates[i].purpose)
            .fold(f64::NEG_INFINITY, f64::max);
        let baseline_index = appraisal
            .selected_plan()
            .and_then(|s| s.strip_prefix("candidate-"))
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or(Error::Invalid("selected plan required"))?;
        let purpose_floor = candidates[baseline_index].purpose.min(best_purpose - 0.15);
        let outside_score = outside.score + biases.get("candidate-000").copied().unwrap_or(0.);
        let mut choices: Vec<_> = appraisal
            .decision()
            .scores
            .iter()
            .filter_map(|s| {
                let i = s.id.strip_prefix("candidate-")?.parse::<usize>().ok()?;
                if i == 0
                    || !s.eligible
                    || (!reachable.is_empty() && !permitted.contains(&i))
                    || (has_bias && candidates[i].purpose < purpose_floor - 1e-9)
                {
                    return None;
                }
                // Respect personality tiers first, then compare within the same tier.
                let score = s.score + biases.get(&s.id).copied().unwrap_or(0.);
                if outside.eligible
                    && if allow_indifference {
                        score < outside_score - 1e-9
                    } else {
                        score <= outside_score + 1e-9
                    }
                {
                    return None;
                }
                Some((i, score))
            })
            .collect();
        let mut supports = BTreeMap::new();
        if self.config.forecast_guard && self.config.horizon > 1 && outside.eligible {
            let mut stable = Vec::new();
            for choice in choices {
                let support = crate::robustness::check_with_appraisal(
                    &owner.config().actor,
                    &owner.state()?.intent,
                    owner.state()?.last_tick,
                    base.context(),
                    &allowed_roots,
                    &forecast,
                    &format!("candidate-{:03}", choice.0),
                    "candidate-000",
                    3,
                    &biases,
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
            let (id, root, first, cached, _) = &intents[i];
            let id = id.as_ref().unwrap();
            let signed = if let Some(input) = cached {
                input.clone()
            } else {
                let mut scenarios =
                    public_scenarios(game, player, beliefs, self.config.model_seed)?;
                for row in &mut scenarios {
                    for (p, a) in first {
                        row[*p] = a.clone();
                    }
                }
                let mut input = game.input_scenarios(player, &scenarios)?;
                signed_context(&mut input);
                input
            };
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
            let milestone = if live.is_some() && (reachable.is_empty() || reachable.contains(&i)) {
                consent_milestone.clone()
            } else {
                new_milestone(
                    &candidates,
                    i,
                    0,
                    &vec![1. / candidates[0].paths.len() as f64; candidates[0].paths.len()],
                    game.purpose(player),
                    game.tick(),
                    self.config.horizon,
                )
            };
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
    memory: Option<&social::Memory>,
) -> Result<Path> {
    let mut social_memory = memory.cloned();
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
        let mut sampled = match &public {
            Some(b) => Some(public_scenarios(&game, player, b, seed)?),
            None if social_memory.is_some() => Some(
                (0..3)
                    .map(|i| game.modeled_choices(player, &i.to_string()))
                    .collect::<Result<Vec<_>>>()?,
            ),
            None => None,
        };
        if t == 0 {
            if let (Some(rows), Some(first)) = (&mut sampled, first) {
                for row in rows {
                    for (p, a) in first {
                        row[*p] = a.clone();
                    }
                }
            }
        }
        let (mut input, bindings) = if social_memory.is_some() {
            let frame = game.decision_frame(player, sampled.as_ref().expect("social scenarios"))?;
            (frame.input, frame.social)
        } else {
            let input = match &sampled {
                Some(choices) => game.input_scenarios(player, choices)?,
                None => game.input(player)?,
            };
            (input, BTreeMap::new())
        };
        if t == 0 && signed && first.is_some() {
            signed_context(&mut input);
        }
        if input.observation.tick != game.tick() || input.responses.legal != ["0", "1", "2"] {
            return Err(Error::Invalid("matching model observation required"));
        }
        let normal = actor.prepare_estimated(&input)?;
        let normal = if let Some(m) = &social_memory {
            let bias = social_bias(m, normal.context(), &bindings)?;
            let purpose = normal
                .context()
                .actions
                .iter()
                .map(|a| {
                    (
                        a.id.clone(),
                        a.outcomes
                            .iter()
                            .map(|o| o.p * o.objective)
                            .sum::<f64>()
                            .clamp(-1., 1.),
                    )
                })
                .collect();
            let selected = social::choose(
                normal.decision(),
                &normal.decision().action_id,
                &purpose,
                &bias,
                0.15,
            )?;
            if selected != normal.decision().action_id {
                actor.prepare_root_estimated(&input, &selected)?
            } else {
                normal
            }
        } else {
            normal
        };
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
        let world_seed = seed
            .wrapping_add((scenario as u64) * 37)
            .wrapping_add(t as u64);
        let (next, feedback) = game.advance(&choices, world_seed)?;
        if let Some(m) = &mut social_memory {
            let events = game.social_events(&next, &choices, world_seed)?;
            if events.len() != game.players() {
                return Err(Error::Invalid("social events require all players"));
            }
            m.observe_batch(&events[player], game.tick())?;
        }
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
        terminal: game.terminal(),
    })
}
#[derive(PartialEq, Eq)]
enum Arrival {
    Reached,
    Impossible,
    Unknown,
}
fn arrival(candidate: &Candidate, target: f64, remaining: usize) -> Arrival {
    let mut reached = 0;
    let mut missed = 0;
    for path in &candidate.paths {
        if path.goals.is_empty() {
            continue;
        }
        if path
            .goals
            .iter()
            .take(remaining)
            .any(|v| *v >= target - 1e-12)
        {
            reached += 1;
        } else if path.goals.len() >= remaining || path.terminal {
            missed += 1;
        }
    }
    if !candidate.paths.is_empty() && reached == candidate.paths.len() {
        Arrival::Reached
    } else if !candidate.paths.is_empty() && missed == candidate.paths.len() {
        Arrival::Impossible
    } else {
        Arrival::Unknown
    }
}

fn goal_options(
    candidates: &[Candidate],
    eligible: &std::collections::BTreeSet<usize>,
    reachable: &std::collections::BTreeSet<usize>,
) -> std::collections::BTreeSet<usize> {
    let best = reachable
        .iter()
        .map(|&i| candidates[i].purpose)
        .fold(f64::NEG_INFINITY, f64::max);
    eligible
        .iter()
        .copied()
        .filter(|i| reachable.contains(i) || candidates[*i].purpose > best + 1e-12)
        .collect()
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
    baseline: usize,
    probabilities: &[f64],
    origin: f64,
    tick: u64,
    horizon: u32,
) -> Option<Milestone> {
    let chosen = candidates.get(selected)?;
    let normal = candidates.get(baseline)?;
    if selected == baseline {
        return None;
    }
    for i in 1..horizon as usize {
        if chosen
            .paths
            .iter()
            .chain(&normal.paths)
            .any(|p| p.goals.len() <= i)
        {
            break;
        }
        let floor = chosen
            .paths
            .iter()
            .map(|p| p.goals[i])
            .fold(f64::INFINITY, f64::min);
        let average: f64 = normal
            .paths
            .iter()
            .zip(probabilities)
            .map(|(p, w)| p.goals[i] * w)
            .sum();
        if floor > origin + 1e-12 && floor > average + 1e-12 {
            return Some(Milestone {
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

    #[derive(Clone)]
    struct MixedWorld {
        tick: u64,
        value: f64,
    }
    impl ArenaGame for MixedWorld {
        fn players(&self) -> usize {
            2
        }
        fn tick(&self) -> u64 {
            self.tick
        }
        fn terminal(&self) -> bool {
            self.tick >= 4
        }
        fn input(&self, player: usize) -> Result<reflex_owned::Input> {
            AgreementWorld { tick: self.tick }.input(player)
        }
        fn behavior(&self, viewer: usize, actor: usize) -> Result<crate::behavior::PublicBehavior> {
            AgreementWorld { tick: self.tick }.behavior(viewer, actor)
        }
        fn input_scenarios(
            &self,
            player: usize,
            choices: &[Vec<String>],
        ) -> Result<reflex_owned::Input> {
            AgreementWorld { tick: self.tick }.input_scenarios(player, choices)
        }
        fn partners(&self, player: usize) -> Vec<usize> {
            vec![1 - player]
        }
        fn offers(&self, _: &[String]) -> Result<Vec<crate::coordination::Offer>> {
            Ok(vec![])
        }
        fn model_world(&self, _: usize, _: u64) -> Result<Self> {
            Ok(self.clone())
        }
        fn advance(&self, choices: &[String], seed: u64) -> Result<(Self, Vec<Feedback>)> {
            let (_, feedback) = AgreementWorld { tick: self.tick }.advance(choices, seed)?;
            Ok((
                Self {
                    tick: self.tick + 1,
                    value: if self.tick == 0 && choices[1] == "act" {
                        0.6
                    } else {
                        0.
                    },
                },
                feedback,
            ))
        }
        fn modeled_choices(&self, _: usize, response: &str) -> Result<Vec<String>> {
            Ok(vec![
                "next".into(),
                if self.tick == 0 && response == "0" {
                    "act"
                } else {
                    "next"
                }
                .into(),
            ])
        }
        fn purpose(&self, _: usize) -> f64 {
            self.value
        }
        fn view(&self) -> serde_json::Value {
            serde_json::json!({"tick": self.tick})
        }
    }
    #[test]
    fn mixed_arrivals_keep_goal_until_next_actual_reassessment() {
        let mut owner = Owner::new(fixture_owner_config()).unwrap();
        let mut planner = Planner::new(
            &owner,
            Config {
                horizon: 2,
                second_moves: 0,
                ..Config::default()
            },
        )
        .unwrap();
        planner.milestone = Some(Milestone {
            value: 0.5,
            deadline: 2,
        });
        let world = MixedWorld { tick: 0, value: 0. };
        let proposal = planner.propose(&owner, &world, 0).unwrap();
        assert_eq!(proposal.audit.reason, "keep");
        let action = proposal.action().to_owned();
        let (next, feedback) = world.advance(&[action.clone(), "next".into()], 0).unwrap();
        planner
            .resolve(&mut owner, proposal, &action, &feedback[0], next.value)
            .unwrap();
        assert!(planner.milestone().is_some());
        let proposal = planner.propose(&owner, &next, 0).unwrap();
        assert_eq!(proposal.audit.reason, "withdraw");
        assert!(proposal.audit.milestone.is_none());
    }

    #[test]
    fn milestone_checkpoint_unknown_reach_deadline_and_withdrawal() {
        let mut config = fixture::config("person");
        config.contract = "arena-estimated-model-v1".into();
        config.predictor.names = vec!["0".into(), "1".into(), "2".into()];
        let mut owner = Owner::new(config).unwrap();
        let mut planner = Planner::new(
            &owner,
            Config {
                horizon: 1,
                second_moves: 0,
                ..Config::default()
            },
        )
        .unwrap();
        planner.milestone = Some(Milestone {
            value: 0.5,
            deadline: 2,
        });
        let bytes = planner.checkpoint().unwrap();
        assert!(Planner::restore(&owner, &bytes).is_err());
        let mut old: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        old["version"] = "arena-planner-v6".into();
        assert!(Planner::restore(&owner, &serde_json::to_vec(&old).unwrap()).is_err());
        let before = owner.state().unwrap().clone();
        let world = AgreementWorld { tick: 0 };
        let proposal = planner.propose(&owner, &world, 0).unwrap();
        assert_eq!(proposal.audit.reason, "keep");
        assert_eq!(owner.state().unwrap(), &before);
        let action = proposal.action().to_owned();
        let (_, feedback) = world.advance(&[action.clone(), "next".into()], 0).unwrap();
        planner
            .resolve(&mut owner, proposal, &action, &feedback[0], 0.)
            .unwrap();
        assert!(planner.milestone().is_some());
        let world = AgreementWorld { tick: 1 };
        let proposal = planner.propose(&owner, &world, 0).unwrap();
        assert_eq!(proposal.audit.reason, "withdraw");
        let action = proposal.action().to_owned();
        let (_, feedback) = world.advance(&[action.clone(), "next".into()], 0).unwrap();
        planner
            .resolve(&mut owner, proposal, &action, &feedback[0], 0.)
            .unwrap();
        assert!(planner.milestone().is_none());

        let mut owner = Owner::new(fixture_owner_config()).unwrap();
        let mut planner = Planner::new(
            &owner,
            Config {
                horizon: 1,
                second_moves: 0,
                ..Config::default()
            },
        )
        .unwrap();
        planner.milestone = Some(Milestone {
            value: 0.5,
            deadline: 3,
        });
        let world = AgreementWorld { tick: 0 };
        let proposal = planner.propose(&owner, &world, 0).unwrap();
        let action = proposal.action().to_owned();
        let (_, feedback) = world.advance(&[action.clone(), "next".into()], 0).unwrap();
        planner
            .resolve(&mut owner, proposal, &action, &feedback[0], 0.5)
            .unwrap();
        assert!(planner.milestone().is_none());
    }
    #[test]
    fn milestone_restore_checks_clock_and_horizon() {
        let mut owner = Owner::new(fixture_owner_config()).unwrap();
        let mut initial = Planner::new(&owner, Config::default()).unwrap();
        let world = AgreementWorld { tick: 0 };
        let proposal = initial.propose(&owner, &world, 0).unwrap();
        let action = proposal.action().to_owned();
        let (_, feedback) = world.advance(&[action.clone(), "next".into()], 0).unwrap();
        initial
            .resolve(&mut owner, proposal, &action, &feedback[0], 0.)
            .unwrap();
        let mut planner = Planner::new(
            &owner,
            Config {
                horizon: 2,
                ..Config::default()
            },
        )
        .unwrap();
        for (deadline, valid) in [
            (0, false),
            (1, true),
            (2, true),
            (3, false),
            (u64::MAX, false),
        ] {
            planner.milestone = Some(Milestone {
                value: 0.5,
                deadline,
            });
            assert_eq!(
                Planner::restore(&owner, &planner.checkpoint().unwrap()).is_ok(),
                valid
            );
        }
    }
    #[test]
    fn inactive_goal_filter_does_not_bypass_forecast_guard() {
        let owner = Owner::new(fixture_owner_config()).unwrap();
        let config = Config {
            horizon: 3,
            second_moves: 0,
            scenario_batches: 3,
            max_regret: 1.,
            forecast_guard: true,
            ..Config::default()
        };
        let plain = Planner::new(&owner, config).unwrap();
        let mut exercised = [false; 2];
        for seed in 0..30 {
            let world = crate::games::Hagetaka::new(seed);
            let baseline = plain.propose(&owner, &world, 0).unwrap();
            if baseline.audit.forecast_support.is_none() {
                continue;
            }
            for (index, deadline) in [3, 4].into_iter().enumerate() {
                let mut guarded = plain.clone();
                guarded.milestone = Some(Milestone {
                    value: 1.,
                    deadline,
                });
                let proposal = guarded.propose(&owner, &world, 0).unwrap();
                if proposal.audit.reason == "keep" || proposal.audit.reason == "withdraw" {
                    assert!(proposal.audit.forecast_support.is_some());
                    assert_eq!(proposal.action(), baseline.action());
                    exercised[index] = true;
                }
            }
            if exercised.iter().all(|x| *x) {
                break;
            }
        }
        assert!(
            exercised.iter().all(|x| *x),
            "fixture must exercise both inactive goal states"
        );
    }
    fn fixture_owner_config() -> OwnerConfig {
        let mut config = fixture::config("person");
        config.contract = "arena-estimated-model-v1".into();
        config.predictor.names = vec!["0".into(), "1".into(), "2".into()];
        config
    }

    #[derive(Clone)]
    struct SocialWorld {
        tick: u64,
    }
    impl ArenaGame for SocialWorld {
        fn players(&self) -> usize {
            2
        }
        fn tick(&self) -> u64 {
            self.tick
        }
        fn terminal(&self) -> bool {
            self.tick >= 4
        }
        fn input(&self, player: usize) -> Result<Input> {
            let mut input = AgreementWorld { tick: self.tick }.input(player)?;
            input.observation.actions[0].target = Some("other".into());
            for a in &mut input.observation.actions {
                for o in &mut a.outcomes {
                    o.objective = 0.;
                }
            }
            for rows in input.conditional.values_mut() {
                for o in rows {
                    o.objective = 0.;
                }
            }
            Ok(input)
        }
        fn behavior(&self, viewer: usize, actor: usize) -> Result<crate::behavior::PublicBehavior> {
            AgreementWorld { tick: self.tick }.behavior(viewer, actor)
        }
        fn input_scenarios(&self, p: usize, _: &[Vec<String>]) -> Result<Input> {
            self.input(p)
        }
        fn model_world(&self, _: usize, _: u64) -> Result<Self> {
            Ok(self.clone())
        }
        fn advance(&self, choices: &[String], seed: u64) -> Result<(Self, Vec<Feedback>)> {
            let (_, mut feedback) = AgreementWorld { tick: self.tick }.advance(choices, seed)?;
            for f in &mut feedback {
                f.observed = Some([0.; 22]);
            }
            Ok((
                Self {
                    tick: self.tick + 1,
                },
                feedback,
            ))
        }
        fn modeled_choices(&self, _: usize, _: &str) -> Result<Vec<String>> {
            Ok(vec!["next".into(), "next".into()])
        }
        fn purpose(&self, _: usize) -> f64 {
            0.
        }
        fn view(&self) -> serde_json::Value {
            serde_json::json!({"tick": self.tick})
        }
        fn partners(&self, p: usize) -> Vec<usize> {
            vec![1 - p]
        }
        fn offers(&self, _: &[String]) -> Result<Vec<crate::coordination::Offer>> {
            Ok(vec![])
        }
        fn social_bindings(&self, _: usize) -> Result<BTreeMap<String, social::Binding>> {
            Ok(BTreeMap::from([(
                "act".into(),
                social::Binding {
                    target: "other".into(),
                    support: 1.,
                    distress: 0.,
                },
            )]))
        }
        fn social_events(&self, _: &Self, _: &[String], _: u64) -> Result<Vec<Vec<social::Event>>> {
            let events = (0..12)
                .map(|i| social::Event {
                    target: "other".into(),
                    event_id: format!("{}-{i}", self.tick),
                    tick: self.tick,
                    benefit: -1.,
                    agency: true,
                    confidence: 1.,
                })
                .collect();
            Ok(vec![events, vec![]])
        }
    }
    #[derive(Clone)]
    struct ConditionalSocialWorld {
        tick: u64,
        frames: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }
    impl ArenaGame for ConditionalSocialWorld {
        fn players(&self) -> usize {
            2
        }
        fn tick(&self) -> u64 {
            self.tick
        }
        fn terminal(&self) -> bool {
            self.tick >= 4
        }
        fn input(&self, p: usize) -> Result<Input> {
            SocialWorld { tick: self.tick }.input(p)
        }
        fn behavior(&self, v: usize, p: usize) -> Result<crate::behavior::PublicBehavior> {
            SocialWorld { tick: self.tick }.behavior(v, p)
        }
        fn input_scenarios(&self, p: usize, _: &[Vec<String>]) -> Result<Input> {
            self.input(p)
        }
        fn decision_frame(&self, p: usize, rows: &[Vec<String>]) -> Result<crate::DecisionFrame> {
            self.frames
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let support =
                rows.iter().filter(|row| row[1] == "act").count() as f64 / rows.len() as f64;
            Ok(crate::DecisionFrame {
                input: self.input(p)?,
                social: BTreeMap::from([(
                    "act".into(),
                    social::Binding {
                        target: "other".into(),
                        support,
                        distress: 0.,
                    },
                )]),
            })
        }
        fn model_world(&self, _: usize, _: u64) -> Result<Self> {
            Ok(self.clone())
        }
        fn advance(&self, choices: &[String], seed: u64) -> Result<(Self, Vec<Feedback>)> {
            let (_, feedback) = SocialWorld { tick: self.tick }.advance(choices, seed)?;
            Ok((
                Self {
                    tick: self.tick + 1,
                    frames: self.frames.clone(),
                },
                feedback,
            ))
        }
        fn modeled_choices(&self, _: usize, _: &str) -> Result<Vec<String>> {
            Ok(vec!["act".into(), "next".into()])
        }
        fn purpose(&self, _: usize) -> f64 {
            0.
        }
        fn view(&self) -> serde_json::Value {
            serde_json::json!({"tick": self.tick})
        }
        fn partners(&self, p: usize) -> Vec<usize> {
            vec![1 - p]
        }
        fn offers(&self, _: &[String]) -> Result<Vec<crate::coordination::Offer>> {
            Ok(vec![])
        }
    }
    #[test]
    fn consent_uses_candidate_specific_effects_for_same_root_and_reuses_frame() {
        let mut cfg = fixture_owner_config();
        cfg.actor.policy.priority = reflex_immediate::Priority::Finite;
        cfg.actor.values[8] = 1.;
        let mut owner = Owner::new(cfg).unwrap();
        let first = SocialWorld { tick: 0 };
        let stage = owner
            .prepare_estimated(&first.input(0).unwrap(), false)
            .unwrap();
        let action = stage.decision().action_id.clone();
        let (_, feedback) = first.advance(&[action.clone(), "next".into()], 0).unwrap();
        let ticket = owner.commit(stage, &action).unwrap();
        owner.resolve(ticket, &feedback[0]).unwrap();
        let mut memory = social::Memory::new(&owner.config().actor).unwrap();
        memory
            .observe_batch(
                &first
                    .social_events(&SocialWorld { tick: 1 }, &[], 0)
                    .unwrap()[0],
                0,
            )
            .unwrap();
        let scope = serde_json::to_string(&(
            &owner.config().actor.scope.game,
            &owner.config().actor.scope.episode,
        ))
        .unwrap();
        let beliefs = Beliefs::new(scope, 0, vec!["act".into(), "next".into()]).unwrap();
        let world = ConditionalSocialWorld {
            tick: 1,
            frames: std::sync::Arc::default(),
        };
        let planner = Planner::new(
            &owner,
            Config {
                horizon: 1,
                second_moves: 0,
                ..Config::default()
            },
        )
        .unwrap();
        let offer = |id: &str, partner: &str| crate::coordination::Offer {
            id: id.into(),
            roots: BTreeMap::from([(0, "act".into()), (1, partner.into())]),
        };
        let saved = memory.checkpoint(&world.view()).unwrap();
        let consent = planner
            .consent_social(
                &owner,
                &world,
                0,
                &beliefs,
                &["act".into(), "next".into()],
                &[offer("delivered", "act"), offer("undelivered", "next")],
                Some(&memory),
                true,
            )
            .unwrap();
        assert_eq!(consent.ranked, vec!["undelivered"]);
        assert_eq!(world.frames.load(std::sync::atomic::Ordering::Relaxed), 12);
        assert_eq!(memory.checkpoint(&world.view()).unwrap(), saved);
        let intentions = |partner: &str| Intentions {
            tick: 1,
            choices: BTreeMap::from([(1, partner.into())]),
        };
        let delivered = planner
            .propose_social(
                &owner,
                &world,
                0,
                Some(&beliefs),
                Some(&intentions("act")),
                Some(&memory),
            )
            .unwrap();
        let undelivered = planner
            .propose_social(
                &owner,
                &world,
                0,
                Some(&beliefs),
                Some(&intentions("next")),
                Some(&memory),
            )
            .unwrap();
        assert_eq!(delivered.action(), "next");
        assert_eq!(undelivered.action(), "act");
        let modeled = simulate(
            &owner,
            &world,
            0,
            "act",
            0,
            2,
            701,
            None,
            None,
            None,
            false,
            false,
            Some(&memory),
        )
        .unwrap();
        assert_eq!(modeled.actions[1], "act"); // The modeled recipient does not reciprocate.
        assert_eq!(memory.checkpoint(&world.view()).unwrap(), saved);
    }

    #[test]
    fn simulated_social_learning_is_detached_and_restores_next_choice() {
        let mut config = fixture_owner_config();
        config.actor.policy.priority = reflex_immediate::Priority::Finite;
        config.actor.values[8] = 1.;
        let mut owner = Owner::new(config).unwrap();
        let world = SocialWorld { tick: 0 };
        let mut memory = social::Memory::new(&owner.config().actor).unwrap();
        let before = memory.checkpoint(&world.view()).unwrap();
        let plain = simulate(
            &owner, &world, 0, "act", 0, 2, 1, None, None, None, false, false, None,
        )
        .unwrap();
        let modeled = simulate(
            &owner,
            &world,
            0,
            "act",
            0,
            2,
            1,
            None,
            None,
            None,
            false,
            false,
            Some(&memory),
        )
        .unwrap();
        assert_eq!(plain.actions[1], "act");
        assert_eq!(modeled.actions[1], "next");
        assert_eq!(memory.checkpoint(&world.view()).unwrap(), before);
        let planner = Planner::new(
            &owner,
            Config {
                horizon: 2,
                second_moves: 0,
                ..Config::default()
            },
        )
        .unwrap();
        let _ = planner
            .propose_social(&owner, &world, 0, None, None, Some(&memory))
            .unwrap();
        assert_eq!(memory.checkpoint(&world.view()).unwrap(), before);
        let stage = prepare_social(
            &owner,
            &world.input(0).unwrap(),
            &world.social_bindings(0).unwrap(),
            Some(&memory),
        )
        .unwrap();
        let action = stage.decision().action_id.clone();
        let choices = vec![action.clone(), "next".into()];
        let (next, feedback) = world.advance(&choices, 1).unwrap();
        let ticket = owner.commit(stage, &action).unwrap();
        owner.resolve(ticket, &feedback[0]).unwrap();
        memory
            .observe_batch(&world.social_events(&next, &choices, 1).unwrap()[0], 0)
            .unwrap();
        let saved = memory.checkpoint(&next.view()).unwrap();
        let restored = social::Memory::restore(
            &owner.config().actor,
            &next.view(),
            owner.state().unwrap().last_tick,
            &saved,
        )
        .unwrap();
        let bindings = next.social_bindings(0).unwrap();
        let a = prepare_social(&owner, &next.input(0).unwrap(), &bindings, Some(&memory)).unwrap();
        let b =
            prepare_social(&owner, &next.input(0).unwrap(), &bindings, Some(&restored)).unwrap();
        assert_eq!(a.decision().action_id, "next");
        assert_eq!(a.decision().action_id, b.decision().action_id);
        let current = Planner::new(
            &owner,
            Config {
                horizon: 1,
                second_moves: 0,
                max_regret: 1.,
                ..Config::default()
            },
        )
        .unwrap();
        let proposal = current
            .propose_social(&owner, &next, 0, None, None, Some(&memory))
            .unwrap();
        assert_eq!(proposal.action(), "next");
        let scope = serde_json::to_string(&(
            &owner.config().actor.scope.game,
            &owner.config().actor.scope.episode,
        ))
        .unwrap();
        let beliefs = Beliefs::new(scope, 0, vec!["act".into(), "next".into()]).unwrap();
        let offer = crate::coordination::Offer {
            id: "joint".into(),
            roots: BTreeMap::from([(0, "act".into()), (1, "next".into())]),
        };
        let announced = vec!["next".into(), "next".into()];
        let neutral = current
            .consent_social(
                &owner,
                &next,
                0,
                &beliefs,
                &announced,
                std::slice::from_ref(&offer),
                None,
                true,
            )
            .unwrap();
        assert_eq!(neutral.ranked, vec!["joint"]);
        let social = current
            .consent_social(
                &owner,
                &next,
                0,
                &beliefs,
                &announced,
                &[offer],
                Some(&memory),
                true,
            )
            .unwrap();
        assert!(social.ranked.is_empty());
        let mut restricted = next.input(0).unwrap();
        restricted
            .observation
            .actions
            .iter_mut()
            .find(|a| a.id == "next")
            .unwrap()
            .legal = false;
        let forced = prepare_social(&owner, &restricted, &bindings, Some(&memory)).unwrap();
        assert_eq!(forced.decision().action_id, "act");
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
        planner.milestone = Some(Milestone {
            value: 0.5,
            deadline: 4,
        });
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
        assert_eq!(proposal.audit.milestone.as_ref().unwrap().value, 0.5);
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
                None,
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
    fn arrival_checks_each_branch_and_preserves_unknown_truncation() {
        let outcome = Outcome {
            p: 1.,
            objective: 0.,
            needs: [0.; 5],
            values: [0.; 10],
            style: [0.; 5],
            cost: 0.,
        };
        let path = |actions: &[&str], goals: Vec<f64>, terminal| Path {
            outcome: outcome.clone(),
            purpose: 0.,
            actions: actions.iter().map(|s| s.to_string()).collect(),
            goals,
            terminal,
        };
        let mut rows = [
            candidate("excluded", 0., true),
            candidate("first", 0., true),
            candidate("excluded", 0., true),
            candidate("second", 0., true),
        ];
        rows[1].paths = vec![
            path(&["first", "a"], vec![0., 0.6], false),
            path(&["first", "b"], vec![0.1, 0.7], false),
        ];
        rows[3].paths = vec![path(&["second"], vec![0.1], false)];
        assert!(arrival(&rows[1], 0.6, 2) == Arrival::Reached);
        let mut mixed = candidate("mixed", 0., true);
        assert!(arrival(&mixed, 0.6, 2) == Arrival::Unknown);
        mixed.paths = vec![path(&[], vec![], true)];
        assert!(arrival(&mixed, 0.6, 2) == Arrival::Unknown);
        mixed.paths = vec![
            path(&["a", "b"], vec![0., 0.6], false),
            path(&["a", "c"], vec![0., 0.1], false),
        ];
        assert!(arrival(&mixed, 0.6, 2) == Arrival::Unknown);
        mixed.paths[0] = path(&["a"], vec![0.], false);
        assert!(arrival(&mixed, 0.6, 2) == Arrival::Unknown);
        mixed.paths[0].terminal = true;
        assert!(arrival(&mixed, 0.6, 2) == Arrival::Impossible);
        assert!(arrival(&rows[3], 0.6, 2) == Arrival::Unknown);
        rows[3].paths[0].terminal = true;
        assert!(arrival(&rows[3], 0.6, 2) == Arrival::Impossible);
        assert!(arrival(&rows[1], 0.6, 1) == Arrival::Impossible);
        let eligible = BTreeSet::from([1, 3]);
        let retained: Vec<_> = eligible
            .into_iter()
            .filter(|&i| arrival(&rows[i], 0.6, 2) == Arrival::Reached)
            .collect();
        assert_eq!(retained, vec![1]);
        rows[1].purpose = 0.4;
        rows[3].purpose = 0.6;
        assert_eq!(
            goal_options(&rows, &BTreeSet::from([1, 3]), &BTreeSet::from([1])),
            BTreeSet::from([1, 3])
        );
        assert_eq!(
            goal_options(&rows, &BTreeSet::from([1]), &BTreeSet::from([1])),
            BTreeSet::from([1])
        );
    }

    #[test]
    fn milestone_compares_same_time_normal_and_accepts_diverging_paths() {
        let outcome = Outcome {
            p: 1.,
            objective: 0.,
            needs: [0.; 5],
            values: [0.; 10],
            style: [0.; 5],
            cost: 0.,
        };
        let path = |actions: &[&str], goals: Vec<f64>| Path {
            outcome: outcome.clone(),
            purpose: 0.,
            actions: actions.iter().map(|s| s.to_string()).collect(),
            goals,
            terminal: false,
        };
        let mut normal = candidate("normal", 0.9, true);
        normal.paths = vec![
            path(&["normal", "a"], vec![0., 0.2]),
            path(&["normal", "b"], vec![0., 0.4]),
        ];
        let mut chosen = candidate("chosen", 0.4, true);
        chosen.paths = vec![
            path(&["chosen", "a"], vec![0., 0.5]),
            path(&["chosen", "b"], vec![0., 0.6]),
        ];
        let rows = vec![normal, chosen];
        let m = new_milestone(&rows, 1, 0, &[0.5, 0.5], 0., 10, 2).unwrap();
        assert_eq!((m.value, m.deadline), (0.5, 12));
        assert!(new_milestone(&rows, 0, 0, &[0.5, 0.5], 0., 10, 2).is_none());
        let mut same_root = rows;
        same_root[1].root = same_root[0].root.clone();
        assert!(new_milestone(&same_root, 1, 0, &[0.5, 0.5], 0., 10, 2).is_some());
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
            &owner, &game, 0, &root, 0, 3, 701, None, None, None, false, false, None,
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
