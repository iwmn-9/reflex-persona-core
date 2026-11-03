//! Headless simultaneous games with isolated personalities and factual learning.
use reflex_arena::routes::{self, RouteState};
use reflex_arena::{
    behavior::{Beliefs, ModelSet},
    games::{Combat, Encounters, Hagetaka},
    observe_public,
    planner::{Config as Search, Planner},
    public_input, public_input_with_intentions, ArenaGame, Intentions,
};
use reflex_immediate::{Config, Policy, Priority, Scope};
use reflex_owned::{prediction, Owner, OwnerConfig};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

const NAMES: [&str; 4] = ["挑戦志向", "安定志向", "協力志向", "自己主張"];
enum Selection {
    Planned(Box<reflex_arena::planner::Proposal>),
    Reflex(Box<reflex_owned::Stage>),
}
const TRAITS: [[f64; 5]; 4] = [
    [0.9, 0.55, 0.6, 0.4, 0.25],
    [0.25, 0.9, 0.3, 0.55, 0.85],
    [0.65, 0.8, 0.7, 0.95, 0.35],
    [0.6, 0.35, 0.85, 0.15, 0.25],
];
fn config(game: &str, seed: u64, player: usize, profile: usize) -> OwnerConfig {
    let mut values = [0.; 10];
    match profile {
        0 => {
            values[3] = 0.75;
            values[0] = 0.5;
        }
        1 => {
            values[5] = 0.8;
            values[6] = 0.4;
        }
        2 => {
            values[8] = 0.85;
            values[9] = 0.5;
        }
        _ => {
            values[4] = 0.7;
            values[3] = 0.45;
        }
    }
    OwnerConfig {
        actor: Config {
            scope: Scope {
                game: game.into(),
                episode: seed.to_string(),
                npc: player.to_string(),
            },
            seed,
            personality: TRAITS[profile],
            values,
            objective: "Pursue the public game objective with personal values".into(),
            policy: Policy {
                priority: Priority::Personal { rigidity: 0.5 },
                residual: [0.; 8],
            },
        },
        predictor: prediction::Definition {
            model_id: format!("{game}-public-behavior-v1"),
            factors: vec![],
            names: vec!["0".into(), "1".into(), "2".into()],
            window: 128,
        },
        contract: format!("arena-{game}-estimated-model-v11"),
        purpose_target: "game-purpose".into(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run<G: ArenaGame>(
    mut world: G,
    game: &str,
    seed: u64,
    horizon: u32,
    second_moves: usize,
    max_regret: f64,
    scenario_batches: usize,
    mode: &str,
    individual: bool,
    coordinate: bool,
    share_intentions: bool,
    frozen_reference: bool,
    planning_protocol: &str,
    route_selection: bool,
    forecast_guard: bool,
    opponent_models: ModelSet,
    output: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let focal: Vec<usize> = if game == "combat" {
        (0..world.players())
            .filter(|i| i / 3 == seed as usize % 2)
            .collect()
    } else {
        vec![0]
    };
    let planning: Vec<bool> = (0..world.players())
        .map(|i| mode == "planned" || mode == "challenger" && focal.contains(&i))
        .collect();
    let modified: Vec<bool> = planning.iter().map(|&p| !frozen_reference || p).collect();
    let sharing: Vec<bool> = modified.iter().map(|&m| share_intentions && m).collect();
    let profiles: Vec<_> = (0..world.players())
        .map(|i| (i + seed as usize % 4) % 4)
        .collect();
    let configs: Vec<_> = profiles
        .iter()
        .enumerate()
        .map(|(i, &p)| config(game, seed, i, p))
        .collect();
    let mut owners: Vec<_> = configs
        .iter()
        .cloned()
        .map(Owner::new)
        .collect::<Result<_, _>>()?;
    let mut planners: Vec<_> = owners
        .iter()
        .enumerate()
        .map(|(i, o)| {
            Planner::new(
                o,
                Search {
                    horizon: if planning[i] { horizon } else { 1 },
                    second_moves: if modified[i] { second_moves } else { 8 },
                    max_regret: if modified[i] { max_regret } else { 0.15 },
                    scenario_batches: if modified[i] { scenario_batches } else { 1 },
                    route_selection: route_selection && modified[i],
                    forecast_guard: forecast_guard && modified[i],
                    ..Search::default()
                },
            )
        })
        .collect::<Result<_, _>>()?;
    let scopes: Vec<_> = configs
        .iter()
        .map(|c| serde_json::to_string(&(&c.actor.scope.game, &c.actor.scope.episode)))
        .collect::<Result<_, _>>()?;
    let alphabets: Vec<_> = (0..world.players())
        .map(|i| world.behavior(i, i).map(|f| f.alphabet))
        .collect::<Result<_, _>>()?;
    let mut beliefs: Vec<_> = (0..world.players())
        .map(|i| {
            Beliefs::new_with_models(
                scopes[i].clone(),
                i,
                alphabets[i].clone(),
                if modified[i] {
                    opponent_models
                } else {
                    ModelSet::Adapter
                },
            )
        })
        .collect::<Result<_, _>>()?;
    let mut route_states = vec![RouteState::default(); world.players()];
    let mut trace = vec![];
    let mut checks = 0;
    let mut counters = BTreeMap::<String, usize>::new();
    let mut simulations = 0;
    while !world.terminal() {
        if world.tick() > 256 {
            return Err("arena failed to terminate".into());
        }
        let before = world.view();
        let tick = world.tick();
        let mut decision_world = world.clone();
        let mut active_planners = planners.clone();
        let mut next_routes = route_states.clone();
        let mut route_audits = vec![json!(null); world.players()];
        for i in 0..world.players() {
            if route_selection && modified[i] {
                if let Some(frame) = world.route_input(i)? {
                    let choice = routes::choose(
                        &configs[i].actor,
                        &owners[i].state()?.intent,
                        &frame,
                        &route_states[i],
                        0.,
                        0.06,
                    )?;
                    if choice.state.chosen != route_states[i].chosen {
                        active_planners[i] =
                            Planner::new(&owners[i], planners[i].config().clone())?;
                    }
                    next_routes[i] = choice.state;
                    decision_world =
                        decision_world.with_route(i, next_routes[i].chosen.as_deref())?;
                    route_audits[i] = serde_json::to_value(choice.audit)?;
                } else {
                    decision_world = decision_world.with_route(i, None)?;
                }
            }
        }
        let mut next_owners = owners
            .iter()
            .map(Owner::fork_model)
            .collect::<Result<Vec<_>, _>>()?;
        let mut next_planners = active_planners.clone();
        let mut next_beliefs = beliefs.clone();
        let mut choices = vec![];
        let mut audits = vec![];
        let mut intention_records = vec![json!(null); decision_world.players()];
        let mut inputs = vec![];
        for (i, belief) in beliefs.iter().enumerate() {
            inputs.push(if individual {
                public_input(&decision_world, i, belief, 701)?
            } else {
                decision_world.input(i)?
            });
        }
        let mut selections = vec![];
        for i in 0..decision_world.players() {
            if planning[i] && (!sharing[i] || planning_protocol == "independent") {
                let proposal = if individual {
                    active_planners[i].propose_with_beliefs(
                        &owners[i],
                        &decision_world,
                        i,
                        &beliefs[i],
                    )?
                } else {
                    active_planners[i].propose(&owners[i], &decision_world, i)?
                };
                choices.push(proposal.action().to_string());
                simulations += proposal.audit.simulated_nodes;
                audits.push(serde_json::to_value(&proposal.audit)?);
                selections.push(Selection::Planned(Box::new(proposal)));
            } else {
                let s = next_owners[i].prepare_estimated(&inputs[i], false)?;
                choices.push(s.decision().action_id.clone());
                selections.push(Selection::Reflex(Box::new(s)));
                audits.push(json!({"reason":"reflex","milestone":null,"simulated_nodes":0}));
            }
        }
        let mut counteroffers = vec![];
        if sharing.iter().any(|s| *s) {
            // A revision relying on partners becomes a joint offer. Until every
            // participant signs, the independent announcements remain executable.
            let initial = choices.clone();
            for offset in 0..decision_world.players() {
                let i = (offset + tick as usize) % decision_world.players();
                if !sharing[i] || planning[i] && planning_protocol == "independent" {
                    continue;
                }
                let intentions = Intentions {
                    tick,
                    choices: decision_world
                        .partners(i)
                        .into_iter()
                        .map(|p| {
                            initial
                                .get(p)
                                .map(|a| (p, a.clone()))
                                .ok_or("invalid communication partner")
                        })
                        .collect::<Result<_, _>>()?,
                };
                inputs[i] = public_input_with_intentions(
                    &decision_world,
                    i,
                    &beliefs[i],
                    701,
                    &intentions,
                )?;
                let proposed = if planning[i] {
                    let proposal = active_planners[i].propose_with_intentions(
                        &owners[i],
                        &decision_world,
                        i,
                        &beliefs[i],
                        &intentions,
                    )?;
                    let action = proposal.action().to_string();
                    simulations += proposal.audit.simulated_nodes;
                    if intentions.choices.is_empty() {
                        choices[i] = action.clone();
                        audits[i] = serde_json::to_value(&proposal.audit)?;
                        selections[i] = Selection::Planned(Box::new(proposal));
                    }
                    action
                } else {
                    let stage = next_owners[i].prepare_estimated(&inputs[i], false)?;
                    stage.decision().action_id.clone()
                };
                if !intentions.choices.is_empty() && proposed != initial[i] {
                    let mut roots = intentions.choices.clone();
                    roots.insert(i, proposed.clone());
                    counteroffers.push(reflex_arena::coordination::Offer {
                        id: format!("intent:{i}"),
                        roots,
                    });
                }
                intention_records[i] = json!({"announced_action":initial[i],"proposed_action":proposed,"received_intentions":intentions.choices});
            }
        }
        let mut agreements = vec![];
        let mut incompatible_offers = vec![];
        if coordinate {
            let mut offers = decision_world.offers(&choices)?;
            offers.extend(counteroffers);
            let mut compatible = Vec::new();
            for offer in offers {
                if offer
                    .roots
                    .keys()
                    .any(|&p| sharing.get(p).copied().unwrap_or(false))
                    && !reflex_arena::offer_compatible(&decision_world, &offer, &choices)?
                {
                    incompatible_offers.push(offer.id);
                } else {
                    compatible.push(offer);
                }
            }
            let offers = compatible;
            if !offers.is_empty() {
                let mut consents = Vec::new();
                let mut ballots = Vec::new();
                for i in 0..decision_world.players() {
                    let consent = if sharing[i] {
                        active_planners[i].consent_allow_indifference(
                            &owners[i],
                            &decision_world,
                            i,
                            &beliefs[i],
                            &choices,
                            &offers,
                        )?
                    } else {
                        active_planners[i].consent(
                            &owners[i],
                            &decision_world,
                            i,
                            &beliefs[i],
                            &choices,
                            &offers,
                        )?
                    };
                    simulations += consent.simulated_nodes;
                    ballots.push(reflex_arena::coordination::Ballot {
                        actor: i,
                        scope: scopes[i].clone(),
                        tick,
                        outside: choices[i].clone(),
                        ranked: consent.ranked.clone(),
                    });
                    consents.push(consent);
                }
                let legal = inputs
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        (
                            i,
                            x.observation
                                .actions
                                .iter()
                                .filter(|a| a.legal && !a.known_failure)
                                .map(|a| a.id.clone())
                                .collect(),
                        )
                    })
                    .collect();
                let order: Vec<_> = (0..decision_world.players())
                    .map(|i| (i + tick as usize) % decision_world.players())
                    .collect();
                let agreement = reflex_arena::coordination::negotiate_checked(
                    &offers,
                    &ballots,
                    &order,
                    &legal,
                    &scopes[0],
                    tick,
                    |offer, projected| {
                        if offer
                            .roots
                            .keys()
                            .any(|&p| sharing.get(p).copied().unwrap_or(false))
                        {
                            reflex_arena::offer_compatible(&decision_world, offer, projected)
                        } else {
                            Ok(true)
                        }
                    },
                )?;
                for id in &agreement.accepted {
                    let offer = offers.iter().find(|o| &o.id == id).unwrap();
                    for &i in offer.roots.keys() {
                        let p = consents[i]
                            .proposals
                            .remove(id)
                            .ok_or("signed proposal required")?;
                        audits[i] = serde_json::to_value(&p.audit)?;
                        selections[i] = Selection::Planned(Box::new(p));
                    }
                }
                choices = agreement.choices;
                agreements = agreement.accepted;
            }
        }
        for audit in &audits {
            if let Some(reason) = audit["reason"].as_str() {
                *counters.entry(reason.into()).or_default() += 1;
            }
        }
        let (next, mut feedback) =
            decision_world.advance(&choices, seed.wrapping_mul(1009).wrapping_add(tick))?;
        if individual {
            for i in 0..decision_world.players() {
                observe_public(&decision_world, i, &mut next_beliefs[i], &choices)?;
                feedback[i].response = None;
            }
        }
        for (i, selection) in selections.into_iter().enumerate() {
            match selection {
                Selection::Planned(p) => next_planners[i].resolve(
                    &mut next_owners[i],
                    *p,
                    &choices[i],
                    &feedback[i],
                    next.purpose(i),
                )?,
                Selection::Reflex(s) => {
                    let ticket = next_owners[i].commit(*s, &choices[i])?;
                    next_owners[i].resolve(ticket, &feedback[i])?;
                    next_planners[i] =
                        Planner::new(&next_owners[i], next_planners[i].config().clone())?;
                }
            }
        }
        let mut participants = vec![];
        for i in 0..decision_world.players() {
            if next_owners[i].config() != &configs[i] {
                return Err("personality changed".into());
            }
            if !inputs[i]
                .observation
                .actions
                .iter()
                .any(|a| a.id == choices[i] && a.legal && !a.known_failure)
            {
                return Err("illegal factual choice".into());
            }
            let save = next_owners[i].checkpoint(&next.view())?;
            let restored = Owner::restore(&configs[i], &next.view(), &save)?;
            if restored.state()? != next_owners[i].state()? {
                return Err("restore changed learning".into());
            }
            next_owners[i] = restored;
            {
                let saved_plan = next_planners[i].checkpoint()?;
                next_planners[i] = Planner::restore(&next_owners[i], &saved_plan)?;
            }
            next_beliefs[i] = Beliefs::restore_with_models(
                &next_beliefs[i].checkpoint()?,
                &scopes[i],
                i,
                &alphabets[i],
                beliefs[i].model_set(),
            )?;
            let route_save = next_routes[i].checkpoint(&configs[i].actor, tick)?;
            next_routes[i] = RouteState::restore(&configs[i].actor, tick, &route_save)?;
            participants.push(json!({"player":i,"profile":profiles[i],"name":NAMES[profiles[i]],
                "action":choices[i],"purpose_before":decision_world.purpose(i),"purpose_after":next.purpose(i),
                "needs":inputs[i].observation.needs,"learning_observations":tick+1,
                "prediction_observations":next_owners[i].state()?.prediction.observations,
                "public_behavior_observations":next_beliefs[i].observations(),
                "planning":audits[i],"intention_input":intention_records[i],"route":route_audits[i],"route_state":next_routes[i]}));
            checks += 1;
        }
        trace.push(json!({"tick":tick,"before":before,"after":next.view(),"actors":participants,"agreements":agreements,"incompatible_offers":incompatible_offers}));
        world = next;
        owners = next_owners;
        planners = next_planners;
        beliefs = next_beliefs;
        route_states = next_routes;
        eprintln!("{game} {mode} tick={} nodes={simulations}", world.tick());
    }
    let uses_independent = (0..world.players())
        .any(|i| sharing[i] && planning[i] && planning_protocol == "independent");
    let uses_counterproposal = (0..world.players())
        .any(|i| sharing[i] && (!planning[i] || planning_protocol == "counterproposal"));
    let intention_protocol = match (uses_independent, uses_counterproposal) {
        (true, true) => "mixed-independent-counterproposal-v1",
        (true, false) => "independent-planning-consent-v1",
        (false, true) => "mutual-counterproposal-v1",
        (false, false) => "none",
    };
    let report = json!({"complete":true,"game":game,"seed":seed,"horizon":horizon,"second_moves":second_moves,"max_regret":max_regret,"scenario_batches":scenario_batches,"mode":mode,"individual_beliefs":individual,"coordination":coordinate,"intent_sharing":share_intentions,"planning_protocol":planning_protocol,"route_selection":route_selection,"forecast_guard":forecast_guard,"opponent_models":opponent_models,"intention_protocol":intention_protocol,"frozen_reference":frozen_reference,
        "agreement_compatibility":"public-joint-constraints-v1",
        "controllers":(0..world.players()).map(|i|json!({"planned":planning[i],"opponent_models":beliefs[i].model_set(),"settings":planners[i].config(),"intent_sharing":sharing[i],"joint_constraints":sharing[i],"planning_protocol":if planning[i] && (!sharing[i] || planning_protocol == "independent") {"independent"} else if sharing[i] {"counterproposal"} else {"reflex"}})).collect::<Vec<_>>(),
        "profiles":profiles,"profile_names":NAMES,"focal_players":focal,"ticks":world.tick(),"factual_decisions":checks,
        "simulated_nodes":simulations,"plan_events":counters,"seconds":started.elapsed().as_secs_f64(),
        "purpose":(0..world.players()).map(|i|world.purpose(i)).collect::<Vec<_>>(),
        "final":world.view(),"trace":trace,
        "scope":"Native integration and behavioral replay; not human-level or generalization certification"});
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(output, serde_json::to_vec(&report)?)?;
    println!(
        "{}",
        json!({"game":game,"mode":mode,"ticks":world.tick(),"decisions":checks,
        "seconds":started.elapsed().as_secs_f64(),"complete":true})
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.len().is_multiple_of(2) {
        return Err("named flag/value pairs required".into());
    }
    let options: BTreeMap<_, _> = args
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| (p[0].as_str(), p[1].as_str()))
        .collect();
    for key in options.keys() {
        if ![
            "--game",
            "--seed",
            "--horizon",
            "--second-moves",
            "--max-regret",
            "--scenario-batches",
            "--turns",
            "--mode",
            "--out",
            "--map",
            "--goal",
            "--beliefs",
            "--coordination",
            "--intent-sharing",
            "--reference",
            "--planning-protocol",
            "--routes",
            "--forecast-guard",
            "--opponent-models",
        ]
        .contains(key)
        {
            return Err("unknown flag".into());
        }
    }
    let game = *options.get("--game").unwrap_or(&"combat");
    let seed = options
        .get("--seed")
        .unwrap_or(&"20261103")
        .parse::<u64>()?;
    let horizon = options.get("--horizon").unwrap_or(&"6").parse::<u32>()?;
    let second_moves = options
        .get("--second-moves")
        .unwrap_or(&"8")
        .parse::<usize>()?;
    if second_moves > 8 {
        return Err("second-moves must be in 0..=8".into());
    }
    let max_regret = options
        .get("--max-regret")
        .unwrap_or(&"0.15")
        .parse::<f64>()?;
    if !max_regret.is_finite() || !(0.0..=1.0).contains(&max_regret) {
        return Err("max-regret must be in 0..=1".into());
    }
    let scenario_batches = options
        .get("--scenario-batches")
        .unwrap_or(&"1")
        .parse::<usize>()?;
    if !(1..=4).contains(&scenario_batches) {
        return Err("scenario-batches must be in 1..=4".into());
    }
    let turns = options.get("--turns").unwrap_or(&"32").parse::<u64>()?;
    let mode = *options.get("--mode").unwrap_or(&"reflex");
    let belief_mode = *options.get("--beliefs").unwrap_or(&"individual");
    if !["joint", "individual"].contains(&belief_mode) {
        return Err("known belief mode required".into());
    }
    let individual = belief_mode == "individual";
    let coordination =
        *options
            .get("--coordination")
            .unwrap_or(&if individual { "consent" } else { "none" });
    if !["consent", "none"].contains(&coordination) || coordination == "consent" && !individual {
        return Err("consent needs individual public models".into());
    }
    let coordinate = coordination == "consent";
    let intent_sharing = *options.get("--intent-sharing").unwrap_or(&"none");
    if !["none", "proposals"].contains(&intent_sharing)
        || intent_sharing == "proposals" && (!individual || !coordinate)
    {
        return Err("intention proposals need individual public models and consent".into());
    }
    let share_intentions = intent_sharing == "proposals";
    let reference = *options.get("--reference").unwrap_or(&"frozen");
    if !["frozen", "matched"].contains(&reference) {
        return Err("known reference required".into());
    }
    let frozen_reference = reference == "frozen";
    let planning_protocol = *options
        .get("--planning-protocol")
        .unwrap_or(&"counterproposal");
    if !["counterproposal", "independent"].contains(&planning_protocol) {
        return Err("known planning protocol required".into());
    }
    let route_mode = *options.get("--routes").unwrap_or(&"none");
    if !["none", "persona"].contains(&route_mode) {
        return Err("known route mode required".into());
    }
    let route_selection = route_mode == "persona";
    let guard_mode = *options.get("--forecast-guard").unwrap_or(&"none");
    if !["none", "batch"].contains(&guard_mode) {
        return Err("known forecast guard required".into());
    }
    let forecast_guard = guard_mode == "batch";
    let opponent_models = match *options.get("--opponent-models").unwrap_or(&"adapter") {
        "adapter" => ModelSet::Adapter,
        "appraisal-mixture" if individual => ModelSet::AppraisalMixture,
        _ => return Err("appraisal-mixture requires individual public beliefs".into()),
    };
    if !["planned", "reflex", "challenger"].contains(&mode) || turns == 0 || turns > 128 {
        return Err("bounded mode/turns required".into());
    }
    let output = PathBuf::from(options.get("--out").ok_or("--out required")?);
    let map = *options.get("--map").unwrap_or(&"cover");
    let goal = *options.get("--goal").unwrap_or(&"either");
    if !["open", "cover", "choke"].contains(&map)
        || !["eliminate", "secure", "either", "both"].contains(&goal)
    {
        return Err("known combat map and goal required".into());
    }
    match game {
        "combat" => run(
            Combat::new(seed, map, goal, turns),
            game,
            seed,
            horizon,
            second_moves,
            max_regret,
            scenario_batches,
            mode,
            individual,
            coordinate,
            share_intentions,
            frozen_reference,
            planning_protocol,
            route_selection,
            forecast_guard,
            opponent_models,
            output,
        ),
        "hagetaka" => run(
            Hagetaka::new(seed),
            game,
            seed,
            horizon,
            second_moves,
            max_regret,
            scenario_batches,
            mode,
            individual,
            coordinate,
            share_intentions,
            frozen_reference,
            planning_protocol,
            route_selection,
            forecast_guard,
            opponent_models,
            output,
        ),
        "exchange" => run(
            Encounters::new(seed, turns),
            game,
            seed,
            horizon,
            second_moves,
            max_regret,
            scenario_batches,
            mode,
            individual,
            coordinate,
            share_intentions,
            frozen_reference,
            planning_protocol,
            route_selection,
            forecast_guard,
            opponent_models,
            output,
        ),
        _ => Err("unknown arena".into()),
    }
}
