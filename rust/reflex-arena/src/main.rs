//! Headless simultaneous games with isolated personalities and factual learning.
use reflex_arena::{
    behavior::Beliefs,
    games::{Combat, Encounters, Hagetaka},
    observe_public,
    planner::{Config as Search, Planner},
    public_input, ArenaGame,
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
        contract: format!("arena-{game}-estimated-model-v2"),
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
    mode: &str,
    individual: bool,
    coordinate: bool,
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
                    second_moves,
                    max_regret,
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
        .map(|i| Beliefs::new(scopes[i].clone(), i, alphabets[i].clone()))
        .collect::<Result<_, _>>()?;
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
        let mut next_owners = owners
            .iter()
            .map(Owner::fork_model)
            .collect::<Result<Vec<_>, _>>()?;
        let mut next_planners = planners.clone();
        let mut next_beliefs = beliefs.clone();
        let mut choices = vec![];
        let mut audits = vec![];
        let mut inputs = vec![];
        for (i, belief) in beliefs.iter().enumerate() {
            inputs.push(if individual {
                public_input(&world, i, belief, 701)?
            } else {
                world.input(i)?
            });
        }
        let mut selections = vec![];
        for i in 0..world.players() {
            if planning[i] {
                let proposal = if individual {
                    planners[i].propose_with_beliefs(&owners[i], &world, i, &beliefs[i])?
                } else {
                    planners[i].propose(&owners[i], &world, i)?
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
        let mut agreements = vec![];
        if coordinate {
            let offers = world.offers(&choices)?;
            if !offers.is_empty() {
                let mut consents = Vec::new();
                let mut ballots = Vec::new();
                for i in 0..world.players() {
                    let consent = planners[i].consent(
                        &owners[i],
                        &world,
                        i,
                        &beliefs[i],
                        &choices,
                        &offers,
                    )?;
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
                let order: Vec<_> = (0..world.players())
                    .map(|i| (i + tick as usize) % world.players())
                    .collect();
                let agreement = reflex_arena::coordination::negotiate(
                    &offers, &ballots, &order, &legal, &scopes[0], tick,
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
            world.advance(&choices, seed.wrapping_mul(1009).wrapping_add(tick))?;
        if individual {
            for i in 0..world.players() {
                observe_public(&world, i, &mut next_beliefs[i], &choices)?;
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
        for i in 0..world.players() {
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
            next_beliefs[i] =
                Beliefs::restore(&next_beliefs[i].checkpoint()?, &scopes[i], i, &alphabets[i])?;
            participants.push(json!({"player":i,"profile":profiles[i],"name":NAMES[profiles[i]],
                "action":choices[i],"purpose_before":world.purpose(i),"purpose_after":next.purpose(i),
                "needs":inputs[i].observation.needs,"learning_observations":tick+1,
                "prediction_observations":next_owners[i].state()?.prediction.observations,
                "public_behavior_observations":next_beliefs[i].observations(),
                "planning":audits[i]}));
            checks += 1;
        }
        trace.push(json!({"tick":tick,"before":before,"after":next.view(),"actors":participants,"agreements":agreements}));
        world = next;
        owners = next_owners;
        planners = next_planners;
        beliefs = next_beliefs;
        eprintln!("{game} {mode} tick={} nodes={simulations}", world.tick());
    }
    let report = json!({"complete":true,"game":game,"seed":seed,"horizon":horizon,"second_moves":second_moves,"max_regret":max_regret,"mode":mode,"individual_beliefs":individual,"coordination":coordinate,
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
            "--turns",
            "--mode",
            "--out",
            "--map",
            "--goal",
            "--beliefs",
            "--coordination",
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
            mode,
            individual,
            coordinate,
            output,
        ),
        "hagetaka" => run(
            Hagetaka::new(seed),
            game,
            seed,
            horizon,
            second_moves,
            max_regret,
            mode,
            individual,
            coordinate,
            output,
        ),
        "exchange" => run(
            Encounters::new(seed, turns),
            game,
            seed,
            horizon,
            second_moves,
            max_regret,
            mode,
            individual,
            coordinate,
            output,
        ),
        _ => Err("unknown arena".into()),
    }
}
