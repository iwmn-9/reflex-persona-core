#[allow(dead_code)]
#[path = "../../reflex-owned/tests/support/mod.rs"]
mod fixture;
use reflex_arena::{
    behavior::Beliefs,
    coordination::{negotiate, Ballot},
    games::{Combat, Encounters, Hagetaka},
    observe_public,
    planner::{Config, Planner},
    ArenaGame,
};
use reflex_owned::Owner;
use std::collections::{BTreeMap, BTreeSet};

fn owner(i: usize) -> Owner {
    let mut cfg = fixture::config(&format!("p{i}"));
    cfg.contract = "arena-estimated-model-v1".into();
    cfg.predictor.names = vec!["0".into(), "1".into(), "2".into()];
    Owner::new(cfg).unwrap()
}
fn scope(owner: &Owner) -> String {
    serde_json::to_string(&(
        &owner.config().actor.scope.game,
        &owner.config().actor.scope.episode,
    ))
    .unwrap()
}
fn new_beliefs<G: ArenaGame>(game: &G, owner: &Owner, viewer: usize) -> Beliefs {
    Beliefs::new(
        scope(owner),
        viewer,
        game.behavior(viewer, viewer).unwrap().alphabet,
    )
    .unwrap()
}
fn planner(owner: &Owner) -> Planner {
    Planner::new(
        owner,
        Config {
            horizon: 1,
            ..Default::default()
        },
    )
    .unwrap()
}

fn restore_round<G: ArenaGame>(game: G) {
    let mut person = owner(0);
    let mut planner = planner(&person);
    let mut beliefs = new_beliefs(&game, &person, 0);
    let person_before = person.state().unwrap().clone();
    let beliefs_before = beliefs.checkpoint().unwrap();
    let proposal = planner
        .propose_with_beliefs(&person, &game, 0, &beliefs)
        .unwrap();
    assert_eq!(person.state().unwrap(), &person_before);
    assert_eq!(beliefs.checkpoint().unwrap(), beliefs_before);
    let action = proposal.action().to_owned();
    let mut choices = game.modeled_choices(0, "0").unwrap();
    choices[0] = action.clone();
    let (next, mut feedback) = game.advance(&choices, 439).unwrap();
    observe_public(&game, 0, &mut beliefs, &choices).unwrap();
    feedback[0].response = None;
    planner
        .resolve(
            &mut person,
            proposal,
            &action,
            &feedback[0],
            next.purpose(0),
        )
        .unwrap();
    assert_eq!(beliefs.observations(), (game.players() - 1) as u64);
    let public_state: serde_json::Value =
        serde_json::from_slice(&beliefs.checkpoint().unwrap()).unwrap();
    assert!(public_state["people"].get("0").is_none());
    for i in 1..game.players() {
        let frame = game.behavior(0, i).unwrap();
        let observed = &public_state["people"][i.to_string()]["memory"]["history"][0]["outcome"];
        assert_eq!(observed.as_str(), Some(frame.actions[&choices[i]].as_str()));
        assert_eq!(
            public_state["people"][i.to_string()]["memory"]["observations"],
            1
        );
    }
    let config = person.config().clone();
    let save = person.checkpoint(&next.view()).unwrap();
    let restored_owner = Owner::restore(&config, &next.view(), &save).unwrap();
    let restored_planner =
        Planner::restore(&restored_owner, &planner.checkpoint().unwrap()).unwrap();
    let alphabet = next.behavior(0, 0).unwrap().alphabet;
    let restored_beliefs = Beliefs::restore(
        &beliefs.checkpoint().unwrap(),
        &scope(&person),
        0,
        &alphabet,
    )
    .unwrap();
    assert_eq!(restored_owner.state().unwrap(), person.state().unwrap());
    assert_eq!(
        restored_beliefs.checkpoint().unwrap(),
        beliefs.checkpoint().unwrap()
    );
    let again = planner
        .propose_with_beliefs(&person, &next, 0, &beliefs)
        .unwrap();
    let replay = restored_planner
        .propose_with_beliefs(&restored_owner, &next, 0, &restored_beliefs)
        .unwrap();
    assert_eq!(again.action(), replay.action());
    assert_eq!(
        serde_json::to_value(&again.audit).unwrap(),
        serde_json::to_value(&replay.audit).unwrap()
    );
    assert!(restored_planner
        .propose_with_beliefs(&restored_owner, &next, 0, &new_beliefs(&next, &person, 1))
        .is_err());
}
#[test]
fn individual_beliefs_do_not_mutate_and_restore_the_next_decision_in_three_games() {
    restore_round(Hagetaka::new(7));
    restore_round(Encounters::new(7, 5));
    restore_round(Combat::new(7, "open", "eliminate", 5));
}

#[test]
fn exchange_offers_require_consent_and_only_agreed_proposals_execute() {
    let game = Encounters::new(7, 5);
    let n = game.players();
    let mut owners: Vec<_> = (0..n).map(owner).collect();
    let mut planners: Vec<_> = owners.iter().map(planner).collect();
    let bs: Vec<_> = (0..n).map(|i| new_beliefs(&game, &owners[i], i)).collect();
    let mut proposals: Vec<_> = (0..n)
        .map(|i| {
            Some(
                planners[i]
                    .propose_with_beliefs(&owners[i], &game, i, &bs[i])
                    .unwrap(),
            )
        })
        .collect();
    let mut fallback_proposals: Vec<_> = (0..n)
        .map(|i| {
            Some(
                planners[i]
                    .propose_with_beliefs(&owners[i], &game, i, &bs[i])
                    .unwrap(),
            )
        })
        .collect();
    let announced: Vec<_> = proposals
        .iter()
        .map(|p| p.as_ref().unwrap().action().to_owned())
        .collect();
    let offers = game.offers(&announced).unwrap();
    assert!(!offers.is_empty());
    let foreign = Beliefs::new(
        "foreign-episode".into(),
        0,
        game.behavior(0, 0).unwrap().alphabet,
    )
    .unwrap();
    assert!(planners[0]
        .consent(&owners[0], &game, 0, &foreign, &announced, &offers)
        .is_err());
    let mut consents: Vec<_> = (0..n)
        .map(|i| {
            planners[i]
                .consent(&owners[i], &game, i, &bs[i], &announced, &offers)
                .unwrap()
        })
        .collect();
    let scope = scope(&owners[0]);
    let ballots: Vec<_> = (0..n)
        .map(|i| Ballot {
            actor: i,
            scope: scope.clone(),
            tick: game.tick(),
            outside: announced[i].clone(),
            ranked: consents[i].ranked.clone(),
        })
        .collect();
    let legal: BTreeMap<usize, BTreeSet<String>> = (0..n)
        .map(|i| {
            (
                i,
                game.input(i)
                    .unwrap()
                    .observation
                    .actions
                    .into_iter()
                    .filter(|a| a.legal && !a.known_failure)
                    .map(|a| a.id)
                    .collect(),
            )
        })
        .collect();
    let order: Vec<_> = (0..n).collect();
    let agreement = negotiate(&offers, &ballots, &order, &legal, &scope, game.tick()).unwrap();
    for id in &agreement.accepted {
        let offer = offers.iter().find(|o| o.id == *id).unwrap();
        for (&i, root) in &offer.roots {
            let signed = consents[i].proposals.remove(id).unwrap();
            assert_eq!(signed.action(), root);
            proposals[i] = Some(signed);
        }
    }
    for (proposal, choice) in proposals.iter().zip(&agreement.choices) {
        assert_eq!(proposal.as_ref().unwrap().action(), choice);
    }
    let (next, mut feedback) = game.advance(&agreement.choices, 827).unwrap();
    for i in 0..n {
        feedback[i].response = None;
        planners[i]
            .resolve(
                &mut owners[i],
                proposals[i].take().unwrap(),
                &agreement.choices[i],
                &feedback[i],
                next.purpose(i),
            )
            .unwrap();
    }
    assert!(owners
        .iter()
        .all(|o| o.state().unwrap().last_tick == Some(0)));
    let mut vetoed = ballots.clone();
    for b in &mut vetoed {
        b.ranked.clear()
    }
    let refused = negotiate(&offers, &vetoed, &order, &legal, &scope, game.tick()).unwrap();
    assert!(refused.accepted.is_empty());
    assert_eq!(refused.choices, announced);
    let (fallback_world, mut fallback_feedback) = game.advance(&refused.choices, 827).unwrap();
    let mut fallback_owners: Vec<_> = (0..n).map(owner).collect();
    let mut fallback_planners: Vec<_> = fallback_owners.iter().map(planner).collect();
    for i in 0..n {
        assert_eq!(
            fallback_proposals[i].as_ref().unwrap().action(),
            refused.choices[i]
        );
        fallback_feedback[i].response = None;
        fallback_planners[i]
            .resolve(
                &mut fallback_owners[i],
                fallback_proposals[i].take().unwrap(),
                &refused.choices[i],
                &fallback_feedback[i],
                fallback_world.purpose(i),
            )
            .unwrap();
    }
    assert!(fallback_owners
        .iter()
        .all(|o| o.state().unwrap().last_tick == Some(0)));
    assert!(
        !agreement.accepted.is_empty(),
        "test must exercise an independently accepted offer"
    );
}

#[test]
fn indifferent_person_accepts_a_benefit_for_another_but_not_own_loss() {
    use reflex_arena::coordination::Offer;
    let game = Encounters::new(7, 5);
    let make_owner = |i| {
        let mut cfg = fixture::config(&format!("p{i}"));
        cfg.contract = "arena-estimated-model-v1".into();
        cfg.predictor.names = vec!["0".into(), "1".into(), "2".into()];
        cfg.actor.values = [0.; 10];
        Owner::new(cfg).unwrap()
    };
    let owners = [make_owner(0), make_owner(1)];
    let make_planner = |o: &Owner| {
        Planner::new(
            o,
            Config {
                horizon: 1,
                max_regret: 1.,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let planners = [make_planner(&owners[0]), make_planner(&owners[1])];
    let beliefs = [
        new_beliefs(&game, &owners[0], 0),
        new_beliefs(&game, &owners[1], 1),
    ];
    let announced = vec!["rest".to_string(); game.players()];
    let helpful = Offer {
        id: "helpful".into(),
        roots: BTreeMap::from([(0, "rest".into()), (1, "alone".into())]),
    };
    let harmful = Offer {
        id: "harmful".into(),
        roots: BTreeMap::from([(0, "cooperate:1".into()), (1, "claim:0".into())]),
    };
    let (baseline, _) = game.advance(&announced, 701).unwrap();
    let mut joint = announced.clone();
    joint[1] = "alone".into();
    let (improved, _) = game.advance(&joint, 701).unwrap();
    assert!(improved.purpose(1) > baseline.purpose(1));
    assert_eq!(improved.purpose(0), baseline.purpose(0));
    // Same-tier fixture: zero values and neutral traits, so own score comes from objective,
    // needs and risk rather than selecting a different moral tier.
    let strict = planners[0]
        .consent(
            &owners[0],
            &game,
            0,
            &beliefs[0],
            &announced,
            &[helpful.clone(), harmful.clone()],
        )
        .unwrap();
    let beneficiary = planners[1]
        .consent_allow_indifference(
            &owners[1],
            &game,
            1,
            &beliefs[1],
            &announced,
            std::slice::from_ref(&helpful),
        )
        .unwrap();
    assert!(beneficiary.ranked.iter().any(|x| x == "helpful"));
    let allowed = planners[0]
        .consent_allow_indifference(
            &owners[0],
            &game,
            0,
            &beliefs[0],
            &announced,
            &[helpful, harmful],
        )
        .unwrap();
    assert!(strict.ranked.is_empty());
    assert!(allowed.ranked.iter().any(|x| x == "helpful"));
    assert!(!allowed.ranked.iter().any(|x| x == "harmful"));
}

#[derive(Clone, Debug)]
enum Observed {
    Input(u64, Vec<Vec<String>>),
    Advance(u64, Vec<String>),
}
#[derive(Clone)]
struct Recording<G> {
    game: G,
    events: std::sync::Arc<std::sync::Mutex<Vec<Observed>>>,
    partners_override: Option<Vec<usize>>,
}
impl<G: ArenaGame> ArenaGame for Recording<G> {
    fn players(&self) -> usize {
        self.game.players()
    }
    fn tick(&self) -> u64 {
        self.game.tick()
    }
    fn terminal(&self) -> bool {
        self.game.terminal()
    }
    fn input(&self, p: usize) -> reflex_owned::Result<reflex_owned::Input> {
        self.game.input(p)
    }
    fn behavior(
        &self,
        v: usize,
        p: usize,
    ) -> reflex_owned::Result<reflex_arena::behavior::PublicBehavior> {
        self.game.behavior(v, p)
    }
    fn input_scenarios(
        &self,
        p: usize,
        c: &[Vec<String>],
    ) -> reflex_owned::Result<reflex_owned::Input> {
        self.events
            .lock()
            .unwrap()
            .push(Observed::Input(self.tick(), c.to_vec()));
        self.game.input_scenarios(p, c)
    }
    fn model_world(&self, v: usize, seed: u64) -> reflex_owned::Result<Self> {
        Ok(Self {
            game: self.game.model_world(v, seed)?,
            events: self.events.clone(),
            partners_override: self.partners_override.clone(),
        })
    }
    fn advance(
        &self,
        c: &[String],
        seed: u64,
    ) -> reflex_owned::Result<(Self, Vec<reflex_owned::Feedback>)> {
        self.events
            .lock()
            .unwrap()
            .push(Observed::Advance(self.tick(), c.to_vec()));
        let (game, f) = self.game.advance(c, seed)?;
        Ok((
            Self {
                game,
                events: self.events.clone(),
                partners_override: self.partners_override.clone(),
            },
            f,
        ))
    }
    fn modeled_choices(&self, v: usize, r: &str) -> reflex_owned::Result<Vec<String>> {
        self.game.modeled_choices(v, r)
    }
    fn purpose(&self, p: usize) -> f64 {
        self.game.purpose(p)
    }
    fn view(&self) -> serde_json::Value {
        self.game.view()
    }
    fn partners(&self, p: usize) -> Vec<usize> {
        self.partners_override
            .clone()
            .unwrap_or_else(|| self.game.partners(p))
    }
    fn offers(&self, a: &[String]) -> reflex_owned::Result<Vec<reflex_arena::coordination::Offer>> {
        self.game.offers(a)
    }
}
fn frame<G: ArenaGame>(game: &G, viewer: usize, action: &str) -> reflex_arena::Intentions {
    reflex_arena::Intentions {
        tick: game.tick(),
        choices: game
            .partners(viewer)
            .into_iter()
            .map(|p| (p, action.to_owned()))
            .collect(),
    }
}

#[test]
fn intentions_reject_old_incomplete_illegal_and_enemy_frames() {
    use reflex_arena::public_input_with_intentions;
    let game = Combat::new(7, "open", "eliminate", 5);
    let person = owner(0);
    let beliefs = new_beliefs(&game, &person, 0);
    let valid = frame(&game, 0, "guard");
    assert!(public_input_with_intentions(&game, 0, &beliefs, 701, &valid).is_ok());
    let mut old = valid.clone();
    old.tick = 1;
    assert!(public_input_with_intentions(&game, 0, &beliefs, 701, &old).is_err());
    let mut missing = valid.clone();
    missing.choices.remove(&1);
    assert!(public_input_with_intentions(&game, 0, &beliefs, 701, &missing).is_err());
    let mut enemy = valid.clone();
    enemy.choices.insert(3, "guard".into());
    assert!(public_input_with_intentions(&game, 0, &beliefs, 701, &enemy).is_err());
    let mut out_of_range = valid.clone();
    out_of_range.choices.insert(game.players(), "guard".into());
    assert!(public_input_with_intentions(&game, 0, &beliefs, 701, &out_of_range).is_err());
    let mut own = valid.clone();
    own.choices.insert(0, "guard".into());
    assert!(public_input_with_intentions(&game, 0, &beliefs, 701, &own).is_err());
    let mut illegal = valid.clone();
    illegal.choices.insert(1, "not-an-action".into());
    assert!(public_input_with_intentions(&game, 0, &beliefs, 701, &illegal).is_err());
    let malformed = Recording {
        game: game.clone(),
        events: Default::default(),
        partners_override: Some(vec![1, 1]),
    };
    let duplicate = reflex_arena::Intentions {
        tick: game.tick(),
        choices: BTreeMap::from([(1, "guard".into())]),
    };
    assert!(public_input_with_intentions(&malformed, 0, &beliefs, 701, &duplicate).is_err());
}

#[test]
fn empty_partners_preserve_prior_path_exactly() {
    let game = Hagetaka::new(7);
    let person = owner(0);
    let beliefs = new_beliefs(&game, &person, 0);
    let planner = planner(&person);
    let prior = planner
        .propose_with_beliefs(&person, &game, 0, &beliefs)
        .unwrap();
    let new = planner
        .propose_with_intentions(&person, &game, 0, &beliefs, &frame(&game, 0, ""))
        .unwrap();
    assert_eq!(prior.action(), new.action());
    assert_eq!(
        serde_json::to_value(prior.audit).unwrap(),
        serde_json::to_value(new.audit).unwrap()
    );
    let a = reflex_arena::public_input(&game, 0, &beliefs, 701).unwrap();
    let b =
        reflex_arena::public_input_with_intentions(&game, 0, &beliefs, 701, &frame(&game, 0, ""))
            .unwrap();
    assert_eq!(
        serde_json::to_value(a).unwrap(),
        serde_json::to_value(b).unwrap()
    );
}

#[test]
fn same_intentions_reach_initial_input_scout_and_each_modeled_root_without_mutation() {
    let game = Recording {
        game: Encounters::new(7, 5),
        events: Default::default(),
        partners_override: None,
    };
    let person = owner(0);
    let beliefs = new_beliefs(&game, &person, 0);
    let planner = Planner::new(
        &person,
        Config {
            horizon: 2,
            second_moves: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let intentions = frame(&game, 0, "rest");
    let before = person.state().unwrap().clone();
    let memory = beliefs.checkpoint().unwrap();
    let input =
        reflex_arena::public_input_with_intentions(&game, 0, &beliefs, 701, &intentions).unwrap();
    assert!(!input.observation.facts.contains_key("agreement"));
    assert!(input
        .bindings
        .values()
        .all(|b| !b.situation.contains("accepted-current-agreement")));
    let roots = person
        .prepare_estimated(&input, false)
        .unwrap()
        .allowed()
        .len();
    game.events.lock().unwrap().clear();
    let proposal = planner
        .propose_with_intentions(&person, &game, 0, &beliefs, &intentions)
        .unwrap();
    assert!(!proposal.action().is_empty());
    assert_eq!(person.state().unwrap(), &before);
    assert_eq!(beliefs.checkpoint().unwrap(), memory);
    let events = game.events.lock().unwrap();
    let root_inputs: Vec<_> = events
        .iter()
        .filter_map(|e| {
            if let Observed::Input(0, rows) = e {
                Some(rows)
            } else {
                None
            }
        })
        .collect();
    let root_advances: Vec<_> = events
        .iter()
        .filter_map(|e| {
            if let Observed::Advance(0, row) = e {
                Some(row)
            } else {
                None
            }
        })
        .collect();
    // One initial input; each legal root has one scout plus 2 candidates x 3 branches.
    assert_eq!(root_inputs.len(), 1 + roots * 2 * 3);
    assert_eq!(root_advances.len(), roots * (1 + 2 * 3));
    for rows in root_inputs {
        for row in rows {
            for (&p, action) in &intentions.choices {
                assert_eq!(&row[p], action);
            }
        }
    }
    for row in root_advances {
        for (&p, action) in &intentions.choices {
            assert_eq!(&row[p], action);
        }
    }
    // Future inputs are resampled, so the first-turn frame is not held as an agreement.
    assert!(events.iter().any(|e| match e {
        Observed::Input(t, rows) if *t > 0 => rows
            .iter()
            .any(|row| intentions.choices.iter().any(|(&p, a)| &row[p] != a)),
        _ => false,
    }));
}
