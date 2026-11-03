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
