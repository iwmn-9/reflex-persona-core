//! Public appraisals supplement the adapter hypotheses; real opponent state is never read.
use crate::{behavior::{ModelSet, PublicBehavior}, routes, ArenaGame};
use reflex_immediate::{Agent, Config, Policy, Priority, Scope, State};
use reflex_owned::{Error, Result};

pub fn frame<G: ArenaGame>(game: &G, viewer: usize, target: usize, models: ModelSet) -> Result<PublicBehavior> {
    if viewer >= game.players() || target >= game.players() {
        return Err(Error::Invalid("valid public viewer and target required"));
    }
    if models == ModelSet::Adapter || viewer == target {
        return game.behavior(viewer, target);
    }
    let public = game.model_world(viewer, 0x5eed)?.with_route(target, None)?;
    let mut result = public.behavior(viewer, target)?;
    let mut suggestions = result.suggestions.clone();
    for value_index in [5, 3, 8] {
        let mut values = [0.; 10];
        values[value_index] = 0.8;
        let config = Config {
            scope: Scope {
                game: "synthetic-public".into(),
                episode: "forecast".into(),
                npc: "hypothesis".into(),
            },
            seed: 701,
            personality: [0.5; 5],
            values,
            objective: "Pursue the public game objective with personal values".into(),
            policy: Policy {
                priority: Priority::Personal { rigidity: 0.5 },
                residual: [0.; 8],
            },
        };
        let state = State::default();
        let mut branch = public.clone();
        if let Some(observation) = branch.route_input(target)? {
            let route = routes::choose(
                &config,
                &state,
                &observation,
                &routes::RouteState::default(),
                0.,
                0.06,
            )?;
            branch = branch.with_route(target, route.state.chosen.as_deref())?;
        }
        let observation = branch.input(target)?.observation;
        let decision = Agent::from_state(config, state, None)?.preview(&observation, false)?;
        let action = decision.decision().action_id.clone();
        if !result.actions.contains_key(&action) {
            return Err(Error::Invalid(
                "synthetic proposal must be a public legal action",
            ));
        }
        suggestions.push(action);
    }
    result.suggestions = suggestions;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::{Combat, Encounters, Hagetaka};

    fn check_frame<G: ArenaGame>(game: &G) {
        let before = game.view();
        for target in 1..game.players() {
            let adapter = game.behavior(0, target).unwrap();
            let old = frame(game, 0, target, ModelSet::Adapter).unwrap();
            assert_eq!(
                serde_json::to_value(&old).unwrap(),
                serde_json::to_value(&adapter).unwrap()
            );
            let mixed = frame(game, 0, target, ModelSet::AppraisalMixture).unwrap();
            assert_eq!(mixed.alphabet, adapter.alphabet);
            assert_eq!(mixed.context, adapter.context);
            assert_eq!(mixed.actions, adapter.actions);
            assert_eq!(mixed.suggestions.len(), 6);
            assert_eq!(&mixed.suggestions[..3], adapter.suggestions.as_slice());
            assert!(mixed.suggestions[3..]
                .iter()
                .all(|action| mixed.actions.contains_key(action)));
        }
        assert_eq!(before, game.view());
    }

    #[test]
    fn adapter_and_synthetic_candidates_preserve_public_legal_frame() {
        check_frame(&Combat::new(1, "cover", "both", 20));
        check_frame(&Encounters::new(1, 20));
        check_frame(&Hagetaka::new(1));
    }

    #[test]
    fn target_private_route_does_not_influence_hypotheses_or_mutate_world() {
        let initial = Combat::new(1, "cover", "both", 20)
            .with_route(0, Some("secure"))
            .unwrap();
        let secure = initial.with_route(1, Some("secure")).unwrap();
        let eliminate = initial.with_route(1, Some("eliminate")).unwrap();
        assert_eq!(secure.view(), eliminate.view());
        assert_eq!(secure.current_route(1), Some("secure"));
        assert_eq!(eliminate.current_route(1), Some("eliminate"));
        let a = frame(&secure, 0, 1, ModelSet::AppraisalMixture).unwrap();
        let b = frame(&eliminate, 0, 1, ModelSet::AppraisalMixture).unwrap();
        assert_eq!(
            serde_json::to_value(a).unwrap(),
            serde_json::to_value(b).unwrap()
        );
        assert_eq!(secure.current_route(0), Some("secure"));
        assert_eq!(secure.current_route(1), Some("secure"));
        assert_eq!(eliminate.current_route(1), Some("eliminate"));
        assert_eq!(initial.current_route(1), None);
    }
    #[test]
    fn hidden_future_deck_order_cannot_change_public_hypotheses() {
        let original = Hagetaka::new(11);
        let first = original.model_world(0, 17).unwrap();
        let second = original.model_world(0, 29).unwrap();
        assert_eq!(first.view(), second.view());
        let a = frame(&first, 0, 1, ModelSet::AppraisalMixture).unwrap();
        let b = frame(&second, 0, 1, ModelSet::AppraisalMixture).unwrap();
        assert_eq!(
            serde_json::to_value(a).unwrap(),
            serde_json::to_value(b).unwrap()
        );
        assert_eq!(original.view(), first.view());
    }
}
