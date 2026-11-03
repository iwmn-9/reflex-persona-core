//! One-turn voluntary coordination over privately ranked, already evaluated offers.
use reflex_owned::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Offer {
    pub id: String,
    pub roots: BTreeMap<usize, String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ballot {
    pub actor: usize,
    pub scope: String,
    pub tick: u64,
    pub outside: String,
    /// Only offers the actor independently accepts over their outside action.
    pub ranked: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Agreement {
    /// Indexed by actor, including each unassigned actor's independent choice.
    pub choices: Vec<String>,
    pub accepted: Vec<String>,
}
fn label(value: &str) -> bool {
    !value.is_empty() && value.chars().count() <= 128
}

/// Each unassigned actor tries ranked offers in explicit priority order.
/// Every participant must independently sign; overlapping accepted offers are
/// impossible. Inputs are validated before any choices are produced.
pub fn negotiate(
    offers: &[Offer],
    ballots: &[Ballot],
    order: &[usize],
    legal: &BTreeMap<usize, BTreeSet<String>>,
    scope: &str,
    tick: u64,
) -> Result<Agreement> {
    let count = ballots.len();
    if !(2..=64).contains(&count) || offers.len() > 128 || !label(scope) || tick >= (1_u64 << 63) {
        return Err(Error::Invalid(
            "bounded current coordination scope required",
        ));
    }
    let actors: BTreeSet<usize> = (0..count).collect();
    if legal.keys().copied().collect::<BTreeSet<_>>() != actors
        || order.len() != count
        || order.iter().copied().collect::<BTreeSet<_>>() != actors
        || legal
            .values()
            .any(|roots| roots.is_empty() || roots.len() > 256 || roots.iter().any(|r| !label(r)))
    {
        return Err(Error::Invalid(
            "one priority and nonempty legal set per actor required",
        ));
    }
    let mut bank = BTreeMap::new();
    for offer in offers {
        if !label(&offer.id)
            || !(2..=8).contains(&offer.roots.len())
            || offer
                .roots
                .iter()
                .any(|(actor, root)| !legal.get(actor).is_some_and(|xs| xs.contains(root)))
            || bank.insert(offer.id.as_str(), offer).is_some()
        {
            return Err(Error::Invalid(
                "distinct offers with legal participant roots required",
            ));
        }
    }
    let mut by_actor = BTreeMap::new();
    for ballot in ballots {
        if !actors.contains(&ballot.actor)
            || ballot.scope != scope
            || ballot.tick != tick
            || !legal[&ballot.actor].contains(&ballot.outside)
            || ballot.ranked.len() > offers.len()
            || ballot.ranked.iter().collect::<BTreeSet<_>>().len() != ballot.ranked.len()
            || ballot.ranked.iter().any(|id| {
                !bank
                    .get(id.as_str())
                    .is_some_and(|offer| offer.roots.contains_key(&ballot.actor))
            })
            || by_actor.insert(ballot.actor, ballot).is_some()
        {
            return Err(Error::Invalid(
                "one current legal ballot per actor required",
            ));
        }
    }
    if by_actor.len() != count {
        return Err(Error::Invalid("complete ballots required"));
    }
    let mut choices: Vec<_> = (0..count)
        .map(|actor| by_actor[&actor].outside.clone())
        .collect();
    let mut assigned = BTreeSet::new();
    let mut accepted = Vec::new();
    for actor in order {
        if assigned.contains(actor) {
            continue;
        }
        for id in &by_actor[actor].ranked {
            let offer = bank[id.as_str()];
            if offer.roots.keys().any(|member| assigned.contains(member)) {
                continue;
            }
            if offer
                .roots
                .keys()
                .any(|member| !by_actor[member].ranked.contains(id))
            {
                continue;
            }
            for (member, root) in &offer.roots {
                choices[*member] = root.clone();
                assigned.insert(*member);
            }
            accepted.push(id.clone());
            break;
        }
    }
    Ok(Agreement { choices, accepted })
}

#[cfg(test)]
mod tests {
    use super::*;
    type Fixture = (
        Vec<Offer>,
        Vec<Ballot>,
        Vec<usize>,
        BTreeMap<usize, BTreeSet<String>>,
    );
    fn setup() -> Fixture {
        let legal = (0..3)
            .map(|i| (i, BTreeSet::from(["alone".into(), "join".into()])))
            .collect();
        let offers = vec![
            Offer {
                id: "ab".into(),
                roots: BTreeMap::from([(0, "join".into()), (1, "join".into())]),
            },
            Offer {
                id: "bc".into(),
                roots: BTreeMap::from([(1, "join".into()), (2, "join".into())]),
            },
        ];
        let ballot = |actor, ranked: Vec<&str>| Ballot {
            actor,
            scope: "world".into(),
            tick: 4,
            outside: "alone".into(),
            ranked: ranked.into_iter().map(str::to_owned).collect(),
        };
        (
            offers,
            vec![
                ballot(0, vec!["ab"]),
                ballot(1, vec!["ab", "bc"]),
                ballot(2, vec!["bc"]),
            ],
            vec![0, 1, 2],
            legal,
        )
    }
    #[test]
    fn mutual_consent_never_double_assigns_and_unassigned_uses_outside() {
        let (offers, ballots, order, legal) = setup();
        let result = negotiate(&offers, &ballots, &order, &legal, "world", 4).unwrap();
        assert_eq!(result.accepted, vec!["ab"]);
        assert_eq!(result.choices, vec!["join", "join", "alone"]);
        let result = negotiate(&offers, &ballots, &[2, 1, 0], &legal, "world", 4).unwrap();
        assert_eq!(result.accepted, vec!["bc"]);
        assert_eq!(result.choices, vec!["alone", "join", "join"]);
    }
    #[test]
    fn veto_falls_back_and_does_not_prevent_other_offer() {
        let (offers, mut ballots, order, legal) = setup();
        ballots[1].ranked = vec!["bc".into()];
        let result = negotiate(&offers, &ballots, &order, &legal, "world", 4).unwrap();
        assert_eq!(result.accepted, vec!["bc"]);
        assert_eq!(result.choices, vec!["alone", "join", "join"]);
        ballots[2].ranked.clear();
        let result = negotiate(&offers, &ballots, &order, &legal, "world", 4).unwrap();
        assert!(result.accepted.is_empty());
        assert_eq!(result.choices, vec!["alone"; 3]);
    }
    #[test]
    fn stale_invalid_and_duplicate_inputs_are_rejected() {
        let (offers, ballots, order, legal) = setup();
        let mut bad = ballots.clone();
        bad[1].tick = 3;
        assert!(negotiate(&offers, &bad, &order, &legal, "world", 4).is_err());
        bad = ballots.clone();
        bad[1].scope = "other".into();
        assert!(negotiate(&offers, &bad, &order, &legal, "world", 4).is_err());
        bad = ballots.clone();
        bad[1].ranked.push("ab".into());
        assert!(negotiate(&offers, &bad, &order, &legal, "world", 4).is_err());
        bad = ballots.clone();
        bad[2].ranked.push("ab".into());
        assert!(negotiate(&offers, &bad, &order, &legal, "world", 4).is_err());
        let mut bad_offers = offers.clone();
        bad_offers[0].roots.insert(0, "illegal".into());
        assert!(negotiate(&bad_offers, &ballots, &order, &legal, "world", 4).is_err());
        assert!(negotiate(&offers, &ballots, &[0, 1, 1], &legal, "world", 4).is_err());
        let mut duplicate = offers.clone();
        duplicate.push(offers[0].clone());
        assert!(negotiate(&duplicate, &ballots, &order, &legal, "world", 4).is_err());
    }

    #[test]
    fn offer_limit_accepts_128_and_rejects_129() {
        let (_, mut ballots, order, legal) = setup();
        for ballot in &mut ballots {
            ballot.ranked.clear();
        }
        let mut offers: Vec<_> = (0..128)
            .map(|i| Offer {
                id: format!("offer-{i}"),
                roots: BTreeMap::from([(0, "join".into()), (1, "join".into())]),
            })
            .collect();
        assert!(negotiate(&offers, &ballots, &order, &legal, "world", 4).is_ok());
        offers.push(Offer {
            id: "overflow".into(),
            roots: offers[0].roots.clone(),
        });
        assert!(negotiate(&offers, &ballots, &order, &legal, "world", 4).is_err());
    }
}
