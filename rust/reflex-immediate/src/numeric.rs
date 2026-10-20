//! Translation of the unframed, non-social Policy.decide path in reflex/core.py.
//! Reduction order matches NumPy's two-lane, non-FMA f64 einsum kernel.
//! Supported parity targets: the tested x86-64 Windows/Linux reference builds.
use crate::{
    ActionScore, Config, Decision, Error, Mode, Need, Observation, Priority, Result, State,
};
use sha2::{Digest, Sha256};

fn argmax(xs: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..xs.len() {
        if xs[i] > xs[best] {
            best = i;
        }
    }
    best
}

fn dot(xs: &[f64], ys: &[f64]) -> f64 {
    // Keep the operation tree explicit. Sequential addition can reverse the
    // winner between mathematically equivalent candidates. This is scalar
    // emulation of the reference reduction, not platform-native SIMD.
    let mut lanes = [0.0; 2];
    let mut offset = 0;
    while offset + 8 <= xs.len() {
        for (lane, accumulator) in lanes.iter_mut().enumerate() {
            for group in (0..4).rev() {
                let i = offset + group * 2 + lane;
                *accumulator += xs[i] * ys[i];
            }
        }
        offset += 8;
    }
    while offset < xs.len() {
        for (lane, accumulator) in lanes.iter_mut().enumerate() {
            if offset + lane < xs.len() {
                *accumulator += xs[offset + lane] * ys[offset + lane];
            }
        }
        offset += 2;
    }
    lanes[0] + lanes[1]
}

fn random(seed: u64, tick: u64) -> [f64; 3] {
    std::array::from_fn(|i| {
        let mut x = seed
            .wrapping_add(tick.wrapping_mul(0x9E3779B97F4A7C15))
            .wrapping_add((i as u64 + 1).wrapping_mul(0xD1B54A32D192ED03));
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 31;
        (x >> 11) as f64 * (1.0 / 9007199254740992.0)
    })
}

pub(crate) fn choose(
    c: &Config,
    state: &State,
    o: &Observation,
    stochastic: bool,
) -> Result<Decision> {
    let hash = Sha256::digest(serde_json::to_vec(&(c.seed, &c.scope))?);
    let mut seed = [0_u8; 8];
    seed.copy_from_slice(&hash[..8]);
    let rng = random(u64::from_be_bytes(seed), o.tick);
    let mut actions: Vec<_> = o.actions.iter().collect();
    actions.sort_by(|a, b| a.id.cmp(&b.id));
    let a = actions.len();
    let outcome_capacity = actions.iter().map(|a| a.outcomes.len()).max().unwrap();
    let traits = c.personality;
    let values = c.values;
    let residual = c.policy.residual;
    let needs: [f64; 5] = std::array::from_fn(|i| o.needs[i].deficit.unwrap_or(0.));
    let enabled: [f64; 5] = std::array::from_fn(|i| f64::from(o.needs[i].enabled));
    let previous = state.primary_need.map_or(-1, |n| n as i32);
    let previous_mode = state.mode.map_or(-1, |m| i32::from(m == Mode::Principle));
    let intent = actions
        .iter()
        .position(|a| Some(&a.id) == state.intent_action.as_ref());
    let priority = match c.policy.priority {
        Priority::Lexicographic => 0,
        Priority::Finite => 1,
        Priority::Personal { .. } => 2,
    };
    let rigidity = match c.policy.priority {
        Priority::Personal { rigidity } => rigidity,
        _ => 0.,
    };
    let read_confidence = o.opponent.as_ref().map_or(0., |r| r.confidence);
    let uncertainty = o.opponent.as_ref().map_or(0., |r| r.uncertainty);
    let threatened = o.opponent.as_ref().is_some_and(|r| r.threatened);
    let ahead = o.opponent.as_ref().is_some_and(|r| r.maintains_advantage);
    let best = argmax(&needs);
    let old = previous.max(0) as usize;
    let keep =
        previous >= 0 && enabled[old] != 0. && needs[old] > 0.05 && needs[best] < needs[old] + 0.15;
    let primary = if enabled.iter().any(|x| *x != 0.) && needs[best] > 0.05 {
        if keep {
            previous
        } else {
            best as i32
        }
    } else {
        -1
    };
    let urgency = if primary >= 0 {
        needs[primary as usize]
    } else {
        0.
    };
    let top = argmax(&values);
    let strength = values[top];
    let hold = (0.5 + 0.35 * (traits[1] - 0.5) - 0.25 * (traits[4] - 0.5)
        + 0.3 * (strength - urgency))
        .clamp(0.1, 0.9);
    let valid_old =
        previous_mode >= 0 && primary == previous && (urgency - state.mode_urgency).abs() < 0.15;
    let mut mode = if valid_old {
        previous_mode
    } else {
        i32::from(rng[0] < hold)
    };
    if priority == 2 && rigidity == 1. {
        mode = 1;
    }
    if strength == 0. {
        mode = 0;
    }
    let mode_urgency = if valid_old {
        state.mode_urgency
    } else {
        urgency
    };
    let mut w = [0.; 22];
    w[0] = 0.65;
    w[21] = -0.6;
    for i in 0..5 {
        w[1 + i] = 0.12 * needs[i] * enabled[i];
        w[16 + i] = 0.12 * (2. * traits[i] - 1.);
    }
    if primary >= 0 {
        w[1 + primary as usize] += 1.5 * urgency;
    }
    for i in 0..10 {
        w[6 + i] = 0.08 * values[i];
    }
    w[6 + top] += strength * if mode == 1 { 1.6 } else { 0.3 };
    let risk = 0.25 + 0.75 * traits[4] + 0.25 * needs[1];
    let mut scores = vec![0.; a];
    let mut top_effect = vec![0.; a];
    let mut features = vec![[0.; 8]; a];
    let mut viable = vec![false; a];
    let mut reading = vec![0.; a];
    for (j, action) in actions.iter().enumerate() {
        let confidence = action.confidence;
        viable[j] = action.legal && !action.known_failure;
        reading[j] = o
            .opponent
            .as_ref()
            .and_then(|r| r.deltas.get(&action.id))
            .copied()
            .unwrap_or(0.);
        let mut means = [0.; 22];
        let mut probabilities = Vec::with_capacity(action.outcomes.len());
        let mut outcomes = Vec::with_capacity(action.outcomes.len());
        for e in &action.outcomes {
            let mut perceived = [0.0; 22];
            for (d, v) in e.effects().iter().enumerate() {
                perceived[d] = if d < 21 && *v > 0. {
                    v * confidence
                } else {
                    *v
                };
                means[d] += e.p * perceived[d];
            }
            probabilities.push(e.p);
            outcomes.push(dot(&perceived, &w));
        }
        // compile_batch([context]) pads every candidate to this observation's
        // maximum. Zero padding still affects the floating reduction tree.
        probabilities.resize(outcome_capacity, 0.);
        outcomes.resize(outcome_capacity, 0.);
        let expected = dot(&probabilities, &outcomes);
        let losses: Vec<f64> = outcomes.iter().map(|x| (-x).max(0.)).collect();
        let loss = dot(&probabilities, &losses);
        let familiar = 0.10 * (1. - confidence) * action.familiarity;
        let switch = if intent != Some(j) {
            action.switch_cost
        } else {
            0.
        };
        top_effect[j] = means[6 + top];
        features[j] = [
            means[0],
            if primary >= 0 {
                means[1 + primary as usize]
            } else {
                0.
            },
            top_effect[j],
            loss,
            means[21],
            familiar,
            switch,
            confidence,
        ];
        let adjustment = dot(&features[j], &residual);
        scores[j] = expected - risk * loss + familiar - switch + adjustment.clamp(-0.08, 0.08);
    }
    let tier = (0..a)
        .filter(|j| viable[*j])
        .map(|j| top_effect[j])
        .fold(f64::NEG_INFINITY, f64::max);
    let tolerance = if priority == 2 {
        0.02 + 1.98 * (1. - rigidity).powi(2)
    } else {
        0.02
    };
    let eligible: Vec<bool> = (0..a)
        .map(|j| viable[j] && (priority == 1 || mode == 0 || top_effect[j] >= tier - tolerance))
        .collect();
    let masked = |s: &[f64]| -> Vec<f64> {
        (0..a)
            .map(|j| if eligible[j] { s[j] } else { f64::NEG_INFINITY })
            .collect()
    };
    let own = argmax(&masked(&scores));
    let candidate_scores: Vec<f64> = (0..a)
        .map(|j| scores[j] + reading[j] * read_confidence)
        .collect();
    let candidate = argmax(&masked(&candidate_scores));
    let gain = candidate_scores[candidate] - candidate_scores[own];
    let read_used = read_confidence >= 0.6
        && (!ahead || threatened)
        && (gain > 0.05 + uncertainty * 0.15 || threatened);
    if read_used {
        scores = candidate_scores;
    }
    let best_action = argmax(&masked(&scores));
    let maximum = scores[best_action];
    let mut weights: Vec<f64> = (0..a)
        .map(|j| {
            if eligible[j] && scores[j] >= maximum - 0.025 {
                ((scores[j] - maximum) / 0.008).clamp(-80., 0.).exp()
            } else {
                0.
            }
        })
        .collect();
    let total: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total;
    }
    let mut cumulative = 0.;
    let mut sampled = 0;
    for weight in weights {
        cumulative += weight;
        if cumulative < rng[1] {
            sampled += 1;
        }
    }
    let chosen = if stochastic {
        sampled.min(a - 1)
    } else {
        best_action
    };
    if !eligible[chosen] {
        return Err(Error::Invalid(
            "floating sampling boundary produced an ineligible choice",
        ));
    }
    let request = (threatened || (!ahead && maximum < 0.15)) && read_confidence < 0.6;
    Ok(Decision {
        action_id: actions[chosen].id.clone(),
        target: actions[chosen].target.clone(),
        reading_used: read_used,
        request_reading: request,
        next_state: State {
            primary_need: if primary >= 0 {
                Some(Need::at(primary as usize))
            } else {
                None
            },
            mode: Some(if mode == 1 {
                Mode::Principle
            } else {
                Mode::Need
            }),
            mode_urgency,
            intent_action: Some(actions[chosen].id.clone()),
            age: if intent == Some(chosen) {
                (state.age + 1).min(1_000_000)
            } else {
                0
            },
        },
        scores: (0..a)
            .map(|j| ActionScore {
                id: actions[j].id.clone(),
                score: scores[j],
                features: features[j],
                eligible: eligible[j],
            })
            .collect(),
        rng,
    })
}
