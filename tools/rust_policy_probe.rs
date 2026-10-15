//! Isolated port feasibility probe, not a production API.
//! Input is an already validated numeric actor from Python compile_batch.
//! No social appraisal, return-frame conversion, JSON/hash, or learning here.
use std::io::{self, Read};

fn argmax(xs: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..xs.len() { if xs[i] > xs[best] { best = i; } }
    best
}
fn random(seed: u64, tick: u64) -> [f64; 3] {
    std::array::from_fn(|i| {
        let mut x = seed.wrapping_add(tick.wrapping_mul(0x9E3779B97F4A7C15))
            .wrapping_add((i as u64 + 1).wrapping_mul(0xD1B54A32D192ED03));
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 31;
        (x >> 11) as f64 * (1.0 / 9007199254740992.0)
    })
}
struct Input<'a>(std::str::SplitWhitespace<'a>);
impl Input<'_> {
    fn get<T: std::str::FromStr>(&mut self) -> T where T::Err: std::fmt::Debug {
        self.0.next().expect("missing numeric input").parse().expect("numeric token")
    }
    fn array<const N: usize>(&mut self) -> [f64; N] { std::array::from_fn(|_| self.get()) }
}
fn main() {
    let mut text = String::new(); io::stdin().read_to_string(&mut text).unwrap();
    let mut input = Input(text.split_whitespace());
    let count: usize = input.get();
    for _ in 0..count {
        let a: usize = input.get(); let k: usize = input.get();
        let stochastic: i32 = input.get(); let priority: i32 = input.get();
        let rigidity: f64 = input.get(); let seed = input.get(); let tick = input.get();
        let traits = input.array::<5>(); let values = input.array::<10>();
        let needs = input.array::<5>(); let enabled = input.array::<5>();
        let previous: i32 = input.get(); let previous_mode: i32 = input.get();
        let intent: i32 = input.get(); let age: usize = input.get();
        let old_urgency: f64 = input.get(); let read_confidence: f64 = input.get();
        let uncertainty: f64 = input.get(); let threatened: i32 = input.get(); let ahead: i32 = input.get();
        let residual = input.array::<8>(); let rng = random(seed, tick);
        let best = argmax(&needs); let old = previous.max(0) as usize;
        let keep = previous >= 0 && enabled[old] != 0.0 && needs[old] > 0.05 && needs[best] < needs[old] + 0.15;
        let primary = if enabled.iter().any(|x| *x != 0.0) && needs[best] > 0.05 {
            if keep { previous } else { best as i32 }
        } else { -1 };
        let urgency = if primary >= 0 { needs[primary as usize] } else { 0.0 };
        let top = argmax(&values); let strength = values[top];
        let hold = (0.5 + 0.35*(traits[1]-0.5) - 0.25*(traits[4]-0.5) + 0.3*(strength-urgency)).clamp(0.1,0.9);
        let valid_old = previous_mode >= 0 && primary == previous && (urgency-old_urgency).abs() < 0.15;
        let mut mode = if valid_old { previous_mode } else { i32::from(rng[0] < hold) };
        if priority == 2 && rigidity == 1.0 { mode = 1; }
        if strength == 0.0 { mode = 0; }
        let mode_urgency = if valid_old { old_urgency } else { urgency };
        let mut w = [0.0;22]; w[0] = 0.65; w[21] = -0.6;
        for i in 0..5 { w[1+i] = 0.12*needs[i]*enabled[i]; w[16+i] = 0.12*(2.0*traits[i]-1.0); }
        if primary >= 0 { w[1+primary as usize] += 1.5*urgency; }
        for i in 0..10 { w[6+i] = 0.08*values[i]; }
        w[6+top] += strength*if mode == 1 { 1.6 } else { 0.3 };
        let risk = 0.25 + 0.75*traits[4] + 0.25*needs[1];
        let mut scores = vec![0.0; a]; let mut top_effect = vec![0.0;a];
        let mut features = vec![[0.0;8];a]; let mut viable = vec![false;a]; let mut reading = vec![0.0;a];
        for j in 0..a {
            let confidence: f64 = input.get(); let familiarity: f64 = input.get();
            let failure: i32 = input.get(); let switch_cost: f64 = input.get();
            let legal: i32 = input.get(); reading[j] = input.get(); viable[j] = legal != 0 && failure == 0;
            let probabilities: Vec<f64> = (0..k).map(|_| input.get()).collect();
            let mut means = [0.0;22]; let mut expected = 0.0; let mut loss = 0.0;
            for p in probabilities {
                let mut outcome = 0.0;
                for d in 0..22 {
                    let v: f64 = input.get(); let perceived = if d < 21 && v > 0.0 { v*confidence } else { v };
                    outcome += perceived*w[d]; means[d] += p*perceived;
                }
                expected += p*outcome; loss += p*(-outcome).max(0.0);
            }
            let familiar = 0.10*(1.0-confidence)*familiarity;
            let switch = if intent != j as i32 { switch_cost } else { 0.0 };
            top_effect[j] = means[6+top];
            features[j] = [means[0],if primary >= 0 { means[1+primary as usize] } else { 0.0 },
                top_effect[j],loss,means[21],familiar,switch,confidence];
            let adjustment: f64 = features[j].iter().zip(residual).map(|(x,w)|x*w).sum();
            scores[j] = expected - risk*loss + familiar - switch + adjustment.clamp(-0.08,0.08);
        }
        assert!(viable.iter().any(|v|*v));
        let tier = (0..a).filter(|j|viable[*j]).map(|j|top_effect[j]).fold(f64::NEG_INFINITY,f64::max);
        let tolerance = if priority == 2 {0.02 + 1.98*(1.0-rigidity).powi(2)} else {0.02};
        let eligible: Vec<bool> = (0..a).map(|j|viable[j] && (priority == 1 || mode == 0 || top_effect[j] >= tier-tolerance)).collect();
        let masked = |s: &[f64]| -> Vec<f64> {(0..a).map(|j|if eligible[j] {s[j]} else {f64::NEG_INFINITY}).collect()};
        let own = argmax(&masked(&scores));
        let candidate_scores: Vec<f64> = (0..a).map(|j|scores[j]+reading[j]*read_confidence).collect();
        let candidate = argmax(&masked(&candidate_scores));
        let gain = candidate_scores[candidate]-candidate_scores[own];
        let read_used = read_confidence >= 0.6 && (ahead == 0 || threatened != 0) && (gain > 0.05+uncertainty*0.15 || threatened != 0);
        if read_used { scores = candidate_scores; }
        let best_action = argmax(&masked(&scores)); let maximum = scores[best_action];
        let mut weights: Vec<f64> = (0..a).map(|j| if eligible[j] && scores[j] >= maximum-0.025 {
            ((scores[j]-maximum)/0.008).clamp(-80.0,0.0).exp()
        } else {0.0}).collect();
        let total: f64 = weights.iter().sum(); for w in &mut weights { *w /= total; }
        let mut cumulative = 0.0; let mut sampled = 0;
        for weight in weights { cumulative += weight; if cumulative < rng[1] { sampled += 1; } }
        let chosen = if stochastic != 0 {sampled.min(a-1)} else {best_action};
        let request = (threatened != 0 || (ahead == 0 && maximum < 0.15)) && read_confidence < 0.6;
        let age = if intent == chosen as i32 {(age+1).min(1000000)} else {0};
        print!("{} {} {} {:.17e} {} {} {}",chosen,primary,mode,mode_urgency,i32::from(read_used),i32::from(request),age);
        for v in rng {print!(" {:.17e}",v);}
        for v in scores {print!(" {:.17e}",v);}
        for row in features {for v in row {print!(" {:.17e}",v);}}
        for v in eligible {print!(" {}",i32::from(v));}
        println!();
    }
    assert!(input.0.next().is_none(),"trailing input");
}
