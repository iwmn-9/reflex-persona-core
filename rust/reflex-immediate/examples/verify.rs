//! JSON-lines differential-test driver. Uses the public API, no private numeric bypass.
use reflex_immediate::{Agent, Config, Observation, State, MAX_JSON_BYTES};
use serde::Deserialize;
use std::io::{self, BufRead, Write};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    config: Config,
    state: Option<State>,
    checkpoint: Option<String>,
    observation: Observation,
    stochastic: bool,
}

fn run(line: &str) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    if line.len() > MAX_JSON_BYTES {
        return Err("input byte limit exceeded".into());
    }
    let request: Input = serde_json::from_str(line)?;
    let mut actor = match request.checkpoint {
        Some(bytes) => {
            if request.state.is_some() {
                return Err("checkpoint and initial state are mutually exclusive".into());
            }
            Agent::restore(&request.config, bytes.as_bytes())?
        }
        None => Agent::from_state(
            request.config.clone(),
            request.state.unwrap_or_default(),
            None,
        )?,
    };
    let prepared = actor.preview(&request.observation, request.stochastic)?;
    let key = prepared.decision().action_id.clone();
    let decision = actor.commit(prepared, &key)?;
    let checkpoint = actor.checkpoint()?;
    let restored = Agent::restore(&request.config, &checkpoint)?;
    if restored.checkpoint()? != checkpoint {
        return Err("checkpoint roundtrip drift".into());
    }
    Ok(serde_json::json!({"decision": decision, "checkpoint": String::from_utf8(checkpoint)?}))
}

fn main() -> io::Result<()> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let result = match run(&line?) {
            Ok(value) => value,
            Err(e) => serde_json::json!({"error": e.to_string()}),
        };
        writeln!(out, "{result}")?;
        out.flush()?;
    }
    Ok(())
}
