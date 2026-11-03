//! JSON-lines driver for differential route-selection checks.
use reflex_arena::routes::{choose, RouteState};
use reflex_immediate::{Config, Observation, State, MAX_JSON_BYTES};
use serde::Deserialize;
use std::io::{self, BufRead, Write};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    config: Config,
    state: State,
    observation: Observation,
    previous: RouteState,
    uncertainty: f64,
    switch_margin: f64,
}

fn run(line: &str) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    if line.len() > MAX_JSON_BYTES {
        return Err("input byte limit exceeded".into());
    }
    let input: Input = serde_json::from_str(line)?;
    Ok(serde_json::to_value(choose(
        &input.config,
        &input.state,
        &input.observation,
        &input.previous,
        input.uncertainty,
        input.switch_margin,
    )?)?)
}

fn main() -> io::Result<()> {
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let result = match run(&line?) {
            Ok(value) => value,
            Err(e) => serde_json::json!({"error": e.to_string()}),
        };
        writeln!(output, "{result}")?;
        output.flush()?;
    }
    Ok(())
}
