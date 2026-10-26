//! Test transport, not a production network server.
use reflex_owned::{Owner, OwnerConfig, Stage, Ticket};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

fn execute(
    v: Value,
    owner: &mut Option<Owner>,
    stage: &mut Option<Stage>,
    ticket: &mut Option<Ticket>,
) -> Result<Value, Box<dyn std::error::Error>> {
    let required = || io::Error::other("owner/stage required");
    match v["op"].as_str().unwrap_or("") {
        "new" => {
            let config: OwnerConfig = serde_json::from_value(v["config"].clone())?;
            *owner = Some(if v["state"].is_null() {
                Owner::new(config)?
            } else {
                Owner::from_state(config, serde_json::from_value(v["state"].clone())?)?
            });
            *stage = None;
            Ok(json!({"ok": true}))
        }
        "prepare" => {
            let s = owner.as_ref().ok_or_else(required)?.prepare(
                &serde_json::from_value(v["input"].clone())?,
                v["stochastic"].as_bool().unwrap_or(false),
            )?;
            let result =
                json!({"decision": s.decision(), "context": s.context(), "allowed": s.allowed()});
            *stage = Some(s);
            Ok(result)
        }
        "forecast" => {
            let forecast = serde_json::from_value(v["forecast"].clone())?;
            let s = owner
                .as_ref()
                .ok_or_else(required)?
                .with_continuation(stage.take().ok_or_else(required)?, &forecast)?;
            let result = json!({"decision": s.decision()});
            *stage = Some(s);
            Ok(result)
        }
        "commit" => {
            *ticket = Some(owner.as_mut().ok_or_else(required)?.commit(
                stage.take().ok_or_else(required)?,
                v["action"].as_str().ok_or_else(required)?,
            )?);
            Ok(json!({"ticket": "pending"}))
        }
        "resolve" => {
            owner.as_mut().ok_or_else(required)?.resolve(
                ticket.ok_or_else(required)?,
                &serde_json::from_value(v["feedback"].clone())?,
            )?;
            *ticket = None;
            Ok(json!({"ok": true}))
        }
        "state" => Ok(json!({"state": owner.as_ref().ok_or_else(required)?.state()?})),
        "predict" => Ok(
            json!({"probabilities": owner.as_ref().ok_or_else(required)?.predict(&serde_json::from_value(v["inputs"].clone())?)?}),
        ),
        "save" => Ok(
            json!({"save": String::from_utf8(owner.as_ref().ok_or_else(required)?.checkpoint(&v["world"])?)?}),
        ),
        "restore" => {
            let o = Owner::restore(
                &serde_json::from_value(v["config"].clone())?,
                &v["world"],
                v["save"].as_str().ok_or_else(required)?.as_bytes(),
            )?;
            *owner = Some(o);
            *stage = None;
            Ok(json!({"ok": true}))
        }
        _ => Err(io::Error::other("unknown operation").into()),
    }
}
fn main() {
    let mut owner = None;
    let mut stage = None;
    let mut ticket = None;
    for line in io::stdin().lock().lines() {
        let result = line.map_err(|e| e.to_string()).and_then(|line| {
            serde_json::from_str(&line)
                .map_err(|e| e.to_string())
                .and_then(|v| {
                    execute(v, &mut owner, &mut stage, &mut ticket).map_err(|e| e.to_string())
                })
        });
        println!("{}", result.unwrap_or_else(|e| json!({"error": e})));
        io::stdout().flush().unwrap();
    }
}
