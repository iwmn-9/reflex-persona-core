//! Test protocol, not a production network API. Holds actors and opaque batches
//! across requests; exercises the same public API as an embedded game host.
use reflex_runtime::{immediate::Config, *};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Init {
        session: SessionId,
        configs: Vec<Config>,
        revision: u64,
    },
    Prepare {
        revision: u64,
        inputs: Vec<Input>,
        stochastic: bool,
        workers: usize,
    },
    Resolve {
        next_revision: u64,
        receipts: Vec<Receipt>,
    },
    Checkpoint {
        world: String,
    },
    Restore {
        session: SessionId,
        configs: Vec<Config>,
        checkpoint: String,
    },
}
#[derive(Default)]
struct Server {
    runtime: Option<Runtime>,
    pending: Option<Batch>,
}
impl Server {
    fn run(&mut self, c: Command) -> Result<Value> {
        match c {
            Command::Init {
                session,
                configs,
                revision,
            } => {
                self.runtime = Some(Runtime::new(session, revision, &configs)?);
                self.pending = None;
                Ok(json!({"initialized":true}))
            }
            Command::Restore {
                session,
                configs,
                checkpoint,
            } => {
                let (r, world) = Runtime::restore(
                    session,
                    &configs,
                    "python-continuing-world-v1",
                    checkpoint.as_bytes(),
                )?;
                self.runtime = Some(r);
                self.pending = None;
                Ok(
                    json!({"world":String::from_utf8(world).map_err(|_|Error::Invalid("test world is not UTF8"))?}),
                )
            }
            Command::Prepare {
                revision,
                inputs,
                stochastic,
                workers,
            } => {
                if self.pending.is_some() {
                    return Err(Error::Invalid("unresolved test batch"));
                }
                let b = self
                    .runtime
                    .as_ref()
                    .ok_or(Error::Invalid("initialize first"))?
                    .prepare(revision, &inputs, stochastic, workers)?;
                let result = json!({"decisions":b.decisions().collect::<BTreeMap<_,_>>()});
                self.pending = Some(b);
                Ok(result)
            }
            Command::Resolve {
                next_revision,
                receipts,
            } => {
                let r = self
                    .runtime
                    .as_mut()
                    .ok_or(Error::Invalid("initialize first"))?;
                let result = r.resolve(
                    self.pending.take().ok_or(Error::Invalid("prepare first"))?,
                    next_revision,
                    &receipts,
                )?;
                Ok(
                    json!({"resolutions":result,"states":r.actors().map(|(id,a)|(id,a.state())).collect::<BTreeMap<_,_>>()}),
                )
            }
            Command::Checkpoint { world } => {
                if self.pending.is_some() {
                    return Err(Error::Invalid("test save requires resolved batch"));
                }
                let r = self
                    .runtime
                    .as_ref()
                    .ok_or(Error::Invalid("initialize first"))?;
                let b = r.checkpoint(
                    r.world_revision(),
                    "python-continuing-world-v1",
                    world.as_bytes(),
                )?;
                Ok(json!({"checkpoint":String::from_utf8(b).expect("JSON UTF8")}))
            }
        }
    }
}
fn main() -> io::Result<()> {
    let mut server = Server::default();
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        let response = if line.len() > MAX_SAVE_BYTES {
            Err(Error::Invalid("test request too large"))
        } else {
            serde_json::from_str::<Command>(&line)
                .map_err(Error::from)
                .and_then(|c| server.run(c))
        };
        let result = match response {
            Ok(value) => value,
            Err(e) => json!({"error":e.to_string()}),
        };
        writeln!(output, "{result}")?;
        output.flush()?;
    }
    Ok(())
}
