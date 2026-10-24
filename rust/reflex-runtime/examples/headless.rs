mod worlds;
use reflex_runtime::{Execution, GameSession};
use worlds::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for kind in [Kind::Resources, Kind::Combat] {
        let configs = configs(kind, 6);
        let mut game = GameSession::new(session(kind), &configs, initial(kind, &configs))?;
        let mut executed = 0;
        let mut blocked = 0;
        for tick in 0..256 {
            for result in game.step(true, 2)? {
                match result.execution {
                    Execution::Executed { .. } => executed += 1,
                    Execution::NotExecuted { .. } => blocked += 1,
                }
            }
            if tick == 127 {
                game = GameSession::<DemoWorld>::restore(
                    session(kind),
                    &configs,
                    &game.checkpoint()?,
                )?;
            }
        }
        println!(
            "{}",
            serde_json::json!({"world":kind,"ticks":game.world().tick,"executed":executed,"not_executed":blocked,"final":game.world()})
        );
    }
    Ok(())
}
