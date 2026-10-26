# Factual learning owner and fixed-H32 selection boundary

`reflex-owned` owns the person who actually acts: fixed personality/policy,
intent, bounded selected-outcome experience, segmented public-event prediction,
prequential diagnostic evidence, need pressure, and observed purpose progress.
It delegates numeric preference selection to `reflex-immediate`. It contains no
game rules, model-specific action rankings, hidden-world process, or real-world RNG.

The supported composition is the Python stable owner's **public-conditional
profile**: `EventReader(learning='segmented', key='event-response')`, 64-node read
gate, default `NeedPressure`/`ProgressWatch`, default 64-entry outcome/evidence
memories, no social or route overrides, and no certified-exact masks. Bindings
explicitly identify which outcome features may be learned. Other Python
`DecisionLoop` configurations are not silently interpreted as this profile.
The owner accepts 1..8 legal public responses, at most 4 declared factors and
16 library models, and a 2..512-event history window. Bounded identifiers in the
learning schema use at most 128 Unicode characters. Continuation packets may
contain up to 256 modeled outcomes per candidate. These are explicit API limits.

## One execution unit

1. Construct `Owner` from an application-supplied `OwnerConfig`.
2. `prepare(input, false)` produces a disposable `Stage` without changing memory.
3. For long decisions, supply a complete modeled continuation through
   `with_continuation(stage, forecast)`. The modeled purpose corridor, cumulative
   return units, corrected linear means, risk, and personality tier are applied
   in Rust. The selected result is a legal **real root**, not a plan identifier.
4. Execute that action. `commit(stage, executed_action)` returns an opaque receipt.
5. `resolve(receipt, actual_feedback)` validates every feedback channel before
   updating the factual learner. A stochastic failure is still executed. Missing
   feedback is explicitly `None`, not an invented failure or success.
6. `checkpoint(world_record)` saves only a resolved owner. Restore with the
   expected configuration and exact public world record. Save your actual world
   RNG and other host state with that record. The checksum detects corruption;
   it is not authentication.

Wrong-owner/replayed receipts, stale proposals, changed configuration, invalid
observed conditional effects, and partial saves are rejected. `fork_model()`
creates a separate learner; imagined updates cannot mutate the factual owner.
Restore creates a new process-local owner identity, invalidating old proposals.
The immediate crate/runtime APIs remain available and are not changed to learn.

For a population and world transaction, use
`reflex_runtime::learning::LearningSession<World>`. Its `Fixed32` control requests
a forecast for every scheduled decision. A failed generator never silently
substitutes reflex behavior. All actor feedback and the staged next world must
validate before the session commits any of them. Host transitions must be pure;
external engine/file effects occur only after successful `step`.

## Python remains the continuation generator

This is **not a Python-free implementation of the full H32 planner**.
`reflex.native_owned.NativeOwnedBoundary` imports the current native learned state
into a detached Python owner and invokes the frozen `FixedHorizonOwner` generator:
H32, seeds `(701,702,703)`, recovery cache 64, conditional purpose estimate,
coherent cumulative preference, and the existing continuation portfolio/recovery.
The returned typed forecast is selected independently by Rust. No separate live
Python NPC is maintained. Early terminal worlds may report a shorter effective
prefix; they do not receive invented post-terminal actions.

The boundary is an in-process Python API for a host bridge, not a subprocess
service, engine binding, or native port of search. The host owns communication and
public game callbacks. The test transport `examples/owned_verify.rs` is a JSON-line
fidelity probe, not a supported network protocol. Pure Rust immediate/learned
decisions work without Python; this supplied H32 generator requires Python.

Experiments, alternative forecast generation/replanning, social memory, other
predictors, route selection and new value proposals stay in Python. A future
native search implementation must meet its own equivalence gate before replacing
this explicit boundary. Rust migration does not itself improve intelligence or
establish real-time planning performance.

## Verification

```sh
cargo test --locked --manifest-path rust/Cargo.toml --workspace
cargo build --locked --manifest-path rust/Cargo.toml -p reflex-owned --example owned_verify
python tools/verify_rust_owned.py --output FRESH_HISTORY --steps 160 --profiles 4
python tools/verify_rust_owned_prediction.py --output FRESH_FACTORS
python tools/run_rust_owned_parity.py --output FRESH_H32_INTEGRATION --workers 4
```

The finite integration run retains all 16 rule-family/personality combinations,
164 real decisions each, H32 selection after 129 observed decisions, per-step
learning comparison and save/restore, and separate-process restarts. It compares
two selectors against one immutable Python-generated forecast; it does not claim
two independently implemented search engines. Discrete choices/state must match;
numerical forecast/diagnostic differences are bounded at `2e-12`. Native saves
must resume bit-exactly. This gate is separate from the original Python 40-pair
quality study and never changes its criteria or results.

The completed comparison and its exact scope are recorded in
[the integration report](../../evidence/rust_port/owned/REPORT.md).
