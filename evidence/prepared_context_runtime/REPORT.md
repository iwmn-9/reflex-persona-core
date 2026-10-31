# Reuse an already resolved owner context

The private deterministic, one-owner H1 continuation can reuse the context
which the surrounding decision lifecycle has just resolved. It verifies the
complete bound owner and request; other owners or requests use the original
preview. Public planner calls retain their independent preview and mismatch
checks. The supplied prepared context is separate from the mutable planner
input. There is no global context, persistent cache, record change, new reward,
or changed personality rule.

This removes a repeated observation/selection lifecycle, not future learning.
Every continued owner still decides, observes and updates as before.

| Corresponding workload | Prior CPU seconds | Current CPU seconds |
| --- | ---: | ---: |
| Known H32, three samples, exact original factual owner | 180.84375 | 158.84375 |
| Seven-case exact continuation matrix | 34.0 | 28.765625 |

The H32 result is 12.17% less CPU than the previous qualified runtime and
79.85% less than the original 788.46875-second implementation (about 4.96×).
Current wall time was 254.82 seconds under other local work; it is not a
general frame-time guarantee. The original 96 factual updates and the entire
saved forecast, choice and modeled owner continuation match exactly.

The seven-case matrix covers four stochastic rule families, two preparation
durations and delayed yield. All semantic outputs and modeled final owners
match. Only actual computation counters are excluded from that comparison.
Source hashes were checked before and after both final measurements.

Affected regression coverage totals 53 distinct passing methods. The first
43-method run had 42 passes and an existing sibling-test import error; that
method passed with the required `tests` Python path, together with ten gain
continuation/owner methods. This is targeted coverage, not a newly completed
full repository suite. New boundary tests cover experienced owners across four
families and four personas, exact feedback and restoration, foreign request/
owner fallback, mutated planner-input rejection, stale owner rejection, and
exclusion of stochastic or multiple-owner calls. The previous full-runtime
qualification remains separate.

The running original cohort and gain-mechanism cohort retain their original
sources. This runtime is an explicitly qualified candidate for subsequent
registered work. It is not evidence that long-play quality improved, and it is
not adopted into the stable release or Rust.

Evidence: [verification](verification.json), [test accounting](tests.json),
[all comparison data and scripts](verification_data.zip), [fixed source](source.zip).
