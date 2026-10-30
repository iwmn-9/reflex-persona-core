# Fixed sample-budget comparison

Completed and audited on 2026-10-30: **10 pairs, 20 executions, 2,560 actual decisions**. Increasing the existing H32 posterior path bank from three to eight samples did **not** pass the unchanged development gate. Gambling's conditional return still decreases. This candidate is not adopted and does not proceed to independent confirmation. Stable v0.1.0 and Rust remain unchanged.

The authoritative design is [plan.json](plan.json), with computation bytes bound by [registration.json](registration.json) to commit `19462563a54cd36f43ff14679bf0e731de687cfe` and tag `research/adverse-sample-budget-20261028`. The candidate changes only outer path seeds `[701,702,703]` to the previously fixed `[701,702,703,704,705,706,707,708]`. H32, H2 future recovery, posterior world assumptions, personality, factual learning and decision rules are unchanged. There is no game-name or action-name branch in the controller.

## Complete long-play result

Mean differences from stable H32, in each environment's registered normalized units:

| Environment | Realized final score | Conditional return |
| --- | ---: | ---: |
| Auction | +0.00640869140625 | +0.01578369140625 |
| Combat | +0.03125 | +0.028125 |
| Gambling | +0.01822916666667 | **−0.01223958333333** |
| Resources | +0.00911458333333 | +0.00638020833333 |

Realized score improves in seven conditions, ties in two and worsens in one. Conditional return improves in eight, ties in one and worsens in one. These are the ten selected development conditions, not ten independent discoveries or the reserved forty-condition confirmation. Every condition, including both kinds of loss, remains in [the complete archive](completed_local/verification.json), [outcome review](completed_local/outcome_review.json) and [full traces](completed_local/runs.jsonl.gz).

All 425 registered source files, original world draws, fixed personalities, full factual owner histories, saved final owners and replay receipts passed the unchanged audit. Original Linux executions were continued on this PC in the recovered original Linux/Python/NumPy arithmetic environment. The unnecessary fresh Windows replicas remain incomplete and are not substituted. A subsequent Windows update reboot was recovered from the same saved prefixes; no seed, source, history, threshold or failed condition was replaced. The archive states the distinction between exact factual replay and re-running every historical planning search. GitHub Actions are disabled; computation is local.

The complete [prediction-error review](prediction_error_review.json) finds exactly the same paired one-step response predictions in all ten conditions. Thus this experiment improves some decisions and preparation without improving the factual response learner. Proper response scores do not certify the whole imagined future-world model.

## What the remaining losses mean

In auction profile 3, realized score decreases by 2 currency-equivalent units while conditional return increases by 33.1 and the reference-choice metric also improves. The realized-minus-conditional residual is −35.1. A lower realized outcome alone therefore does not establish worse expected judgment. It still triggers the predeclared loss investigation and remains in the gate evidence.

In gambling profile 1, realized currency increases by 7 and matured investments increase from 17 to 18. Conditional currency decreases by 4.7: the extra investment contributes +1.6 relative to a modal bet, while three extra bets contrary to the evaluator's true modal response contribute −6.3. Every candidate bet follows its own learned modal belief. The actor never receives the evaluator's law. Better preparation and inaccurate beliefs coexist; the unfavorable conditional result cannot be dismissed as merely unlucky realized draws.

The [predeclared loss-root procedure](loss_root_plan.json) retains every negative final-score, conditional or existing reference-choice condition. It selects the first action divergence and the beginning of the worst 32-turn realized-gain phase, deduplicating roots. The [resolver](loss_root_resolver.py) verifies the complete archive before resolving them. All four roots and all 24 evaluations finished: the 12 original gambling evaluations were reused exactly and 12 auction evaluations were run locally. Four factual prefixes and all four original eight-sample forecast controls reproduce exactly. See [proof](root_diagnosis_all/verification.json), [all forecasts](root_diagnosis_all/rows.json.gz) and [root review](root_diagnosis_all/review.json).

| Fixed factual root | Three samples, banks 701 / 9101 / 9201 | Eight samples, same bank starts |
| --- | --- | --- |
| Auction profile 3, turn 0 | bid 3 / bid 3 / bid 2 | bid 2 / bid 3 / bid 2 |
| Auction profile 3, turn 96 | bid 1 / bid 2 / bid 2 | bid 2 / bid 2 / bid 2 |
| Gambling profile 1, turn 6 | bet 2 / bet 0 / bet 0 | bet 0 / bet 0 / bet 0 |
| Gambling profile 1, turn 64 | bet 2 / bet 2 / bet 2 | bet 2 / bet 2 / bet 2 |

Sampling sensitivity remains at auction turn 0 even with eight paths. Eight-sample choices agree at the other three roots, but agreement is not correctness. The review retains the full estimated root-value rankings and the personality-selected actions separately: they need not be identical. In particular, at gambling turn 6 with bank 701, bet 2 has a slightly higher raw estimated purpose than the chosen bet 0. It would be incorrect to describe all eight-sample raw rankings as identical or to infer that more samples alone repair belief/selection limitations.

These three diagnostic banks are a sensitivity check, not a calibrated confidence interval. They were fixed before the completed outcomes and cannot replace the registered controller. Local agreement does not prove long-play convergence; one-step model errors, future-world assumptions, personality selection and sampling can all matter. None of these known-root evaluations counts as an independent world.

## Decision and retained scope

The useful result is a complete separation of two problems: extra integration paths help some long preparation and local stability, but cannot correct unchanged response beliefs and do not eliminate all auction sensitivity. The discarded proposal is adopting eight samples as a sufficient general repair. No persona component is removed and no favorable diagnostic seed is selected.

Existing separate Python studies test supplied-prior transfer and consistent use of the actor's current prediction in immediate and long-horizon selection. They must finish their own registered ten-condition comparisons and all loss/personality reviews. Only a passing candidate may enter the original reserved forty-condition comparison against both original H16 and stable H32. This failed study is retained alongside them, not replaced by their results.

## Reproduction and validation

The complete archive contains frozen sources, original plan/registration, every checkpoint/result/status, condensed traces, comparisons and checksums. `python tools/review_adverse_predictions.py --evidence <complete-audit-directory> --output <fresh-output.json>` derives proper-score diagnostics after checksum validation. Root execution uses the unchanged `probe_frozen_sample_budget.py`; its source, finite driver, raw outputs and proofs are preserved in `root_diagnosis_all/execution.zip`, with the collector beside it.

Pre-registration checks covered eight-sample full-versus-resumed owner/RNG equality, fixed personality/root retention, normalized response mass, fixed-bank rejection, incomplete results and collector/audit failure handling. The frozen numeric CI previously passed Windows/Linux with Python 3.12/3.13; both Rust jobs also passed. This is historical evidence, not a new Actions run. The standalone prediction reviewer additionally rejects incomplete/duplicate/unpaired evidence and tampered input. No active scientific file or adoption threshold was changed by the later audit tools.
