# Adverse-condition recovery — research branch

Status: the registered long-play comparison is running; completion and quality are unverified. Stable v0.1.0 is unchanged and this candidate is **not adopted**.

## Question and limits

Can the existing stepwise stratified sampler reduce errors of the accepted fixed-H32 continuation owner, while preserving its personality, actual learning, and long-term preparation? This is a Python-only experiment. Rust, stable main/tag, and the published wheel remain unchanged.

The ten fixed development conditions are the five realized losses, the three score-positive cases with adverse conditional/reference-choice evidence, and two success guards. These are known conditions, **not independent confirmation**. `plan.json` is the canonical registration of conditions, gates, and the subsequent independent 40-condition requirement. No condition, seed, or threshold may be changed after outcomes. Stable H32 controls run fresh under the same pinned Python/NumPy/OS contract as the candidate.

## Cause evidence before the experiment

`historical_diagnosis.json` replays 2,560 actual decisions from the original H16/H32 comparison, including every owner hash and observed result. The predictions match exactly in all ten paired conditions. Thus the H32 losses cannot be explained by it receiving inferior observations or different factual opponent learning. Its planning, action sequence, and accumulated resources differ despite the same response probabilities.

In the changing world, eight-step windows after turns 32, 64, and 96 have only 4/8, 6/8, and 5/8 correct dominant-response predictions. Before the first change, the 24–31 window has 8/8 correct. Recovery is occurring, but surprise and later fluctuations remain. The truth is used only to evaluate these historical predictions, never as a planning input. These scores do not prove that a different learner would improve decisions.

The combat growth and gambling steady losses must not be treated as pure incompetence: their original conditional gains are positive despite lower realized outcomes. Gambling also changed the timing and count of completed investments. Conversely the auction steady and combat steady/care cases have both realized and conditional losses. All cases remain in the comparison.

## Candidate

Reuse `reflex.rollout_sampling.stratified_uniforms` inside the continuation portfolio: at each future step, the same three paths cover the three equal-width uniform strata. First responses remain fully enumerated. All actual RNG, horizon 32, seeds 701/702/703, future learner, retry proposals, two-step recovery, conditional estimates, risk paths, and personality selector retain their roles. No game/action names are consulted by the candidate change. Default independent sampling is unchanged.

Stratification reduces one-step sampling imbalance; it does **not** guarantee lower variance for nonlinear complete returns and cannot correct a wrong predicted law. Earlier stratification experiments under a different endpoint composition were not a successful adoption; this is a new test in the accepted H32 composition, not a reversal of those results. Reference: [Owen, basic variance reduction](https://artowen.su.domains/mc/Ch-var-basic.pdf). Related policy-evaluation motivation: [Arnold et al., RQMC policy evaluation](https://proceedings.mlr.press/v151/arnold22a.html). This experiment does not implement that paper's complete method or inherit an improvement guarantee.

## Integrity and execution

The executable source hashes and plan hash are fixed in `registration.json`. Each resolved real decision saves the complete owner and world RNG. Restoring different source/configuration is rejected. A worker separately replays every real action and learning update, preserving the original prefix after a budget handoff. Terminal results alone enter paired quality summaries; `handoff_pending` and `failed` remain distinct. Intermediate step counts are emitted every 15 seconds and checkpoint artifacts are returned after each finite 80-minute slice. This is one finite study, not a recurring schedule.

Verified before registration: new full/resumed three-step learning/decision/RNG equality; wrong-arm/changed-plan rejection; partial execution cannot create a completed result; candidate roots/probability mass/personality retained. Four rule families also matched the original stable portfolio source exactly for forecasts and decisions in a bounded compatibility check. These checks establish execution integrity, not long-term intelligence improvement. An initial harness test exposed an incorrect forecast-key lookup and a context-field assumption; both were corrected before registration.

A preliminary timing-only auction decision took about 123 seconds with three independent paths versus 312 seconds with eight. Different selected actions are not quality evidence. The registered candidate retains three paths and does not adopt a larger sample bank from that probe.

## Adoption and tradeoff

No adoption yet. The candidate gives up independent sampling between the three hypothetical paths to test more balanced response coverage at the same path/step budget. It does not give up personality parameters, learning history, rule independence, complete long-play checks, or the stable release. A development failure stops promotion; a pass still requires the registered independent forty-condition confirmation and adverse/personality review.

## Additional prediction-only diagnosis

The same replay exposes both slow novelty detection (turns 32–35) and a false switch after two rare observations (turns 73–74). Removing supplied hypotheses was therefore tested separately, before any controller change, in 16 fresh categorical streams of 128 observations. The registered independent forty-condition seeds were not used. Both models receive identical observations; the existing hazard/segmentation machinery remains unchanged.

| Observation regime | Unknown-only minus current log loss | Unknown-only minus current Brier excess |
| --- | ---: | ---: |
| Unrepresented stationary law | -0.00580 | -0.00425 |
| Change and return | +0.02067 | +0.01187 |
| Irregular changes | +0.00826 | +0.00672 |
| Represented stationary law | +0.01818 | +0.00899 |

Lower is better. Dropping the supplied hypotheses helps the wholly unrepresented law, but loses on the other regime averages. **Do not remove the hypothesis library on this evidence.** This is a completed prediction diagnostic, not evidence of improved action selection. Its source, preregistered streams, all observations and summary are in `prediction_probe/`; reproduction: `python tools/probe_adverse_prediction.py --output <fresh-directory>`. No candidate or acceptance threshold was changed by this diagnosis.

## Active execution and resumption

Frozen computation source: `9e1a467cab6fb2ea142ba216f0423cd187d0b808`. [Initial twenty-arm run](https://github.com/iwmn-9/reflex-persona-core/actions/runs/40506293826). Both arms are fresh, with identical pinned Python 3.12.14 / NumPy 2.3.5 runtime contracts. The local finite collector stores status, original downloaded artifacts and a complete-pairs-only summary in `results/adverse_recovery_20261027/`. It may resume the same worlds for at most four workflow segments, then exits; it never starts an independent study or adopts a controller.

The collector was hardened after dispatch: a dispatch with unknown outcome is not retried, and uncollected progress is null rather than an asserted zero. These are operational-only repairs. The registered worker source and its registration remain fixed at the commit above; do not rebuild or substitute it from the changing checkout. Current source verification must be run against that frozen checkout. Real failures stop continuation for diagnosis. Stable main, v0.1.0, its wheel and Rust remain untouched.

## Completed known-root sampling diagnosis

`forecast_probe/` holds 24 complete forecasts: four original public states with their exactly replayed factual owners, two samplers, and three declared model seed banks. They are a separate post-hoc mechanism check, never replacement world runs or an adoption gate.

| Known root | Stable choices over three banks | Stratified choices over three banks |
| --- | --- | --- |
| Auction steady, turn 63 | bid 1 once, bid 3 twice | bid 3 three times |
| Auction steady, turn 87 | bid 2 once, bid 3 twice | bid 2 twice, bid 3 once |
| Combat steady, turn 65 | guard twice, strike 2 once | guard twice, strike 2 once |
| Combat care, turn 32 | strike 0 three times | strike 0 three times |

At turn 32 every tested bank/sampler retains the choice made immediately before the unseen change is observed. Sampling alone does not remove that local model-information limit. At other roots action rankings can move. Agreement across banks is not proof of a correct choice; the stationary-belief reference also omits future changing beliefs and personality. No favorable bank is substituted into the long study.

The frozen research source passed the four Windows/Linux × Python 3.12/3.13 CI jobs ([run](https://github.com/iwmn-9/reflex-persona-core/actions/runs/40506293216)) and both Rust parity CI jobs ([run](https://github.com/iwmn-9/reflex-persona-core/actions/runs/40506293381)). Local collection-safety tests also passed. The two duplicate CI runs triggered by creating a source-pinning tag were cancelled deliberately; the study itself was not cancelled.

The immutable compute ref is also available as `research/adverse-recovery-20261027`. The collector uses that tag for `workflow_dispatch` and verifies its commit against the frozen SHA. This follows the documented branch/tag requirement and leaves stable `v0.1.0` unchanged.

On completion, the finite collector also writes `outcome_review.json`: 32-step phase gains, conditional/residual accounting, preparation/action counts, available perfect-information reference accounts, and selected-forecast versus actual-replanning residuals. The accountant was checked against all eight published adverse-review records and reproduced every recorded difference within 1e-10. This evaluator-only addition does not change the worker or acceptance gate.

## Structural hypothesis

Source inspection identifies a separate approximation: each first action currently receives its own retry continuation plus the learning owner with short recovery. Changing the first action therefore also changes which retry policy is offered in the tail. A good first action followed by a different productive policy may be underrepresented. Comparing first actions against a shared bank of feasible continuations could distinguish this proposal-coverage bias from sampling variability. This is a separate research variation, not part of the registered stratified candidate. Its diagnostic status is recorded below; do not infer an improvement guarantee or add a game-specific sequence.


## Additional cross-platform integrity diagnosis

Fifteen first-slice checkpoints (706 actual decisions) were also replayed on Windows. All physical transitions, actual random draws, final RNG and fixed personality axes matched. Four saved final owners had tiny differences only in numeric evidence-gain entries; the largest absolute difference was 5.551115123125783e-17. Their final prediction state and discrete adoption states matched. Exact owner hashes therefore do not match between these Linux and Windows arithmetic builds, even with the same Python/NumPy version numbers. This diagnostic does not recompute planning choices or establish cross-platform decision identity at every prior turn.

`platform_replay.json` records every mismatch, input checksum and runtime. No tolerance was added to the original worker. Its exact same-runtime action/learning replay, source binding and save/restore checks remain mandatory. No histories were replaced, and completed-study accounting does not substitute a Windows owner for a Linux owner.

## Separate continuation-coverage experiment

The shared-bank hypothesis is now implemented only as an opt-in research setting, `continuation_scope="shared"`, on this branch. Stable/default `root` behavior reproduced the frozen computation exactly in bounded checks across all four families. The ongoing stratified study still executes its original pinned source and has no shared bank.

The shared setting compares every real first action against all the same currently public retry targets, preserving legality/progress checks, owner learning and the full personality selector. It retains the original diagonal and reflex trajectories. Local integration tests verified those paths, their values and actual-root bindings. A test harness initially confused synthetic plan IDs with proposal IDs; that assertion was corrected without changing controller behavior.

Two separate diagnostics were fixed before execution: six original public states/owners under H32 and seeds 701/702/703, and the existing five-operation preparation mechanism played for 128 turns with growth/care owners. The latter specifically checks whether broader hypothetical tails cause perpetual deferral. Neither is independent confirmation or a basis for adoption. The connection follows the idea of evaluating each action against a common policy bank in [Parallel Rollout](https://engineering.purdue.edu/~givan/papers/deds01.pdf); its improvement theorem is not claimed for finite-horizon, learned-model personality decisions.


### Shared-bank long-play result: reject the simple expansion

The deterministic preparation check completed all four 128-turn runs (512 actual decisions), including exact replay of every factual owner and unchanged source/personality checks. Both growth and care owners ended with cash 236 under the existing root-specific bank and 160 under the shared bank. Existing policy began preparation at turn 0 and performed 125 preparation operations. The shared bank collected immediate income 98 times before starting preparation, performing only 30 preparation operations.

At the first decision its estimated future purpose was higher (0.140625 versus 0.13541666666666666), but every new decision again selected “income now, preparation next.” The rolling cutoff moved the promised preparation forward until the actual world endpoint forced a shrinking horizon. This is an observed planning-consistency failure in this experiment, not a personality sacrifice or stochastic bad luck. Reject simple common-bank expansion as a drop-in improvement. Do not promote it even if a known gaming root receives a higher estimate. This does not prove all possible shared-policy methods fail.

`shared_longplay/` contains all traces, checkpoints, the original plan, checksums and a source archive verified against the registered computation files. Reproduction uses `python tools/probe_shared_longplay.py --output <fresh-directory>` on this research source. The first six-root coverage diagnostic separately failed because its harness bypassed public response-learning fact integration. Fact-consistency validation correctly rejected it; the harness was repaired to use the public DecisionLoop context and the identical cases/seeds were rerun. That harness failure is not a completed result or a failure of the ongoing frozen sampling study.
