# Adverse-condition recovery — research branch

Status: registered, long-play execution pending. Stable v0.1.0 is unchanged and this candidate is **not adopted**.

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
