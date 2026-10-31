# Equivalent response preparation runtime

The owned rollout now calculates the public response distribution once per
modeled node, on a fully detached owner. All action-specific conditional
outcomes and future learner updates are retained. External kernels keep their
existing execution path. This relies on the existing pure-model callback
contract; it adds no persistent cache, runtime dependency, or gameplay rule.

Seven complete forecast/path/owner comparisons were exact. On the archived
96-step experienced owner, the H32 forecast, choice, actual feedback and restored
owner also matched exactly. CPU time was 191.390625 seconds before and 169.5
seconds after (11.44% less). The small matrix was slightly slower, 25.296875 to
25.859375 seconds, so this is not a uniform speed guarantee. Elapsed times were
measured on a busy PC and are not used as the speed claim.

All 64 affected regression tests and nine additional targeted methods passed.
The new method compares four rule families across three world-model modes.
One existing sibling-import failure was rerun with the test directory on
PYTHONPATH; no implementation change was needed.

`verification.json` binds the committed implementation, 602 Python source files,
raw exact comparisons and test output. This qualifies an equivalent runtime,
not the long-horizon candidate's scientific quality. Existing registered workers
were not changed. Full development and independent confirmation gates remain.
