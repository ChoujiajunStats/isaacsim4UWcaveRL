# Exit-control diagnosis and bounded PPO repair

The previous equal-additional-budget continuation completed, but both final
policies failed the unchanged 90-episode full-route evaluation. More training
transitions and curriculum promotion were not evidence of a usable navigator.

## Frozen action diagnosis

`scripts/diagnose_navigation_actions.py` compares checkpoints without learning:

- Starts: 8 m from the exit along the reference route, saved training frontier,
  and/or the real entrance. Frontiers are frozen, including ignored workers'
  automatically restarted episodes.
- Actions: standard deterministic inference, native training-time stochastic
  inference, and an experimental deterministic expectation of the bounded
  Gaussian action (analytic clipping for PPO; quadrature for FlashSAC).
- One episode per worker, six workers per cave by default. Initial poses are
  checked for exact equality across action modes. This does not assert that
  rendered pixels or stochastic trajectories are bit-identical.
- PPO advances recurrent memory exactly once per step and resets on episode
  boundaries. Actor weights and normalization buffers must remain unchanged.
- Terminal positions, goal distances, contacts, route deviations, applied
  action saturation and standard deviations are captured before auto-reset.
  Privileged measurements in the report never enter the actor.

The final PPO checkpoint `model_932.pt` produced zero successes in all nine
18-episode cases, including all three 8 m action variants. That rejects the
simple hypothesis that merely retaining exploration noise or taking a bounded
action expectation fixes this checkpoint. It does not establish the only root
cause. The full report is `outputs/action_tuning_v1/ppo_diagnostic.json`.

These are small diagnostic cohorts, not statistical acceptance tests. The
alternative bounded-action expectation is **not** installed into either
algorithm's standard evaluation or training path.

FlashSAC's completed six-case diagnostic is
`outputs/exit_repair_v1/flash_diagnostic.json`. Standard deterministic inference
failed all 18 short starts and all 18 frontier starts. Stochastic inference
achieved 1/18 in each cohort. Bounded-mean inference achieved 2/18 short starts
and 6/18 frontier starts (all six easy workers). The easy frontier is already
the full reference-route distance, but this small cohort and alternative
inference rule do not establish multi-cave acceptance. They warrant a separate
true-entrance, balanced confirmation.

## Repair experiment

`scripts/run_navigation_repair.py` creates a new, finite experiment directory,
optionally diagnoses the frozen FlashSAC checkpoint, then trains PPO and runs
short-route diagnostics and the standard 90-episode full evaluation.

The corrected repair experiment under `outputs/exit_repair_v2/` **completed**.
All stages exited normally; `accepted` is **false**. It was launched in tmux
session `cave_exit_repair_v2`. Its `status.json` records the completed stages.

The first queue, `exit_repair_v1`, completed FlashSAC diagnostics but its PPO
launch failed before training: Isaac Lab's runtime-typed config updater would
not replace a `None` default with a float. The noise-reset option now uses the
float sentinel `0.0` (disabled). The corrected queue skips the already completed
FlashSAC diagnosis; the original failure and reports are preserved.

| Setting | Repair recipe |
| --- | --- |
| Initialization | PPO 932 weights and normalizers; **fresh optimizer and curriculum** |
| New training | 96 workers × 64 steps × 96 updates = 589,824 transitions |
| Initial exploration std | Reset to 0.4; remains learnable |
| Learning rate / entropy coefficient | Fixed 0.0001 / 0.001 |
| Frontier promotion | Start 4 m; 20-result window; 80% success; growth ×1.25 |
| Rehearsal | 50% of starts sampled uniformly from 2 m to min(12 m, frontier) |
| Other starts | Current per-cave frontier |

Rehearsal outcomes record their actual starting distance and therefore cannot
promote a harder frontier. Rehearsal is disabled by default and consumes no
extra RNG draws when disabled. Sampling settings are saved with PPO navigation
metadata; a normal resume rejects a mismatched sampling configuration. Changing
it uses an explicit weights-only warm start. Warm starts record their source
checkpoint and reset iteration numbering; they are not optimizer/replay resumes.

This is a bundled repair hypothesis, not an ablation that attributes improvement
to one hyperparameter. It targets fragile exit control and the frontier-only
training distribution; success is not assumed before evaluation.

Actor inputs, CNN/GRU architecture, six actions, dynamics, rewards, collision
thresholds, 2.25 m route-tube bounds and actual-exit success remain unchanged.
No reference-route commands or cave IDs are added to the deployed actor. The
full evaluation still starts at the real entrances and uses a 300 s horizon,
fixed balanced quotas and deterministic standard PPO inference. Acceptance
requires at least 30 episodes per cave, success ≥80%, collisions ≤15%, SPL ≥0.5.

The queue runs one simulator at a time with the existing owned-process resource
supervisor. Training/evaluation stages are capped at 30 minutes each. Prior
models are neither overwritten nor removed. To stop this queue and its owned
child without advancing to the next stage:

```bash
tmux send-keys -t cave_exit_repair_v2 C-c
```

`plan.json` records the exact command, source checkpoint hash and implementation
hashes. `ppo_train.log`, `short_diagnostic.json`, `full_evaluation.log` and the
new run's `evaluation_full.json` separate optimization, diagnostics and actual
acceptance. A workflow status of `complete` does not imply `accepted: true`.

## Completed result (2026-09-09)

The repair run is
`logs/rsl_rl/underwater_cave_multinav/2026-09-09_12-56-59_exit_rehearsal_20260909_125652/`.
Its final `model_95.pt` contains 589,824 new transitions, fresh-optimizer state,
rehearsal settings and source-model provenance. Optimization took 579.07 s.
Final frontiers were 6.25/6.25/5.0 m and learned exploration std stayed near 0.398.

| Test | Easy | Medium | Hard |
| --- | --- | --- | --- |
| 8 m, deterministic | 0/6 | 0/6 | 0/6 |
| 8 m, stochastic | 2/6 | 0/6 | 0/6 |
| Real entrance, deterministic | 0/30 | 0/30 | 0/30 |
| Full-route collisions | 0/30 | 30/30 | 30/30 |
| Full-route bounds failures | 30/30 | 0/30 | 0/30 |
| Full-route mean travelled distance | 18.70 m | 5.19 m | 5.95 m |

Full-route SPL remains zero. This recipe is **not** an improved deployable
checkpoint: it did not repair deterministic exit control, and easy/medium full
trajectories were shorter than PPO 932. Keep both checkpoints for analysis;
do not automatically continue the repair checkpoint merely because it is newer.
The 5090 handoff should prioritize understanding the local-control/observation
problem and a balanced confirmation of the FlashSAC bounded-mean signal before
spending another large unchanged training budget.

Validation at handoff: 43 CPU unit tests across checkpointing, rehearsal,
evaluation allocation/gates, action expectations, official FlashSAC replay,
resource supervision and transfer packaging passed. The visual-policy and
44-file cave-checksum contracts also passed. The real-simulator repair and both
frozen-policy diagnostic suites completed; completion is not navigation success.
