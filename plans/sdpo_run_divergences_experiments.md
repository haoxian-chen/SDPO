# Run SDPO with New Divergence Types on LCB + ToolUse

Goal: take the 6 f-divergences added in `sdpo_add_divergences.md`
(`reverse_kl`, `forward_kl`, `jsd`, `improved_forward_kl`, `improved_jsd`)
and produce sweep scripts that train SDPO on **LiveCodeBench v6** and on
**ToolUse**, leaving the existing recipes untouched.

## What's there today

- `experiments/rich_feedback/run_sdpo.sh` — LCB sweep, uses the **full-logit**
  branch (`alpha=1.0`, `distillation_topk=20`). Does not exercise the new
  `divergence_type` knob.
- `experiments/generalization/run_sdpo_all.sh` — ToolUse + SciKnowEval sweep,
  also full-logit (`alpha=0.5`, `distillation_topk=100`). Also does not
  exercise `divergence_type`.

Both scripts target Slurm. Per the user, the cluster is multi-GPU; no
`run_local_*.sh` is needed.

## What needs to change

The new `divergence_type` field lives on the **sampled-token** branch of
`compute_self_distillation_loss` (active when
`full_logit_distillation=False`). The existing recipes are on the full-logit
branch, so to use it we need a small set of overrides:

- Set `actor_rollout_ref.actor.self_distillation.full_logit_distillation=False`.
- Set `actor_rollout_ref.actor.self_distillation.divergence_type=<one of 5>`.
- Stop passing `actor_rollout_ref.actor.self_distillation.alpha=...` (ignored on
  this branch) and `actor_rollout_ref.actor.self_distillation.distillation_topk=...`
  (only meaningful for full-logit, the teacher forward path skips topk
  extraction when `full_logit_distillation=False` per
  `verl/workers/actor/dp_actor.py:778-779`).

Everything else in each recipe (LR, mini-batch, teacher EMA rate,
rollout-IS correction, val n, etc.) stays as-is, so the only intentional
delta vs the current published recipe is the loss formulation.

## Deliverables

Two new sweep scripts, side-by-side with the originals (do not modify the
originals):

### 1. `experiments/rich_feedback/run_sdpo_divergences.sh`

Copy of `run_sdpo.sh` with these edits:

- Replace the comment + `ALPHAS=(1.0)` block (`run_sdpo.sh:42-44`) with:
  ```bash
  # One of: reverse_kl, forward_kl, jsd, improved_forward_kl, improved_jsd
  DIVERGENCE_TYPES=(reverse_kl forward_kl jsd improved_forward_kl improved_reverse_kl improved_jsd)
  ```
- Replace the inner sweep header `for ALPHA in "${ALPHAS[@]}"` with
  `for DIVERGENCE_TYPE in "${DIVERGENCE_TYPES[@]}"`.
- Replace the `-alpha${ALPHA}-` segment of `EXP_NAME` with `-div${DIVERGENCE_TYPE}-`.
- In `ARGS=` (`run_sdpo.sh:118-132`):
  - **Remove** `actor_rollout_ref.actor.self_distillation.distillation_topk=20`.
  - **Remove** `actor_rollout_ref.actor.self_distillation.alpha=$ALPHA`.
  - **Add** `actor_rollout_ref.actor.self_distillation.full_logit_distillation=False`.
  - **Add** `actor_rollout_ref.actor.self_distillation.divergence_type=$DIVERGENCE_TYPE`.
- Bump `trainer.group_name` to `SDPO-rich-feedback-div` so wandb groups the
  new runs separately from the published `alpha=1.0` baseline.

Result: a 6-way sweep on `datasets/lcb_v6` × `Qwen/Qwen3-8B` × LR `1e-6`,
matching the rich-feedback recipe in every dimension except the loss.

### 2. `experiments/generalization/run_sdpo_tooluse_divergences.sh`

Copy of `run_sdpo_all.sh` with these edits:

- Narrow `DATA_PATHS` (`run_sdpo_all.sh:18-23`) to **just** ToolUse:
  ```bash
  DATA_PATHS=(
      "datasets/tooluse"
  )
  ```
  (Drops the four SciKnowEval entries, per the user wanting LCB + ToolUse only.)
- Replace the comment + `ALPHAS=(0.5)` block (`run_sdpo_all.sh:40-42`) with:
  ```bash
  DIVERGENCE_TYPES=(reverse_kl forward_kl jsd improved_forward_kl improved_reverse_kl improved_jsd)
  ```
- Same loop / EXP_NAME / ARGS edits as in deliverable #1 (drop `alpha`,
  drop `distillation_topk`, add `full_logit_distillation=False` and
  `divergence_type=$DIVERGENCE_TYPE`).
- Bump `trainer.group_name` to `SDPO-generalization-div`.

Result: a 6-way sweep on `datasets/tooluse` × `Qwen/Qwen3-8B` × LR `1e-5`
= **6 Slurm jobs**. (Olmo dropped per scope; uncomment in the
`MODEL_PATHS=(...)` block if you want to add it back.)

## Hyperparameters that carry over

These come from the source recipes and are unchanged by the divergence swap.
Listing them so it's clear no extra retune is being done in this pass.

| Setting | LCB recipe | ToolUse recipe |
| --- | --- | --- |
| `train_batch_size` | 32 | 32 |
| `rollout.n` | 8 | 8 |
| `ppo_mini_batch_size` | 1 (on-policy) | 32 |
| `total_training_steps` | 80 (overridden in ARGS) | 80 (overridden in ARGS) |
| `optim.lr` | 1e-6 | 1e-5 |
| `optim.lr_warmup_steps` | 0 | 10 |
| `teacher_update_rate` | 0.01 | (config default 0.05) |
| `rollout_correction.rollout_is` | token | token |
| `dont_reprompt_on_self_success` | True | True |
| `val_kwargs.n` | 4 | 16 |
| Models | Qwen3-8B | Qwen3-8B |
| Slurm: nodes / GPUs / mem / time | 1 / 4 / 460G / 12h | 1 / 4 / 460G / 2h |

## How to run

After creating the two scripts above:

```bash
# preview sbatch commands without submitting
bash experiments/rich_feedback/run_sdpo_divergences.sh --dry-run
bash experiments/generalization/run_sdpo_tooluse_divergences.sh --dry-run

# submit
bash experiments/rich_feedback/run_sdpo_divergences.sh
bash experiments/generalization/run_sdpo_tooluse_divergences.sh
```

Total jobs: **6 (LCB) + 6 (ToolUse) = 12 Slurm jobs**.

## Sanity controls

- `divergence_type=reverse_kl` on the sampled-token branch is the bit-exact
  port of the prior `alpha=1.0, full_logit_distillation=False` form (see
  `sdpo_add_divergences.md` backwards-compat section). It is **not**
  bit-exact with the published `alpha=1.0, full_logit_distillation=True`
  full-logit recipe — that path mixes top-20 + tail bucket. So the
  `reverse_kl` run in each sweep is a baseline for the new sampled-token
  setup, not for the published full-logit one. If you want to compare new
  vs. published, treat the existing wandb runs from `run_sdpo.sh` /
  `run_sdpo_all.sh` as the reference.
- The sampled-token branch is cheaper per step (no top-k extraction on the
  teacher), so expect faster wall-clock per step than the full-logit
  baseline.

## Caveats

- **Loss magnitudes differ across divergences.** Per `sdpo_add_divergences.md`
  the per-token signal is bounded only by the `log_u ∈ [-10, 10]` clamp; JSD
  variants are typically smaller in magnitude than reverse/forward KL, so
  the effective update size for the same LR will differ. SDPO has no
  loss-scaling knob (`actor.kl_loss_coef` only scales the optional
  reference-policy KL, off by default). If `improved_jsd` or `jsd` runs
  look under-trained, the right knob is `actor_rollout_ref.actor.optim.lr`,
  not the KL coef.
- **Forward KL on a sampled-token estimator can be high-variance.** The
  `-u * log_u_clamped` term grows quickly when teacher and student diverge.
  If you see grad-norm spikes in early steps, the `log_u ∈ [-10, 10]` clamp
  already bounds `u ≤ 22026`, but the LR may still need to be lower than
  for `reverse_kl`. Consider adding a sub-sweep on LR for these two.
- **Bit-exact regression test recommended.** If you want extra confidence
  before submitting, run a single-step pilot with `divergence_type=reverse_kl`
  and verify gradients match the prior `alpha=1.0, full_logit_distillation=False`
  output (the new branch is bit-exact with that — see
  `sdpo_add_divergences.md`). This was *not* added as an automated test in
  the implementation pass.

## Out of scope (for this experiment plan)

- No new model added — the ToolUse sweep keeps both Qwen3-8B and Olmo-3-7B,
  the LCB sweep keeps only Qwen3-8B as the original recipe does.
- No LR sweep per divergence. If you want one, add an inner loop over `LRS`
  in either script (the harness for it is already in the source scripts;
  you only need to expand `LRS=(...)`).
- No SciKnowEval. The original `run_sdpo_all.sh` included four SciKnowEval
  datasets; the new tooluse-only script drops them per the user request.
- No edits to the original `run_sdpo.sh` / `run_sdpo_all.sh` — they remain
  the published full-logit recipes for back-compat.
- No `run_local_*.sh` — the cluster is multi-GPU and the user has rejected
  the local-runner option.

## Implementation status

Done. Both scripts created and executable:
- `experiments/rich_feedback/run_sdpo_divergences.sh` (LCB, 6 jobs).
- `experiments/generalization/run_sdpo_tooluse_divergences.sh` (ToolUse, 6 jobs).

Verified with `bash -n` (syntax) and `bash ... --dry-run` (sbatch expansion):
both scripts emit the expected per-job commands. The originals
(`run_sdpo.sh`, `run_sdpo_all.sh`) are unchanged.

Fixed an inherited naming bug from the source LCB script while porting:
`LAMBDA` and `CLIP_ADV_HIGH` were referenced in `EXP_NAME` but never set in
the loop body of `run_sdpo.sh`, which made the published runs render with
empty `-lambda-clip_adv_high-` segments. The new `run_sdpo_divergences.sh`
drops those placeholders from `EXP_NAME` and removes the dead
`LAMBDAS=(...)` / `CLIP_ADV_HIGHS=(...)` arrays. Resulting names look like
`FINAL-SDPO-train32-divjsd-rollout8-lr1e-6-drossTrue-Qwen-Qwen3-8B`.
