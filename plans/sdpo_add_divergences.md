# Port f-Divergence Knob to SDPO (Sampled-Token Path)

Companion to `add_divergences_implementation.md` (which documents the
tinker-cookbook version). This file documents the SDPO port that adds
`divergence_type` to `SelfDistillationConfig` and uses it inside
`compute_self_distillation_loss` when `full_logit_distillation=False`.

## Scope

- **Only** the sampled-token branch of `compute_self_distillation_loss`
  (`verl/trainer/ppo/core_algos.py`, the `else` branch that previously
  asserted `alpha == 1.0`).
- The full-logit branch is unchanged. There `alpha` keeps its existing
  meaning (0=forward KL, 1=reverse KL, in-between=Generalized JSD via
  alpha-mixture, as in GKD).
- No change to how the loss is invoked from `dp_actor.py`; the new field
  rides on `self_distillation_config`.

## Math

With `u = p/q`, `p = teacher`, `q = student`, the per-token signal is `(-g(u))`
for one of six divergences:

| `divergence_type`     | `g(u)`                              | `-g(u)`                                           |
| --------------------- | ----------------------------------- | ------------------------------------------------- |
| `reverse_kl`          | `-ln u`                             | `log_u`                                           |
| `forward_kl`          | `u ln u`                            | `-u * log_u`                                      |
| `jsd`                 | `0.5 [u ln u - (1+u) ln((1+u)/2)]`  | `-0.5 * (u*log_u - (u+1)*(log1p(u) - log 2))`     |
| `improved_forward_kl` | `-u`                                | `u`                                               |
| `improved_reverse_kl` | `-ln u + 1`                         | `log_u - 1`                                       |
| `improved_jsd`        | `-0.5 ln((1+u)/2)`                  | `0.5 * (log1p(u) - log 2)`                        |

### Behavior at `u = 1` (student matches teacher)

`reverse_kl`, `forward_kl`, `jsd`, and `improved_jsd` all satisfy `g(1) = 0`,
so their per-token signal vanishes when student matches teacher. The two
"improved" variants intentionally use baseline-shifted generators that
**do not** vanish at the match:

| `divergence_type`     | `g(1)` | `-g(1)` (advantage at match) |
| --------------------- | ------ | ---------------------------- |
| `reverse_kl`          | 0      | 0                            |
| `forward_kl`          | 0      | 0                            |
| `jsd`                 | 0      | 0                            |
| `improved_forward_kl` | −1     | 1                            |
| `improved_reverse_kl` | 1      | −1                           |
| `improved_jsd`        | 0      | 0                            |

For `improved_forward_kl` (`g(u) = -u`) and `improved_reverse_kl`
(`g(u) = -ln u + 1`), the per-token advantage is offset by a constant when
student equals teacher. The constant offset does **not** change the expected
gradient under the policy (because `E_q[c · ∇ log q] = 0`), but it shifts the
loss-value scale and can change the variance of the sampled REINFORCE
estimator. These are intentional design choices for the "improved"
variants — they are not standard f-divergences (a real f-divergence requires
`g(1) = 0` so that `D_f(p ‖ q) = 0` exactly when `p = q`).

## How it plugs into SDPO's sampled-token loss

SDPO's sampled-token loss is a REINFORCE-style policy gradient with the
divergence signal as advantage. Before:

```python
log_ratio = student_log_probs - teacher_log_probs  # = -log_u
per_token_loss = log_ratio.detach() * student_log_probs
```

After:

```python
log_u = teacher_log_probs - student_log_probs
neg_g_u = _compute_neg_g_u(log_u, divergence_type)
per_token_loss = (-neg_g_u).detach() * student_log_probs
```

For `divergence_type='reverse_kl'`, `neg_g_u == log_u`, so
`(-neg_g_u).detach() * student_log_probs == log_ratio.detach() * student_log_probs`
— bit-exact match with the prior loss.

> Note (advantage-style vs REINFORCE-style): tinker-cookbook adds `-g(u)` to
> the trajectory advantages and lets the outer policy-gradient loss apply the
> `-A * log q` term. SDPO computes the per-token loss directly here. With the
> detach() in place, both produce the same gradient on the sampled token.
> Unlike the strict score-function estimator of `∇ E_q[g(u)]`, both setups
> drop the `u g'(u)` corrective term — i.e. this is the same biased
> sample-level estimator that the tinker-cookbook plan describes, used
> consistently across all six divergences. For the two "improved" variants
> whose `g(1) ≠ 0`, the dropped corrective term plus the non-zero baseline
> means the per-token loss value is offset, but the expected gradient is
> unchanged (constants cancel under `E_q[ · ∇ log q]`).

## Numerical stability

Same strategy as tinker-cookbook (`_compute_neg_g_u` in
`core_algos.py`):

- Compute `log_u` first (subtraction in log-space), avoiding underflow that a
  direct `p/q` would suffer when probabilities are tiny.
- Clamp `log_u` to `[-10, 10]` before `exp()` for the four variants that
  use `u` explicitly (`forward_kl`, `jsd`, `improved_forward_kl`,
  `improved_jsd`). Without the clamp, an early-training mismatch
  (`log_u ≈ 30`) would push `u` to ~1e13 and `u * log_u` to extreme values
  that saturate gradients; for larger mismatches (`log_u ≳ 89` in float32)
  `exp(log_u)` overflows to `+inf` outright. The clamp keeps
  `u ∈ [4.5e-5, 22026]`, all comfortably representable in float32.
- `torch.log1p(u)` for the `ln((1+u)/2)` term in the JSD variants.
- `reverse_kl` and `improved_reverse_kl` skip the clamp because they only
  use `log_u` directly (no `exp()`). For `reverse_kl` this preserves
  bit-exact behavior with the previous code path; `improved_reverse_kl` is
  new and just subtracts a constant from `log_u`.
- The four exp-using variants use the **clamped** `log_u` in the
  multiplicative slots (`u * log_u`) as well, matching the tinker-cookbook
  "clipped estimator" choice for self-consistency.

## IS clipping and rollout-IS correction

Unchanged. `is_clip` and `rollout_is_weights` are applied to the resulting
`per_token_loss` after the divergence computation, so they compose cleanly
with any choice of `divergence_type`.

## Files changed

### 1. `verl/workers/config/actor.py`

- Added module-level constant `DIVERGENCE_TYPES`.
- Added `SelfDistillationConfig.divergence_type: str = "reverse_kl"`.
- Extended `__post_init__` to validate `divergence_type ∈ DIVERGENCE_TYPES`.
- Updated docstring: clarified that `alpha` is for the full-logit path and
  `divergence_type` is for the sampled-token path.

### 2. `verl/trainer/config/actor/actor.yaml`

- Added `divergence_type: reverse_kl` under `self_distillation:` so Hydra
  knows the key. Without this, dot-path CLI overrides (e.g.
  `actor_rollout_ref.actor.self_distillation.divergence_type=jsd`) would
  need the `+` prefix to add the key at runtime.
- Clarified the `alpha:` comment to note it is ignored on the sampled-token
  path.

### 3. `verl/trainer/ppo/core_algos.py`

- Added `import math`.
- Imported `DIVERGENCE_TYPES` from `verl.workers.config.actor`.
- Added private helper `_compute_neg_g_u(log_u, divergence_type)` returning
  per-token `-g(u)` for the selected divergence (clamping + `log1p` strategy
  above).
- `compute_self_distillation_loss` sampled-token branch
  (`full_logit_distillation=False`):
  - Removed `assert alpha == 1.0`.
  - Reads `divergence_type` from `self_distillation_config` (defaulting to
    `"reverse_kl"`), validates membership in `DIVERGENCE_TYPES`.
  - Computes `log_u = teacher - student`, then `neg_g_u = _compute_neg_g_u(...)`,
    then `per_token_loss = (-neg_g_u).detach() * student_log_probs`.
  - Adds metrics:
    - `actor/distill_div/{divergence_type}` — masked mean of `g(u)` for the
      selected divergence.
    - `actor/distill_teacher_kl` — masked mean of the reverse-KL k1 estimator
      `-log_u`, added as a stable reverse-KL diagnostic that is emitted
      regardless of `divergence_type` so you can compare student-teacher gap
      across runs that use different divergences. This is a new metric key;
      it did not exist in the prior sampled-token branch.

## Backwards compatibility

- Default `divergence_type="reverse_kl"` reproduces the previous loss
  bit-exactly. **Previously valid** sampled-token configs (i.e.
  `full_logit_distillation=False` with `alpha=1.0`, which is what the old
  branch required) keep the same behavior, as does the default full-logit
  SDPO config in `verl/trainer/config/sdpo.yaml`.
- **Behavior change for `full_logit_distillation=False` with `alpha != 1.0`:**
  the prior code raised `AssertionError("Only reverse KL is supported for
  non-full-logit distillation")`. The new code removes that assert and
  silently uses `divergence_type` (default `"reverse_kl"`), so a config that
  used to fail-fast now runs reverse KL instead of erroring. If you relied
  on that assertion as a guardrail, set `divergence_type` explicitly to
  document intent.
- The full-logit branch is untouched; existing `alpha` settings are honored.

## How to use

In yaml (e.g. an override or in `verl/trainer/config/sdpo.yaml` under
`actor_rollout_ref.actor.self_distillation:`):

```yaml
self_distillation:
  full_logit_distillation: false   # sampled-token path
  divergence_type: jsd             # or forward_kl, improved_forward_kl, improved_jsd
```

CLI override:

```bash
python -m verl.trainer.main_ppo \
    --config-path=verl/trainer/config --config-name=sdpo \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=false \
    actor_rollout_ref.actor.self_distillation.divergence_type=jsd
```

## Out of scope / follow-ups

- No auto-tuning of any loss coefficient per divergence, and **no knob in
  SDPO mode that scales the self-distillation loss itself**. In `dp_actor.py`
  the result of `compute_self_distillation_loss` is assigned directly to
  `policy_loss` (`verl/workers/actor/dp_actor.py:881`); `actor.kl_loss_coef`
  only scales the optional reference-policy KL term gated by
  `actor.use_kl_loss` (`dp_actor.py:888`, off by default in
  `verl/trainer/config/actor/actor.yaml`) and so is **not** a magnitude knob
  for the SDPO term. Practical implications when switching divergences:
  - The sampled per-token signal is bounded only by the `log_u ∈ [-10, 10]`
    clamp in `_compute_neg_g_u`, not by the divergence's expectation bound:
    `jsd` can be large in magnitude (driven by `u * log_u_clamped` terms) and
    `improved_jsd` can exceed `log 2` per token. The expected JS divergence
    is `≤ log 2`, but a sampled-token estimator is not.
  - Empirically, `forward_kl` and the JSD variants still have different
    per-token magnitudes than `reverse_kl` for the same student/teacher gap,
    so expect a different effective update size.
  - To compensate, retune broader knobs (e.g. `actor.optim.lr`), or add a
    new `SelfDistillationConfig.loss_coef` and multiply `pg_loss` in
    `compute_self_distillation_loss` before returning.
- No unit tests added in this pass. Recommended:
  - Reverse-KL regression: with `divergence_type="reverse_kl"`, the scalar
    loss and gradient from `compute_self_distillation_loss` match the prior
    code bit-exactly on a fixed random batch. The returned `metrics` dict
    will differ — the new branch adds `actor/distill_div/{divergence_type}`
    and `actor/distill_teacher_kl` keys not present before — so compare the
    loss/grad, not the full return value.
  - Zero-at-match: for the four divergences with `g(1) = 0` (`reverse_kl`,
    `forward_kl`, `jsd`, `improved_jsd`), when
    `student_log_probs == teacher_log_probs`, `per_token_loss == 0`. For the
    two improved variants with non-zero `g(1)`, instead assert the per-token
    advantage equals the expected constant (`1` for `improved_forward_kl`,
    `-1` for `improved_reverse_kl`) before multiplication by
    `student_log_probs`.
  - Shape parity: returned loss is a scalar and metrics keys are populated.
- Full-logit path was intentionally not unified under `divergence_type` —
  could be a follow-up if you want a single knob across both paths
  (e.g., add a `gjsd_alpha` value or replace `alpha` entirely).

## Implementation status

Done — three files edited:
- `verl/workers/config/actor.py` (added `DIVERGENCE_TYPES`, `divergence_type`
  field, validation, docstring update).
- `verl/trainer/config/actor/actor.yaml` (added `divergence_type: reverse_kl`
  to the Hydra schema; clarified `alpha` comment).
- `verl/trainer/ppo/core_algos.py` (added `import math`, `_compute_neg_g_u`
  helper, refactored sampled-token branch, added metrics).

`python -m py_compile` clean on the two Python files. The pre-existing
pylance "not accessed" warnings in other functions of `core_algos.py` are
unrelated to this change.
