# Add 4 New f-Divergence Advantage Types to On-Policy Distillation

Documents the refactor that extends `incorporate_kl_penalty` in
`tinker_cookbook/distillation/train_on_policy.py` so the per-token distillation
signal can use 5 different f-divergences instead of only Reverse KL.

## Goal

The previous implementation hard-coded Reverse KL: per-token advantage
contribution = `kl_penalty_coef * mask * (teacher_logprob - sampled_logprob)`.
We generalise this to any f-divergence with generator `g(u)`, where
`u = p/q` with `p` = teacher probability and `q` = student probability. The
per-token contribution becomes `kl_penalty_coef * mask * (-g(u))`.

## Math

Let `log_u = log p - log q = teacher_logprobs - sampled_logprobs` and
`u = exp(log_u)`. Five supported divergences:

| `divergence_type`     | `g(u)`                              | `-g(u)` used in code                              |
| --------------------- | ----------------------------------- | ------------------------------------------------- |
| `reverse_kl`          | `-ln u`                             | `log_u`                                           |
| `forward_kl`          | `u ln u`                            | `-u * log_u`                                      |
| `jsd`                 | `0.5 [u ln u - (1+u) ln((1+u)/2)]`  | `-0.5 * (u*log_u - (u+1)*(log1p(u) - log 2))`     |
| `improved_forward_kl` | `1 - u`                             | `u - 1`                                           |
| `improved_jsd`        | `-0.5 ln((1+u)/2)`                  | `0.5 * (log1p(u) - log 2)`                        |

### Behavior at `u = 1` (student matches teacher)

`g(1)` is zero for all five generators, so the per-token advantage
contribution `-g(u)` vanishes when student == teacher:

| `divergence_type`     | `g(1)` | `-g(1)` (advantage at match) |
| --------------------- | ------ | ---------------------------- |
| `reverse_kl`          | 0      | 0                            |
| `forward_kl`          | 0      | 0                            |
| `jsd`                 | 0      | 0                            |
| `improved_forward_kl` | 0      | 0                            |
| `improved_jsd`        | 0      | 0                            |

`improved_forward_kl` uses the baseline-shifted generator `g(u) = 1 - u`
rather than the unshifted `g(u) = -u`. This keeps the gradient-equivalent
f-divergence signal while making the per-token advantage exactly zero at
`u = 1`.

## Numerical stability

- **Compute `log_u` first** (subtraction in log-space), avoiding underflow
  that a direct `p/q` would suffer when `p` or `q` is tiny.
- **Clamp `log_u` to `[-10, 10]` before `exp()`**. Without clamping, an
  early-training mismatch (e.g. `log_u = 30`) would give `u ≈ 1e13`, and
  `u * log_u` would explode and saturate gradients to Inf. The clamp keeps
  `u ∈ [4.5e-5, 22026]`, all representable in float32.
- **Use `torch.log1p(u)`** instead of `torch.log(1 + u)` so the `ln((1+u)/2)`
  terms in the JSD variants stay accurate when `u` is small.
- **Reverse KL is left unclamped** because it doesn't use `exp` — this
  preserves bit-exact behavior with the previous implementation.

### Clipped-estimator note (important)

For the four non-reverse-KL variants, the implementation uses the **clamped**
`log_u` not only to compute `u = exp(...)` but **also** in the multiplicative
terms (`u * log_u` in forward_kl and jsd). See `train_on_policy.py:85-92`:

```python
log_u_clamped = torch.clamp(log_u, min=-10.0, max=10.0)
u = torch.exp(log_u_clamped)
...
return -u * log_u_clamped         # forward_kl
return -0.5 * (u * log_u_clamped - (u + 1.0) * (torch.log1p(u) - log2))  # jsd
```

This means outside `log_u ∈ [-10, 10]` the per-token signal is a **clipped
estimator**, not the exact `-g(u)`. The rationale is mathematical
consistency: `u` came from the clamped value, so pairing it with the clamped
`log_u` keeps the formulas self-consistent (e.g. for forward KL,
`-u * log_u_clamped` is exactly `-g(u_clamped)`).

If you want the unclipped `log_u` in those multiplicative slots, swap
`log_u_clamped` for `log_u` on those lines — but be aware that large negative
`log_u` paired with very small `u` can still produce huge `|u * log_u|`
through the unclamped path in unlucky edge cases.

## Files changed

### 1. `tinker_cookbook/distillation/train_on_policy.py`

- Added `import math`.
- Added module-level constant `DIVERGENCE_TYPES` (tuple of the 5 valid strings).
- Added private helper `_compute_neg_g_u(log_u, divergence_type) -> Tensor`
  that returns the per-token `-g(u)` for the selected divergence, applying
  the clamping/log1p strategy described above.
- Refactored `incorporate_kl_penalty`:
  - New parameter `divergence_type: str = "reverse_kl"`.
  - Computes `log_u_D` (unmasked) once per datum.
  - Per-datum: `neg_g_u = _compute_neg_g_u(log_u_D[i], divergence_type)`,
    then `kl_advantages = kl_penalty_coef * mask * neg_g_u`
    (mask applied **once**, replacing the prior `mask² == mask` redundancy).
  - Optional `discounted_future_sum_vectorized` is applied as before when
    `kl_discount_factor > 0`.
  - Validates `divergence_type ∈ DIVERGENCE_TYPES` up-front with a clear error.
- Metrics (returned dict):
  - `teacher_kl` — kept for dashboard back-compat. Always the reverse-KL k1
    estimator `mean(-log_u over masked tokens)`, independent of the divergence
    in use.
  - `teacher_kl/dataset_{idx}` — kept, same definition, per dataset.
  - `teacher_div/{divergence_type}` — new, `mean(g(u))` for the selected
    divergence.
- Added `Config.divergence_type: str = "reverse_kl"`.
- Threaded `divergence_type` through `prepare_minibatch` (new keyword arg) and
  `do_train_step_and_get_sampling_client` (reads `config.divergence_type`).

### 2. `tinker_cookbook/recipes/distillation/on_policy_distillation.py`

- Added `CLIConfig.divergence_type: str = "reverse_kl"`.
- `cli_main` forwards it into `train_on_policy.Config`.

## Backwards compatibility

- Default value (`"reverse_kl"`) preserves existing behavior.
- For `reverse_kl`, the update is mathematically identical to the prior code:
  - old: `-kl_penalty_coef * mask * ((sampled - teacher) * mask) = kl_penalty_coef * mask² * log_u`
  - new: `kl_penalty_coef * mask * log_u`
  - and `mask² == mask` since mask ∈ {0, 1}.
- `teacher_kl` metric kept and computed the same way, so existing dashboards
  and run comparisons are unaffected.

## How to use

CLI:

```bash
python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
    model_name=Qwen/Qwen3-8B-Base \
    dataset=deepmath \
    divergence_type=jsd \
    learning_rate=1e-4 \
    groups_per_batch=512 \
    lora_rank=128 \
    wandb_project=cookbook_distillation
```

Programmatically:

```python
config = train_on_policy.Config(
    ...,
    divergence_type="forward_kl",   # or "jsd", "improved_forward_kl", "improved_jsd"
    kl_penalty_coef=1.0,
)
```

## Out of scope / follow-ups

- No auto-tuning of `kl_penalty_coef` per divergence. JSD is bounded by
  `log 2` so it has a different natural scale than reverse KL; users may need
  to retune `kl_penalty_coef`.
- No unit tests added in this pass. Recommended next steps:
  - Shape test: each divergence returns a tensor matching the input shape.
  - Zero-at-match test: for all divergence types,
    `_compute_neg_g_u(torch.zeros(...), name)` returns all zeros.
  - Reverse-KL regression: with `divergence_type="reverse_kl"`, advantages
    after `incorporate_kl_penalty` match the prior implementation bit-exactly
    on a fixed batch.
  - Clipping test: at `log_u = 20` (well outside the clamp), the returned
    value matches the clamped formula (`log_u_clamped = 10`), not the unclamped
    one — documents the clipped-estimator behavior.
- No README update in `recipes/distillation/README.md` yet.
- The Harbor multi-turn recipe shares `incorporate_kl_penalty` indirectly via
  `train_on_policy.main`; if a corresponding `divergence_type` CLI flag is
  desired there, it would need a small mirror edit in
  `on_policy_distillation_harbor_multi_turn.py`.

## Implementation status

Done — both files edited, syntax-checked via `python -m py_compile`. The
diagnostics from Pylance about `chz` / `tinker.types` not resolving are an
environment issue unrelated to this change.
