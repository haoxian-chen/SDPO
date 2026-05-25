# Reproducible Sampling Seeds for SDPO (Eval + Training, Agent-Loop Path)

Goal: port the two seed controls from the tinker on-policy distillation recipe
into SDPO so the divergence experiments are reproducible and **paired** across
runs (variance reduction for A/B'ing divergence types):

- tinker `tooluse_eval_seed` → **eval** sampling seed.
- tinker `sampling_seed` → **training-rollout** sampling seed.

Both are opt-in (default `None` ⇒ unseeded, behavior unchanged) and apply to
the **agent-loop (async) rollout path** that the divergence experiments use.

## Why this was needed

The agent-loop rollout (`verl/experimental/agent_loop/agent_loop.py`) builds
per-request sampling params and, for validation, only overrides `top_p` and
`temperature` from `val_kwargs` — it sets **no seed** and ignores `do_sample`.
So validation sampled at `val_kwargs.temperature` with `val_kwargs.n` draws and
was **not reproducible** run-to-run. Training rollouts likewise had only the
single static vLLM engine seed (`config.get("seed", 0)` = 0, set once at engine
init), so two runs could not be paired step-by-step.

The agent loop passes the per-request `n` by `.repeat(n)`-ing the batch at the
trainer level and dispatching each replica as a **separate request** (the dict
has no `n`). A single shared seed would therefore make a prompt's `n`
completions identical, so the seed must be derived **per request**.

## Derivation

`get_trajectory_info` already tags each request with `step` (= `global_steps`),
`sample_index` (per-prompt id), and `rollout_n` (replica counter 0..n-1). The
new helper folds a base seed + identifiers into a `[0, 2**31)` seed:

```python
def _derive_request_seed(base_seed, *components):  # agent_loop.py:841
    h = int(base_seed) & 0x7FFFFFFF
    for c in components:
        h = (h * 1000003 + int(c)) & 0x7FFFFFFF
    return h
```

- **Validation:** `_derive_request_seed(val_kwargs.seed, sample_index, rollout_n)`
  — **omits `step`**, so eval sampling noise is held **constant across training
  steps** (only the model changes between evals).
- **Training:** `_derive_request_seed(sampling_seed, step, sample_index, rollout_n)`
  — **includes `step`**, so each step samples fresh while two runs with the same
  `sampling_seed` (and data order) stay paired. `global_steps` is present on the
  training gen batch (`ray_trainer.py:1633`), so the step term is meaningful.

Invariants (self-tested): a prompt's `n` replicas get distinct seeds (the
trailing `rollout_n` differs); the same `(base, [step,] prompt, replica)` key
maps to the same seed across runs (pairing); different step ⇒ different seed.
Cross-prompt seed collisions are harmless — different prompts produce different
outputs regardless of seed.

End-to-end the seed is honored: the dict is splatted into vLLM
`SamplingParams(max_tokens=..., **sampling_params)` in
`verl/workers/rollout/vllm_rollout/vllm_async_server.py` (~line 497).

## Files changed

### 1. `verl/workers/config/rollout.py`

- `SamplingConfig.seed: Optional[int] = None` (`rollout.py:48`) — the **eval**
  seed, used on `val_kwargs`.
- `RolloutConfig.sampling_seed: Optional[int] = None` (`rollout.py:146`) — the
  **training-rollout** base seed (mirrors tinker's `sampling_seed`).

### 2. `verl/experimental/agent_loop/agent_loop.py`

- Added module-level `_derive_request_seed(base_seed, *components)` helper
  (`:841`).
- In `generate_sequences` (`:405`): `base_sampling_seed = config.val_kwargs.seed
  if is_validate else config.sampling_seed` (`:445`); in the per-sample dispatch
  loop (`:482`), when `base_sampling_seed is not None`, derive the per-request
  seed (eval vs training form above) and pass a **per-task** copy
  `{**sampling_params, "seed": derived_seed}`. The shared `sampling_params` dict
  is untouched when unseeded.

### 3. `verl/trainer/config/rollout/rollout.yaml` (live config)

- Added `seed: null` under `val_kwargs:` and `sampling_seed: null` at the
  rollout level, with comments, so dot-path CLI overrides work without a `+`
  prefix.

### 4. `verl/trainer/config/_generated_ppo_trainer.yaml` + `_generated_ppo_megatron_trainer.yaml` (reference only)

- Hand-patched `seed: null` under `val_kwargs:` and `sampling_seed: null` at the
  rollout level to match. These files are "for reference, never used" but a CI
  check (`scripts/generate_trainer_config.sh`, `git diff --exit-code`) fails if
  stale — see caveat below.

### 5. Experiment scripts (the 4 divergence sweeps)

Added both seeds to the `ARGS` string (fixed `42` across all divergence runs so
sampling noise is identical across divergence types):

- `experiments/generalization/run_sdpo_tooluse_divergences.sh`
- `experiments/generalization/run_sdpo_tooluse_divergences_local.sh`
- `experiments/rich_feedback/run_sdpo_divergences.sh`
- `experiments/rich_feedback/run_sdpo_divergences_local.sh`

```bash
actor_rollout_ref.rollout.val_kwargs.seed=42 \
actor_rollout_ref.rollout.sampling_seed=42
```

## How to use

CLI override:

```bash
python -m verl.trainer.main_ppo ... \
    actor_rollout_ref.rollout.val_kwargs.seed=42 \   # reproducible eval
    actor_rollout_ref.rollout.sampling_seed=42       # paired training rollouts
```

For paired A/B runs, keep `sampling_seed` (and the data order, i.e.
`actor_rollout_ref.actor.data_loader_seed`, default `1`) identical across the
runs you compare; vary only the thing under test (e.g. `divergence_type`).

## Backwards compatibility

- Both fields default to `None` ⇒ no per-request seed is set ⇒
  training/eval sampling is byte-identical to before. No behavior change unless
  a seed is explicitly set.

## Caveats

- **Agent-loop path only.** The non-agent-loop (SPMD / `hf_rollout`) paths are
  not covered. The divergence experiments use the agent loop, so this is the
  right target, but a different rollout backend would not be seeded.
- **Pairing requires identical data order** between the two runs (same
  `data_loader_seed` and batch composition); otherwise `sample_index` maps to a
  different prompt and the runs are no longer paired. Same assumption tinker
  makes.
- **`do_sample` is still ignored on the agent-loop path.** This change does
  *not* fix the separate bug where `val_kwargs.do_sample=false` fails to force
  greedy in the agent loop (it only overrides `top_p`/`temperature`). Eval
  remains stochastic at `val_kwargs.temperature`; the seed just makes that
  stochasticity reproducible.
- **Reference YAMLs were hand-patched.** Regenerate with
  `scripts/generate_trainer_config.sh` in an env with `hydra` installed before
  committing, to satisfy the staleness check exactly (not runnable in the dev
  shell used here — `hydra` missing).

## Out of scope / follow-ups

- No fix for the ignored-`do_sample` greedy-eval bug (tracked separately).
- No seed added to `run_sdpo.sh` / `run_baseline_grpo.sh` / smoke tests — only
  the 4 divergence sweeps, per the user. If a fair SDPO-vs-baseline comparison
  is wanted, add the same two seeds there.
- No unit tests added. Recommended: assert (a) a prompt's `n` replica seeds are
  distinct, (b) the same key maps to the same seed across calls, (c) different
  `step` ⇒ different training seed but same eval seed.

## Implementation status

Done — code + config + scripts edited:
- `verl/workers/config/rollout.py` (`SamplingConfig.seed`,
  `RolloutConfig.sampling_seed`).
- `verl/experimental/agent_loop/agent_loop.py` (`_derive_request_seed` helper;
  per-request eval/training seed in `generate_sequences`).
- `verl/trainer/config/rollout/rollout.yaml` + both `_generated_*` YAMLs
  (`seed: null`, `sampling_seed: null`).
- 4 divergence sweep scripts (`val_kwargs.seed=42`, `sampling_seed=42`).

`python -m py_compile` clean on both Python files; seed-derivation invariants
verified by a standalone self-test; `bash -n` clean on all 4 scripts. Reference
YAMLs still need a real `scripts/generate_trainer_config.sh` run before commit.
