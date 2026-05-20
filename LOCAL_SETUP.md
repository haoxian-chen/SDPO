# Local Setup Guide

End-to-end setup for running the SDPO divergence-sweep experiments on a
local (non-Slurm) multi-GPU box. For Slurm/cluster deployment, see the
`experiments/.../run_*_divergences.sh` Slurm versions and `INSTALL.md`.

---

## 0. Prerequisites

- Linux (tested on Ubuntu 22.04 / SLES 15)
- NVIDIA GPU(s) with CUDA-compatible driver (≥ CUDA 12.4)
- Python 3.12 (`python3.12 --version`)
- ≥ 4 GPUs ideally, since the recipe defaults to `n_gpus_per_node: 4`. Fewer
  is OK if you override (see step 6).

---

## 1. Create a clean Python 3.12 env

Pick one:

```bash
# conda
conda create -n sdpo python=3.12 -y
conda activate sdpo
```

```bash
# or venv
python3.12 -m venv ~/envs/sdpo
source ~/envs/sdpo/bin/activate
```

---

## 2. Install PyTorch first (must come before everything else)

```bash
pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
```

Sanity-check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# Expected: 2.5.1+cu124 True <N>
```

---

## 3. Install repo + dependencies (editable mode)

```bash
cd /path/to/SDPO   # wherever you cloned it
pip install -r requirements.txt   # see note below
pip install -e .
```

> **Note:** `INSTALL.md:29` references `requirements-stable.txt` which doesn't
> exist in this checkout. `requirements.txt` is the closest. If you have
> specific hardware:
>
> - `requirements-gh200.txt` for GH200
> - `requirements-cuda.txt` for general CUDA boxes
>
> All three pin slightly different versions; pick one and stick with it.

---

## 4. Install FlashAttention 2

```bash
pip install flash-attn --no-build-isolation
```

This compiles against your installed PyTorch + CUDA. If it fails with a build
error, try a prebuilt wheel matching your torch/cuda version from
<https://github.com/Dao-AILab/flash-attention/releases>.

---

## 5. Install vLLM + SGLang for rollouts

```bash
pip install -r requirements_sglang.txt
```

This is the rollout engine the recipe uses
(`actor_rollout_ref.rollout.name=vllm` in `user.yaml:30`). Without it the
training will crash at rollout time.

---

## 6. Set environment variables

```bash
export PYTHONPATH=$PWD:$PYTHONPATH      # so `import data` works
export WANDB_API_KEY=...                # your key from https://wandb.ai/authorize
# Optional: if your node has fewer than 4 GPUs:
export N_GPUS_PER_NODE=2                # or 1, or 8 — local wrappers honor this
```

Put the `WANDB_API_KEY` line in `~/.bashrc` if you want it to persist.

If you want to run **offline** (no wandb upload):

```bash
export WANDB_MODE=offline
```

---

## 7. Verify installation

```bash
# Imports work, divergences are wired up
python -c "import verl; from verl.workers.config.actor import DIVERGENCE_TYPES; print(DIVERGENCE_TYPES)"
# Expected:
# ('reverse_kl', 'forward_kl', 'jsd', 'improved_forward_kl', 'improved_reverse_kl', 'improved_jsd')

# vLLM available
python -c "import vllm; print(vllm.__version__)"

# Lightweight tests (skip GPU-requiring ones)
pip install pytest
pytest tests/ -k "not gpu and not megatron" -x
```

---

## 8. Prepare datasets (one-time)

```bash
# LCB (downloads from HuggingFace)
python data/load_dataset.py \
    --dataset_name livecodebench/code_generation_lite-v6 \
    --output_path datasets/lcb_v6.json
python data/split_tests.py \
    --json_path datasets/lcb_v6.json \
    --output_dir datasets/lcb_v6
python data/preprocess.py --data_source datasets/lcb_v6

# ToolUse (already shipped as JSONL, just convert to parquet)
python data/preprocess.py --data_source datasets/tooluse
```

After this you should have:

```
datasets/lcb_v6/{train,test}.parquet
datasets/tooluse/{train,test}.parquet
```

---

## 9. Smoke test (verify the whole pipeline)

```bash
# Preview first (no GPU work)
bash experiments/rich_feedback/smoke_test_sdpo_divergences_local.sh --dry-run
bash experiments/generalization/smoke_test_sdpo_tooluse_divergences_local.sh --dry-run

# Then actually run — ~10-20 min each on 4 GPUs
bash experiments/rich_feedback/smoke_test_sdpo_divergences_local.sh
bash experiments/generalization/smoke_test_sdpo_tooluse_divergences_local.sh
```

Watch for:

- `Total training steps: 10` printed at startup.
- `training/global_step` ticking up to 10 in wandb.
- `actor/distill_div/jsd` and `actor/distill_teacher_kl` showing finite values.
- Clean exit (no `KeyboardInterrupt`, no `SIGKILL`).

---

## 10. Full sweep

Once the smoke is clean:

```bash
bash experiments/rich_feedback/run_sdpo_divergences_local.sh
bash experiments/generalization/run_sdpo_tooluse_divergences_local.sh
```

Sequential — each is 6 divergences run one after another.
