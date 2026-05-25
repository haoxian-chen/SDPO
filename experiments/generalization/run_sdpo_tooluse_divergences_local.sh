#!/bin/bash

# Usage: ./run_sdpo_tooluse_divergences_local.sh [--dry-run]
#
# Slurm-less variant of experiments/generalization/run_sdpo_tooluse_divergences.sh.
# Same DIVERGENCE_TYPES sweep on ToolUse, but each "job" is a foreground
# invocation of training/verl_training.sh on the current node, run one after
# another. Use when:
#   - You don't have Slurm.
#   - You have Slurm but don't have the CSCS pyxis `--environment=sdpo`
#     container plugin.
#   - You're debugging on a single node and want stdout in your terminal.
#
# Assumptions:
#   - PyTorch + verl + flash-attn already installed in the active env
#     (see INSTALL.md). This script does NOT pip-install anything.
#   - WANDB_API_KEY is set in your shell environment (or wandb is off).
#   - You have enough GPUs to satisfy `n_gpus_per_node` in user.yaml
#     (4 by default; override with N_GPUS_PER_NODE=2 etc.).
#
# Logs land in <repo>/output/SDPO/<EXP_NAME>.log via tee.

set -u  # NOT set -e — we want subsequent divergences to run even if one fails.

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo"

DATA_PATHS=(
    "datasets/tooluse"
)

# Sweep Parameters (mirror run_sdpo_tooluse_divergences.sh)
TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
LRS=(1e-5)
DONTS_REPROMPT_ON_SELF_SUCCESSS=(True)

# f-divergences on the sampled-token path. Valid values:
# reverse_kl, forward_kl, jsd, improved_forward_kl, improved_reverse_kl, improved_jsd.
DIVERGENCE_TYPES=(reverse_kl forward_kl jsd improved_forward_kl improved_reverse_kl improved_jsd)

MODEL_PATHS=(
    "Qwen/Qwen3-8B"
)

# =============================================================================
# SETUP
# =============================================================================

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export USER=${USER:-$(whoami)}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}

LOG_DIR="$PROJECT_ROOT/output/SDPO"
mkdir -p "$LOG_DIR"

# =============================================================================
# LOCAL RUNNER (replaces submit_job from the Slurm version)
# =============================================================================

run_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"

    local log_file="$LOG_DIR/${exp_name}.log"
    local cmd=(
        bash "$PROJECT_ROOT/training/verl_training.sh"
        "$exp_name" "$CONFIG_NAME" "$data_path"
    )
    # shellcheck disable=SC2206  # word-splitting on $script_args is intentional
    cmd+=( $script_args )

    if [ "$DRY_RUN" = true ]; then
        echo "----------------------------------------------------------------"
        echo "Would run: $exp_name"
        echo "  log: $log_file"
        echo "  cmd: ${cmd[*]}"
        return 0
    fi

    echo "----------------------------------------------------------------"
    echo "[$(date '+%F %T')] Starting: $exp_name"
    echo "  log: $log_file"
    "${cmd[@]}" 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        echo "[$(date '+%F %T')] FAILED ($status): $exp_name — continuing sweep"
    else
        echo "[$(date '+%F %T')] Done: $exp_name"
    fi
}

# =============================================================================
# MAIN SWEEP LOOP
# =============================================================================

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for DONTS_REPROMPT_ON_SELF_SUCCESS in "${DONTS_REPROMPT_ON_SELF_SUCCESSS[@]}"; do
                for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                    for DIVERGENCE_TYPE in "${DIVERGENCE_TYPES[@]}"; do
                        for DATA_PATH in "${DATA_PATHS[@]}"; do
                            MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                            EXP_NAME="FINAL-SDPO-train${TRAIN_BATCH_SIZE}-div${DIVERGENCE_TYPE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-local"

                            ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SDPO-generalization-div \
trainer.total_training_steps=80 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.self_distillation.full_logit_distillation=False \
actor_rollout_ref.actor.self_distillation.divergence_type=$DIVERGENCE_TYPE \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
actor_rollout_ref.rollout.val_kwargs.seed=42 \
actor_rollout_ref.rollout.sampling_seed=42"

                            run_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
                        done
                    done
                done
            done
        done
    done
done

echo "Sweep finished."
