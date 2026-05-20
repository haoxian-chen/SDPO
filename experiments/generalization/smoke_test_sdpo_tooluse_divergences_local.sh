#!/bin/bash

# Usage: ./smoke_test_sdpo_tooluse_divergences_local.sh [--dry-run]
#
# Local (no-Slurm) smoke test for the ToolUse divergence sweep. Runs 10
# outer steps for one divergence (jsd by default — exercises the new
# clamp + log1p code path in _compute_neg_g_u) to confirm the new
# sampled-token code path works end-to-end before launching the full
# run_sdpo_tooluse_divergences_local.sh sweep.
#
# To smoke a different divergence, change DIVERGENCE_TYPES below.
# reverse_kl is bit-exact with the prior implementation, so use jsd or
# improved_jsd if you want to actually exercise the new math.
#
# Differences vs run_sdpo_tooluse_divergences_local.sh:
#   - trainer.total_training_steps=10 (clean exit after 10 outer steps)
#   - trainer.test_freq=10 (one validation at the last step)
#   - DIVERGENCE_TYPES=(jsd) — single job
#   - rollout.val_kwargs.n=4 (cheaper validation)
#   - EXP_NAME prefix SMOKE-, group_name suffix -smoke
#
# Expected wall-clock: ~5-12 min on 4 GPUs (Qwen3-8B).

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

# Single divergence — jsd exercises the new clamp + log1p code path.
DIVERGENCE_TYPES=(jsd)

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
                            EXP_NAME="SMOKE-SDPO-train${TRAIN_BATCH_SIZE}-div${DIVERGENCE_TYPE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${MODEL_NAME}-local"

                            ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SDPO-generalization-div-smoke \
trainer.total_training_steps=10 \
trainer.test_freq=10 \
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
actor_rollout_ref.rollout.val_kwargs.n=4"

                            run_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
                        done
                    done
                done
            done
        done
    done
done

echo "Sweep finished."
