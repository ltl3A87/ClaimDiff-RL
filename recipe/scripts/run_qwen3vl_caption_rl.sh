#!/usr/bin/env bash
#
# Example launch script for ClaimDiff-RL caption training with Qwen3-VL.
#
# Before running, set these required environment variables:
#
#   export ACTOR_LOAD_PATH=/path/to/Qwen3-VL-8B       # HuggingFace model path
#   export DATA_TRAIN_FILE=/path/to/train.parquet      # Training data (parquet)
#   export DATA_VAL_FILE=/path/to/val.parquet           # Validation data (parquet)
#   export GEMINI_API_KEY=<your-gemini-api-key>         # Gemini API key for reward server
#   export _REMOTE_REWARD_JOB_ID=<reward-server-host>   # Reward server job/host id
#   export TRAIN_SAVE_PATH=/path/to/save/checkpoints    # Checkpoint save directory
#
# See README.md for full setup instructions.

set -x

# ============================================================
# Distributed training
# ============================================================
if [ -z "$VC_WORKER_HOSTS" ] || [ "$VC_WORKER_HOSTS" = "" ]; then
    export NUM_NODES=1
else
    export NUM_NODES=$(($(echo $VC_WORKER_HOSTS | tr ',' '\n' | grep -v '^$' | wc -l) + 1))
fi
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
echo "NUM_NODES: $NUM_NODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

# ============================================================
# Data
# ============================================================
export DATA_FILTER_OVERLONG_PROMPTS=${DATA_FILTER_OVERLONG_PROMPTS:-False}
export DATA_SHUFFLE=${DATA_SHUFFLE:-True}
export DATA_GENERATION_BATCH_SIZE=${DATA_GENERATION_BATCH_SIZE:-256}
export DATA_MAX_PROMPT_LENGTH=${DATA_MAX_PROMPT_LENGTH:-8192}
export DATA_MAX_RES_LENGTH=${DATA_MAX_RES_LENGTH:-4096}
export DATA_IMAGE_KEYWORD=${DATA_IMAGE_KEYWORD:-"images"}
export RETURN_RAW_CHAT=${RETURN_RAW_CHAT:-True}
export RETURN_MULTI_MODAL_INPUTS=${RETURN_MULTI_MODAL_INPUTS:-True}

# ============================================================
# Rollout
# ============================================================
export ROLLOUT_MODE=${ROLLOUT_MODE:-"async"}
export ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-"sglang"}
export ROLLOUT_N=${ROLLOUT_N:-4}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
export ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
export ROLLOUT_MAX_GPU_MEM=${ROLLOUT_MAX_GPU_MEM:-0.45}
export ROLLOUT_CHUNKED_PREFILL=${ROLLOUT_CHUNKED_PREFILL:-True}
export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
export ROLLOUT_FREE_CACHE=${ROLLOUT_FREE_CACHE:-True}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export LOG_P_MICRO_BSZ=${LOG_P_MICRO_BSZ:-2}

# ============================================================
# Actor / Training
# ============================================================
export ACTOR_LR=${ACTOR_LR:-1e-6}
export ACTOR_PPO_GLOBAL_BSZ=${ACTOR_PPO_GLOBAL_BSZ:-128}
export ACTOR_PPO_MICRO_BSZ=${ACTOR_PPO_MICRO_BSZ:-4}
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
export ACTOR_CLIP_RATIO=${ACTOR_CLIP_RATIO:-0.2}
export ACTOR_CLIP_RATIO_LOW=${ACTOR_CLIP_RATIO_LOW:-0.2}
export ACTOR_CLIP_RATIO_HIGH=${ACTOR_CLIP_RATIO_HIGH:-0.28}
export ACTOR_LOSS_AGG_MODE=${ACTOR_LOSS_AGG_MODE:-"token_mean"}
export ACTOR_KL_LOSS_USE=${ACTOR_KL_LOSS_USE:-False}
export ACTOR_KL_LOSS_TYPE=${ACTOR_KL_LOSS_TYPE:-"low_var_kl"}
export ACTOR_KL_LOSS_COEFF=${ACTOR_KL_LOSS_COEFF:-0.001}
export ACTOR_ENTROPY_COEFF=${ACTOR_ENTROPY_COEFF:-0.001}
export ACTOR_USE_LIGER=${ACTOR_USE_LIGER:-False}
export USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-False}
export USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
export USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-False}
export LOSS_MODE=${LOSS_MODE:-"vanilla"}
export ACTOR_DTYPE=${ACTOR_DTYPE:-"float32"}
export FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-True}
export ACTOR_FSDP_PARAM_OFFLOAD=${ACTOR_FSDP_PARAM_OFFLOAD:-False}
export ACTOR_FSDP_OMT_OFFLOAD=${ACTOR_FSDP_OMT_OFFLOAD:-False}
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$((ROLLOUT_N * (DATA_MAX_PROMPT_LENGTH + DATA_MAX_RES_LENGTH)))}

# ============================================================
# LR schedule
# ============================================================
export WARMUP_STYLE=${WARMUP_STYLE:-"constant"}
export LR_WARMUP_STEPS_RATIO=${LR_WARMUP_STEPS_RATIO:-0.05}

# ============================================================
# Algorithm
# ============================================================
export ALGO_ADV_ESTIMATOR=${ALGO_ADV_ESTIMATOR:-"grpo"}
export ALGO_KL_COEF=${ALGO_KL_COEF:-0.001}

# ============================================================
# Reward
# ============================================================
export ACC_SCALE_RANGE=${ACC_SCALE_RANGE:-"[0, 1.0]"}
export FORMAT_SCALE_RANGE=${FORMAT_SCALE_RANGE:-"[0, 1.0]"}

# ============================================================
# Trainer
# ============================================================
export EXP_NAME=${EXP_NAME:-"claimdiff_rl_qwen3vl"}
export TRAIN_PROJECT_NAME=${TRAIN_PROJECT_NAME:-"ClaimDiff-RL"}
export TRAIN_SAVE_FREQ=${TRAIN_SAVE_FREQ:-50}
export TRAIN_TEST_FREQ=${TRAIN_TEST_FREQ:-50}
export TRAIN_TOTAL_EPOCHS=${TRAIN_TOTAL_EPOCHS:-3}

# ============================================================
# Misc
# ============================================================
export BASE_DIR=${BASE_DIR:-$(dirname "$(dirname "$(realpath "$0")")")}
export USE_SHM=${USE_SHM:-False}
export FUSED_KERNEL_BACKEND=${FUSED_KERNEL_BACKEND:-"triton"}
export FILTER_OVERLONG_MASK=${FILTER_OVERLONG_MASK:-False}
export TOKENIZATION_SANITY_CHECK_MODE=${TOKENIZATION_SANITY_CHECK_MODE:-"disable"}

if [[ "$ACTOR_LOAD_PATH" =~ [Qq][Ww][Ee][Nn]3-?[Vv][Ll] ]]; then
  export IMAGE_PATCH_SIZE=${IMAGE_PATCH_SIZE:-16}
else
  export IMAGE_PATCH_SIZE=${IMAGE_PATCH_SIZE:-14}
fi

echo "DATA_TRAIN_FILE: $DATA_TRAIN_FILE"

python3 -m verl.trainer.main_ppo \
  --config-path="${BASE_DIR}"/config \
  --config-name="caption" \
  do_eval=False \
  data.train_files="$DATA_TRAIN_FILE" \
  data.val_files="$DATA_VAL_FILE" \
  data.train_batch_size="$DATA_GENERATION_BATCH_SIZE" \
  data.max_prompt_length="$DATA_MAX_PROMPT_LENGTH" \
  data.max_response_length="$DATA_MAX_RES_LENGTH" \
  data.return_raw_chat="$RETURN_RAW_CHAT" \
  data.return_multi_modal_inputs="$RETURN_MULTI_MODAL_INPUTS" \
  data.filter_overlong_prompts="$DATA_FILTER_OVERLONG_PROMPTS" \
  data.truncation="error" \
  data.image_key="$DATA_IMAGE_KEYWORD" \
  data.shuffle="$DATA_SHUFFLE" \
  data.image_patch_size="$IMAGE_PATCH_SIZE" \
  actor_rollout_ref.model.path="$ACTOR_LOAD_PATH" \
  actor_rollout_ref.model.use_shm="$USE_SHM" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
  actor_rollout_ref.model.use_fused_kernels="$USE_FUSED_KERNELS" \
  actor_rollout_ref.model.fused_kernel_options.impl_backend="$FUSED_KERNEL_BACKEND" \
  actor_rollout_ref.model.use_liger="$ACTOR_USE_LIGER" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=False \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.freeze_vision_tower="$FREEZE_VISION_TOWER" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$ACTOR_PPO_GLOBAL_BSZ" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_PPO_MICRO_BSZ" \
  actor_rollout_ref.actor.use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.clip_ratio="$ACTOR_CLIP_RATIO" \
  actor_rollout_ref.actor.clip_ratio_low="$ACTOR_CLIP_RATIO_LOW" \
  actor_rollout_ref.actor.clip_ratio_high="$ACTOR_CLIP_RATIO_HIGH" \
  actor_rollout_ref.actor.loss_agg_mode="$ACTOR_LOSS_AGG_MODE" \
  actor_rollout_ref.actor.policy_loss.loss_mode="$LOSS_MODE" \
  actor_rollout_ref.actor.use_kl_loss="$ACTOR_KL_LOSS_USE" \
  actor_rollout_ref.actor.use_torch_compile="$USE_TORCH_COMPILE" \
  actor_rollout_ref.actor.kl_loss_coef="$ACTOR_KL_LOSS_COEFF" \
  actor_rollout_ref.actor.kl_loss_type="$ACTOR_KL_LOSS_TYPE" \
  actor_rollout_ref.actor.entropy_coeff="$ACTOR_ENTROPY_COEFF" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO" \
  actor_rollout_ref.actor.optim.warmup_style="$WARMUP_STYLE" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_FSDP_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_FSDP_OMT_OFFLOAD" \
  actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_P_MICRO_BSZ" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
  actor_rollout_ref.rollout.name="$ROLLOUT_BACKEND" \
  actor_rollout_ref.rollout.mode="$ROLLOUT_MODE" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_MAX_GPU_MEM" \
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMP" \
  actor_rollout_ref.rollout.enable_chunked_prefill="$ROLLOUT_CHUNKED_PREFILL" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
  actor_rollout_ref.rollout.free_cache_engine="$ROLLOUT_FREE_CACHE" \
  +actor_rollout_ref.rollout.limit_images=15 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_P_MICRO_BSZ" \
  actor_rollout_ref.ref.fsdp_config.param_offload="$ACTOR_FSDP_PARAM_OFFLOAD" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  algorithm.adv_estimator="$ALGO_ADV_ESTIMATOR" \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef="$ALGO_KL_COEF" \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name="$TRAIN_PROJECT_NAME" \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  trainer.nnodes="$NUM_NODES" \
  trainer.default_local_dir="$TRAIN_SAVE_PATH/$EXP_NAME" \
  trainer.val_before_train=False \
  trainer.resume_mode=auto \
  trainer.save_freq="$TRAIN_SAVE_FREQ" \
  trainer.test_freq="$TRAIN_TEST_FREQ" \
  trainer.total_epochs="$TRAIN_TOTAL_EPOCHS" \
  trainer.filter_overlong_mask="$FILTER_OVERLONG_MASK" \
  reward_model.reward_manager=remote \
  reward_model.reward_kwargs.acc_scale_range="$ACC_SCALE_RANGE" \
  reward_model.reward_kwargs.format_scale_range="$FORMAT_SCALE_RANGE"
