set -x
JOB_NAME=$(date +%Y-%m-%d-%H-%M-%S)

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH=/workdir/project-search/rl4search/verl:$PYTHONPATH

WORLD_SIZE=$AWS_BATCH_JOB_NUM_NODES


if [ $WORLD_SIZE -gt 3 ]; then
    PSIZE=4
elif [ $WORLD_SIZE -gt 1 ]; then
    PSIZE=2
else
    PSIZE=$WORLD_SIZE
fi

# Use environment variables with fallback to default values
# Parse command line arguments
ROLLOUT_COUNT=32  # Default value
TEMP_VALUE=1.0    # Default value
DATA_SOURCE_VALUE="pokerbench-plan"  # Default value
MODEL_TYPE=${MODEL_TYPE:-"Qwen-Qwen2.5-7B"}
TRAIN_BATCH_SIZE=1024
MICRO_BATCH_SIZE_PER_GPU=32
KL_COEF=0.001

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --rollout-n) ROLLOUT_COUNT="$2"; shift ;;
        --temperature) TEMP_VALUE="$2"; shift ;;
        --data-source) DATA_SOURCE_VALUE="$2"; shift ;;
        --model-type) MODEL_TYPE="$2"; shift ;;
        --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift ;;
        --micro-batch-size-per-gpu) MICRO_BATCH_SIZE_PER_GPU="$2"; shift ;;
        --kl-coef) KL_COEF="$2"; shift ;;
        *) shift ;;
    esac
    shift
done

# Construct paths
TRAIN_DATA_PATH="/workdir/project-search/rl4search/data/${DATA_SOURCE_VALUE}/train.parquet"
VAL_DATA_PATH="/workdir/project-search/rl4search/data/${DATA_SOURCE_VALUE}/test.parquet"
MODEL_PATH="/workdir/pretrained-models/${MODEL_TYPE}"

# Update experiment name dynamically based on model type
MODEL_NAME=$(echo $MODEL_TYPE | cut -d'-' -f2)  # Extract model name for experiment name
EXPERIMENT_NAME="${MODEL_NAME}-${DATA_SOURCE_VALUE}-temp${TEMP_VALUE}-rollout${ROLLOUT_COUNT}-${JOB_NAME}"
DEFAULT_LOCAL_DIR="/workdir/project-search/rl4search/checkpoints/"

# # Set PPO batch size based on world size
# if [ $WORLD_SIZE -eq 1 ]; then
#     PPO_BATCH_SIZE=512
#     TRAIN_BATCH_SIZE=1024
# else
#     PPO_BATCH_SIZE=$((512 * WORLD_SIZE))
#     TRAIN_BATCH_SIZE=$((1024 * WORLD_SIZE))
# fi

# # Ensure train batch size is at least as large as PPO batch size
# if [ $TRAIN_BATCH_SIZE -lt $PPO_BATCH_SIZE ]; then
#     TRAIN_BATCH_SIZE=$PPO_BATCH_SIZE
# fi

# aws s3 cp --recursive --quiet s3://linminh/training_logs/rl4search/Qwen-Qwen2.5-7B-Instruct____2025-03-20-22-59-39_0/checkpoints/checkpoints/Poker-R1/Qwen2.5-pokerbench-plan-temp1-rollout32-2025-03-20-23-10-21/ $DEFAULT_LOCAL_DIR
aws s3 cp --recursive --quiet s3://linminh/training_logs/rl4search/Qwen-Qwen2.5-7B____2025-03-27-04-32-04_0/checkpoints/checkpoints/

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-06 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.temperature=$TEMP_VALUE \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$ROLLOUT_COUNT \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='Poker-R1' \
    trainer.resume_mode='auto' \
    trainer.resume_from_path=False \
    trainer.del_local_ckpt_after_load=True \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
