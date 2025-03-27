set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH=/workdir/project-search/rl4search/verl:$PYTHONPATH

WORLD_SIZE=$AWS_BATCH_JOB_NUM_NODES
if [ $WORLD_SIZE -gt 4 ]; then
    PSIZE=4
elif [ $WORLD_SIZE -gt 2 ]; then
    PSIZE=2
else
    PSIZE=$WORLD_SIZE
fi

# Use environment variables with fallback to default values
# Parse command line arguments
ROLLOUT_COUNT=32  # Default value
TEMP_VALUE=1.0    # Default value

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --rollout-n) ROLLOUT_COUNT="$2"; shift ;;
        --temperature) TEMP_VALUE="$2"; shift ;;
        *) shift ;;
    esac
    shift
done

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workdir/project-search/rl4search/data/pokerbench-none/train.parquet \
    data.val_files=/workdir/project-search/rl4search/data/pokerbench-none/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/workdir/pretrained-models/Qwen-Qwen2.5-7B \
    actor_rollout_ref.actor.optim.lr=1e-06 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.temperature=$TEMP_VALUE \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PSIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$ROLLOUT_COUNT \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='Poker-R1' \
    trainer.experiment_name='qwen2.5-7b-rollout32' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
