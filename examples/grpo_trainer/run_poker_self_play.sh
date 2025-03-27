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
ROLLOUT_COUNT=2  # Default value
TEMP_VALUE=1.0    # Default value
DATA_SOURCE_VALUE="pokerbench-plan"  # Default value
MODEL_TYPE=${MODEL_TYPE:-"Qwen-Qwen2.5-7B-Instruct"}

RAY_ADDRESS=${RAY_ADDRESS:-"auto"}
# Poker game settings
# NUM_GAMES=${NUM_GAMES:-100}
# TEMP_VALUE=${TEMP_VALUE:-0.8}
# MODEL_TYPE=${MODEL_TYPE:-"Qwen-Qwen2.5-7B-Instruct"}
# USE_RANDOM_OPPONENT=${USE_RANDOM_OPPONENT:-"false"}
MAIN_PLAYER_IDX=${MAIN_PLAYER_IDX:-0}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --rollout-n) ROLLOUT_COUNT="$2"; shift ;;
        --temperature) TEMP_VALUE="$2"; shift ;;
        --data-source) DATA_SOURCE_VALUE="$2"; shift ;;
        --model-type) MODEL_TYPE="$2"; shift ;;
        --num-games) NUM_GAMES="$2"; shift ;;
        --random-opponent) USE_RANDOM_OPPONENT="$2"; shift ;;
        --main-player) MAIN_PLAYER_IDX="$2"; shift ;;
        --ray-address) RAY_ADDRESS="$2"; shift ;;
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
EXPERIMENT_NAME="${MODEL_NAME}-poker-self-play-temp${TEMP_VALUE}-rollout${ROLLOUT_COUNT}-${JOB_NAME}"
EXPERIMENT_NAME="${MODEL_TYPE}-poker-self-play-temp${TEMP_VALUE}-games${NUM_GAMES}-${JOB_NAME}"
DEFAULT_LOCAL_DIR="/workdir/project-search/rl4search/checkpoints/Poker-Self-Play/${EXPERIMENT_NAME}"


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



python3 -m verl.trainer.main_ppo_sp \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-06 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.temperature=$TEMP_VALUE \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=$ROLLOUT_COUNT \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    self_play.do_self_play=True \
    self_play.max_prompt_length=2048 \
    self_play.max_response_length=2048 \
    self_play.max_turns=4 \
    self_play.max_players=6 \
    self_play.small_blind=1 \
    self_play.big_blind=2 \
    self_play.initial_stack=200 \
    self_play.batch_size=4 \
    self_play.seed=42 \
    self_play.iterations_per_epoch=100 \
    self_play.use_separate_opponent=True \
    self_play.main_player_idx=0 \
    self_play.update_opponent_every=10 \
    self_play.use_format_reward=False \
    self_play.format_reward_weight=0.2 \
    self_play.tag_count_reward_weight=0.2 \
    self_play.use_win_reward=true \
    self_play.win_reward_weight=1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='Poker-R1' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
