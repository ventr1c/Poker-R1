#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import subprocess
from typing import List

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from .evaluation import run_benchmark_jobs
from .hub import push_to_hub_revision
import numpy as np


def is_slurm_available() -> bool:
    # returns true if a slurm queueing system is available
    try:
        subprocess.run(["sinfo"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


class DummyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PushToHubRevisionCallback(TrainerCallback):
    def __init__(self, model_config) -> None:
        self.model_config = model_config

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            global_step = state.global_step

            # WARNING: if you use dataclasses.replace(args, ...) the accelerator dist state will be broken, so I do this workaround
            # Also if you instantiate a new SFTConfig, the accelerator dist state will be broken
            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=f"{args.hub_model_revision}-step-{global_step:09d}",
                output_dir=f"{args.output_dir}/checkpoint-{global_step}",
                system_prompt=args.system_prompt,
            )

            future = push_to_hub_revision(
                dummy_config, extra_ignore_patterns=["*.pt"]
            )  # don't push the optimizer states

            if is_slurm_available():
                dummy_config.benchmarks = args.benchmarks

                def run_benchmark_callback(_):
                    print(f"Checkpoint {global_step} pushed to hub.")
                    run_benchmark_jobs(dummy_config, self.model_config)

                future.add_done_callback(run_benchmark_callback)

class RewardLoggingCallback(TrainerCallback):
    def __init__(self, reward_funcs, reward_func_names):
        self.reward_funcs = reward_funcs
        self.reward_func_names = reward_func_names
        self.log_interval = 1  # 每步都记录
        self.reward_history = {name: [] for name in reward_func_names}
        self.reward_history["total"] = []
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_interval != 0:
            return
            
        # 尝试获取最近生成的响应
        try:
            trainer = kwargs.get('trainer', None)
            if not trainer or not hasattr(trainer, 'all_generated_texts') or not hasattr(trainer, 'all_prompts_text'):
                print(f"Step {state.global_step}: Trainer missing required attributes", file=sys.stderr)
                return
                
            if len(trainer.all_generated_texts) == 0:
                print(f"Step {state.global_step}: No generated texts available", file=sys.stderr)
                return
                
            # 记录每个样本的平均奖励
            avg_rewards = {name: 0.0 for name in self.reward_func_names}
            avg_total = 0.0
            count = 0
            
            for idx in range(min(5, len(trainer.all_generated_texts))):
                prompt = trainer.all_prompts_text[idx] if idx < len(trainer.all_prompts_text) else ""
                response = trainer.all_generated_texts[idx] if idx < len(trainer.all_generated_texts) else ""
                
                if not prompt or not response:
                    continue
                    
                total = 0.0
                for i, reward_func in enumerate(self.reward_funcs):
                    name = self.reward_func_names[i]
                    try:
                        value = reward_func(prompt, response)
                        avg_rewards[name] += value
                        total += value
                    except Exception as e:
                        print(f"Error calculating {name} reward: {e}", file=sys.stderr)
                
                avg_total += total
                count += 1
            
            if count > 0:
                # 计算平均值
                for name in avg_rewards:
                    avg_rewards[name] /= count
                avg_total /= count
                
                # 记录到wandb
                log_dict = {f"reward/{name}": avg_rewards[name] for name in self.reward_func_names}
                log_dict["reward/total"] = avg_total
                
                # 保存到历史记录
                for name in self.reward_func_names:
                    self.reward_history[name].append(avg_rewards[name])
                self.reward_history["total"].append(avg_total)
                
                if args.report_to and "wandb" in args.report_to:
                    import wandb
                    wandb.log(log_dict, step=state.global_step)
                    print(f"Step {state.global_step}: Logged rewards to wandb", file=sys.stderr)
                else:
                    print(f"Step {state.global_step}: rewards={log_dict}", file=sys.stderr)
            
        except Exception as e:
            print(f"Error in reward logging: {e}", file=sys.stderr)

CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
    # "reward_logging": RewardLoggingCallback,
}


def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    callbacks = []
    for callback_name in train_config.callbacks:
        if callback_name not in CALLBACKS:
            raise ValueError(f"Callback {callback_name} not found in CALLBACKS.")
        callbacks.append(CALLBACKS[callback_name](model_config))

    return callbacks
