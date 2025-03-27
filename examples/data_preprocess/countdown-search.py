"""
Preprocess dataset for countdown task - given a target number and N numbers, generate equations to reach target
"""

import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


def gen_dataset(
    num_samples: int,
    num_operands: int = 6,
    max_target: int = 1000,
    min_number: int = 1,
    max_number: int = 100,
    operations: List[str] = ['+', '-', '*', '/'],
    seed_value: int = 42,
) -> List[Tuple]:
    """Generate dataset for countdown task.
    
    Args:
        num_samples: Number of samples to generate
        num_operands: Number of numbers provided in each sample
        max_target: Maximum value for target number
        min_number: Minimum value for provided numbers
        max_number: Maximum value for provided numbers
        operations: List of allowed operations
        seed_value: Random seed for reproducibility
        
    Returns:
        List of tuples containing (target, numbers, solution)
    """
    seed(seed_value)
    samples = []
    
    for _ in tqdm(range(num_samples)):
        # Generate random target
        target = randint(1, max_target)
        
        # Generate random numbers
        numbers = [randint(min_number, max_number) for _ in range(num_operands)]
        
        
        samples.append((target, numbers))
    
    return samples

# def make_prefix(dp, template_type):
#     target = dp['target']
#     numbers = dp['nums']
#     # NOTE: also need to change reward_score/countdown.py
#     if template_type == 'base':
#         """This works for any base model"""
#         prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
# User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
# Assistant: Let me solve this step by step.
# <think>"""
#     elif template_type == 'base-2':
#         prefix = f"""Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags. For example, <answer> (1 + 2) / 3 </answer>.
# Let me see if I can solve this step by step.
# <think>"""
#     elif template_type == 'self-search-base':
#         prefix = f"""Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags. For example, <answer> (1 + 2) / 3 </answer>.
# Let me see if I can solve this step by step.
# <think>"""
#     elif template_type == 'qwen-instruct':
#         """This works for Qwen Instruct Models"""
#         prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
#     return prefix

def make_system_prompt(template_type):
    if template_type == 'self-search-base':
        return """You are a helpful AI Assistant that can search for the best equation to reach the target number using the given numbers. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your reasoning process in <think> </think> tags. 
        During the search, you can follow the following steps in the reasoning process:
        1. Direct Solve: First, attempt a direct solution without any decomposition. In this step, DO NOT backtrack, directly give me the correct answer. DO NOT explicitly verify your answer. DIRECTLY give me the solution. DO NOT produce any other outputs.
        2. Verification: Verify if the direct solution is correct. If it is, return the solution, and stop the thinking process. If it is not, proceed to the next step.
        3. Decomposition: If verification fails, decompose the problem into several subproblems.
        4. Self-Ranking: Once subproblems are identified, rank them by assigning scores to determine the order of exploration. Begin by exploring the subproblems with the highest score.
        5. Backtracking: If you reach the deepest subproblem without finding a solution, backtrack to explore any previously unvisited states.
        And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."""
    elif template_type == 'self-search-tag':
        return """
        You are a helpful AI Assistant that can search for the best equation to reach a target number using the given numbers. You can use the basic arithmetic operations (+, -, *, /), and each number can only be used once. Your reasoning process must be structured into several clearly defined stages, each enclosed in its own special tags:
        1. Direct Solve Stage (<solve> … </solve>)
        - In this stage, attempt to provide a direct solution without any decomposition.
        - Do not provide any extra commentary or additional outputs—simply output the candidate solution.
        2. Verification Stage (<verify> … </verify>)
        - Verify the correctness of your direct solution.
        - If the solution is correct, output the final answer in the format:
        <answer> [Your final solution] </answer> and stop any further reasoning.
        - If the solution is not correct, proceed to the next stage.
        3. Decomposition Stage (<decompose> … </decompose>)
        - Break down the problem into several subproblems that can be solved individually.
        4. Self-Ranking Stage (<rank> … </rank>)
        - Assign scores to the identified subproblems to determine the order in which to explore them.
        - Start by exploring the subproblems with the highest score.
        5. Backtracking Stage (<backtracking> … </backtracking>)
        - If you reach a dead end with a given subproblem, use backtracking to explore any previously unvisited states.
        
        Once you have fully refined your solution through these stages, provide your final answer in the following format:
        <answer> [Your final equation, e.g., (1 + 2) / 3] </answer>
        """

def make_user_prompt(dp, template_type):
    target = dp['target']
    numbers = dp['nums']

    prompt = f"""Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."""
    
    return prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/countdown-search-tag')
    parser.add_argument('--cot_dir', type=str, default=None, required=False)
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--template_type', type=str, default='self-search-tag')

    args = parser.parse_args()

    data_source = 'countdown'
    cot_data = args.cot_dir is not None
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    if not cot_data:
        raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')
    else:
        raw_dataset = load_dataset('json', data_files={'train': args.cot_dir})['train']

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    pattern = r"\[((?:-?\d+(?:\s*,\s*-?\d+)*))\]\s*and\s*target\s*(-?\d+)"

    def make_map_fn(split):
        def process_fn(example, idx):
            if not cot_data:
                # question = make_prefix(example, template_type=args.template_type)
                question = make_user_prompt(example, template_type=args.template_type)
                solution = {
                    "target": example['target'],
                    "numbers": example['nums']
                }
                data = {
                    "data_source": data_source,
                    "prompt": [
                        {
                            "role": "system",
                            "content": make_system_prompt(args.template_type),
                        },
                        {
                            "role": "user",
                            "content": question,
                        }
                    ],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data
            else:
                question = example['query']
                match = re.search(pattern, question)               
                if not match: 
                    return None 
                nums = [int(num.strip() for num in match.group(1).split(','))]
                target = int(match.group(2))
                solution = {
                    "target": target,
                    "numbers": nums
                }
                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                        "answer": example['completion'],
                        "question": question
                    }
                }
                return data

        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)


    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 
