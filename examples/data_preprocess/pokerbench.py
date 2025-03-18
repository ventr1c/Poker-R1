"""
Preprocess PokerBench dataset for poker decision making task
"""

import os
import argparse
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/pokerbench-instruct')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--input_path', default='/workdir/saved-datasets/PokerBench-Parquet')
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--test_size', type=int, default=None)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')

    args = parser.parse_args()

    data_source = 'pokerbench'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load the dataset from parquet files if they exist, otherwise from disk
    try:
        print(f"Attempting to load Parquet files from {args.input_path}")
        raw_dataset = DatasetDict({
            'train': Dataset.from_parquet(os.path.join(args.input_path, 'train.parquet')),
            'test': Dataset.from_parquet(os.path.join(args.input_path, 'test.parquet'))
        })
    except:
        print(f"Loading from HuggingFace dataset disk format at {args.input_path}")
        raw_dataset = load_from_disk(args.input_path)

    print(f"Dataset loaded: {raw_dataset}")
    print(f"Train samples: {len(raw_dataset['train'])}")
    print(f"Test samples: {len(raw_dataset['test'])}")

    # If train_size is specified, use only that many samples
    if TRAIN_SIZE is not None and TRAIN_SIZE < len(raw_dataset['train']):
        # Random sample to specified size
        train_indices = random.sample(range(len(raw_dataset['train'])), TRAIN_SIZE)
        train_dataset = raw_dataset['train'].select(train_indices)
    else:
        train_dataset = raw_dataset['train']

    # If test_size is specified, use only that many samples
    if TEST_SIZE is not None and TEST_SIZE < len(raw_dataset['test']):
        # Random sample to specified size
        test_indices = random.sample(range(len(raw_dataset['test'])), TEST_SIZE)
        test_dataset = raw_dataset['test'].select(test_indices)
    else:
        test_dataset = raw_dataset['test']

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    def make_prefix(instruction, template_type):
        """Create the appropriate prefix based on template type."""
        if template_type == 'base':
            """This works for any base model"""
            prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    User: You are a helpful AI Assistant that provides expert poker analysis. First think about the reasoning process as an internal monologue, then provide the optimal GTO decision. Respond in the following format:\n<think>\n[Your step-by-step reasoning process. ]\n</think>\n<answer>\n[Your final decision in the format: check/call/fold/bet X/raise X]\n</answer>. {instruction}
    Assistant:"""
        elif template_type == 'base-2':
            """Simplified format without conversation framing"""
            prefix = f"""You are a helpful AI Assistant that provides expert poker analysis. First think about the reasoning process as an internal monologue, then provide the optimal GTO decision. Respond in the following format:\n<think>\n[Your step-by-step reasoning process. ]\n</think>\n<answer>\n[Your final decision in the format: check/call/fold/bet X/raise X]\n</answer> 
    {instruction}"""
        elif template_type == 'base-3':
            prefix = f"""
    A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    User: You are a helpful AI Assistant that provides expert poker analysis. Your task is to analyze the poker situation and determine the optimal GTO decision. 1. First, conduct a step-by-step reasoning phase within <analyze> and </analyze> tags where you analyze the current game situation (e.g., board texture, positions, stack sizes, etc.). 2. Next, enter a planning phase inside <plan> and </plan> where you explore multiple potential strategies. During this phase, list possible actions without committing to a single decision. Highlight any uncertainties or multiple viable options. 3. If you require further knowledge to refine your planning, call a game theory calculation function inside <calculation> and </calculation>. These calculations can include, but are not limited to: - Range estimation for you and your opponents. - Detailed estimations of hand strengths. - Expected value calculations for different actions. - Mixed strategy considerations (since GTO may require playing the same hand in different ways with specific frequencies). - Application of the 4-2 Rule. 4. You may call the <calculation> function multiple times. After each calculation, return to the planning phase (<plan> ... </plan>) to reassess your options—either confirming your earlier considerations or revising them based on new insights. 5. Only once your iterative planning and calculation process has fully refined your decision should you provide your final action. Output your final decision in the following format: <answer> [Your final decision in the format: check/call/fold/bet X/raise X] </answer>. For bet or raise decisions, ensure that X is the only numerical value in your output, for example, <answer> raise 10 </answer> or <answer> bet 100 </answer>.
    {instruction}
    Assistant:"""
        elif template_type == 'qwen-instruct':
            """This works for Qwen Instruct Models"""
            prefix = f"""<|im_start|>system
    You are a helpful AI Assistant that provides expert poker analysis. Your task is to analyze the poker situation and determine the optimal GTO decision. 1. First, conduct a step-by-step reasoning phase within <analyze> and </analyze> tags where you analyze the current game situation (e.g., board texture, positions, stack sizes, etc.). 2. Next, enter a planning phase inside <plan> and </plan> where you explore multiple potential strategies. During this phase, list possible actions without committing to a single decision. Highlight any uncertainties or multiple viable options. 3. If you require further knowledge to refine your planning, call a game theory calculation function inside <calculation> and </calculation>. These calculations can include, but are not limited to: - Range estimation for you and your opponents. - Detailed estimations of hand strengths. - Expected value calculations for different actions. - Mixed strategy considerations (since GTO may require playing the same hand in different ways with specific frequencies). - Application of the 4-2 Rule. 4. You may call the <calculation> function multiple times. After each calculation, return to the planning phase (<plan> ... </plan>) to reassess your options—either confirming your earlier considerations or revising them based on new insights. 5. Only once your iterative planning and calculation process has fully refined your decision should you provide your final action. Output your final decision in the following format: <answer> [Your final decision in the format: check/call/fold/bet X/raise X] </answer>. For bet or raise decisions, ensure that X is the only numerical value in your output, for example, <answer> raise 10 </answer> or <answer> bet 100 </answer>.<|im_end|>
    <|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant"""
        else:
            # Default to raw instruction if template type not recognized
            prefix = instruction
        
        return prefix

    def make_map_fn(split):
        def process_fn(example, idx):
            # Extract the instruction and output
            instruction = example['instruction']
            formatted_instruction = make_prefix(instruction, template_type=args.template_type)
            ground_truth = example['output']
            
            # Create the data dict in the required format
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": instruction,
                }],
                "ability": "decision_making",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn
    
    # Process the datasets
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Prepare output directories
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    # Save to parquet format
    print(f"Saving processed dataset to {local_dir}")
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # Copy to HDFS if specified
    if hdfs_dir is not None:
        print(f"Copying to HDFS: {hdfs_dir}")
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
    
    print("Dataset processing complete!")