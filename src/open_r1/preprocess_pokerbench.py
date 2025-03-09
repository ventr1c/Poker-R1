#!/usr/bin/env python3
"""
Preprocess the PokerBench dataset for GRPO training.

This script:
1. Loads the PokerBench dataset from HuggingFace
2. Reformats instructions to work with GRPO's <think> and <answer> pattern
3. Stores the modified dataset for training

Usage:
  python preprocess_pokerbench.py --output_dataset_name processed-pokerbench
"""

import argparse
import os
from datasets import load_dataset, Dataset

def preprocess_pokerbench(
    dataset_name="RZ412/PokerBench", 
    output_dataset_name="processed-pokerbench",
    push_to_hub=True,
    num_proc=8,
    sample_size=None
):
    """
    Preprocess the PokerBench dataset for GRPO training.
    
    Args:
        dataset_name: Name of the original dataset on HuggingFace
        output_dataset_name: Name to save the processed dataset as
        push_to_hub: Whether to push the processed dataset to HuggingFace Hub
        num_proc: Number of parallel processes for preprocessing
        sample_size: If provided, use only this many examples (for testing)
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # Display dataset info before processing
    print("\nOriginal dataset structure:")
    print(dataset)
    
    # Show a few examples from the original dataset
    print("\nExample from original dataset:")
    example = dataset['train'][0]
    print(f"Instruction: {example['instruction']}")
    print(f"Output: {example['output']}")
    
    # Define preprocessing function
    def format_for_grpo(example):
        """Format each example for GRPO training."""
        # Get original instruction and output
        original_instruction = example["instruction"]
        gto_decision = example["output"]
        
        # Add explicit instructions for the desired format
        formatted_instruction = original_instruction + "\n\nFirst, think step by step about the hand strength, board texture, and your opponent's likely range. Then provide your final decision. Do not explain after your decision."
        
        # Create a formatted response with <think> and <answer> tags
        # This serves as an example for supervised learning, if needed
        formatted_response = f"<think>\nAnalyzing this poker situation carefully...\n</think>\n<answer>\n{gto_decision}\n</answer>"
        
        # Return the preprocessed example
        return {
            "instruction": formatted_instruction,
            "output": gto_decision,  # Keep the original GTO decision for reward calculation
            "formatted_response": formatted_response,  # Optional: for supervised fine-tuning
        }
    
    # Apply preprocessing to all splits
    processed_dataset = {}
    for split in dataset:
        # Take a sample if requested (for testing)
        if sample_size is not None:
            dataset_split = dataset[split].select(range(min(sample_size, len(dataset[split]))))
        else:
            dataset_split = dataset[split]
            
        # Apply the preprocessing function
        processed_split = dataset_split.map(
            format_for_grpo,
            num_proc=num_proc,
            desc=f"Preprocessing {split} split"
        )
        processed_dataset[split] = processed_split
    
    # Create a new dataset object
    processed_dataset = Dataset.from_dict({
        split: {
            col: [processed_dataset[split][i][col] for i in range(len(processed_dataset[split]))]
            for col in processed_dataset[split].column_names
        }
        for split in processed_dataset
    })
    
    # Show an example from the processed dataset
    print("\nExample from processed dataset:")
    example = processed_dataset['train'][0]
    print(f"Instruction: {example['instruction']}")
    print(f"Output: {example['output']}")
    print(f"Formatted Response: {example['formatted_response']}")
    
    # Save the processed dataset
    if push_to_hub:
        print(f"\nPushing processed dataset to Hub as: {output_dataset_name}")
        processed_dataset.push_to_hub(output_dataset_name)
    else:
        # Save locally
        output_dir = f"data/{output_dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        processed_dataset.save_to_disk(output_dir)
        print(f"\nSaved processed dataset locally to: {output_dir}")
    
    print("Preprocessing complete!")
    return processed_dataset

def main():
    parser = argparse.ArgumentParser(description="Preprocess PokerBench dataset for GRPO training")
    parser.add_argument("--dataset_name", type=str, default="RZ412/PokerBench", 
                        help="Name of dataset on HuggingFace")
    parser.add_argument("--output_dataset_name", type=str, default="processed-pokerbench",
                        help="Name to save processed dataset as")
    parser.add_argument("--local_only", action="store_true", 
                        help="Save dataset locally instead of pushing to Hub")
    parser.add_argument("--num_proc", type=int, default=8,
                        help="Number of parallel processes for preprocessing")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="If provided, use only this many examples (for testing)")
    
    args = parser.parse_args()
    
    preprocess_pokerbench(
        dataset_name=args.dataset_name,
        output_dataset_name=args.output_dataset_name,
        push_to_hub=not args.local_only,
        num_proc=args.num_proc,
        sample_size=args.sample_size
    )

if __name__ == "__main__":
    main()