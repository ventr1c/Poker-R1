import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from datasets import load_dataset, load_from_disk

# Path to the dataset
dataset_path = "/workdir/saved-datasets/PokerBench-Small"


# Load the dataset from Hugging Face
dataset = load_from_disk(dataset_path)

def analyze_distribution(data, split_name):
    # Extract answers/labels
    answers = [item.get('output') for item in data]
    # print(answers)
    answers = [_parse_poker_decision(answer)[0] for answer in answers]
    # print(answers)
    # Count frequency of each answer
    answer_counts = Counter(answers)
    
    # Print distribution
    print(f"\n{split_name} Distribution:")
    for answer, count in answer_counts.most_common():
        percentage = count / len(answers) * 100
        print(f"{answer}: {count} ({percentage:.2f}%)")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    pd.Series(answer_counts).plot(kind='bar')
    plt.title(f'Answer Distribution in {split_name} Split')
    plt.tight_layout()
    plt.savefig(f"{split_name}_distribution.png")
    plt.show()

def _parse_poker_decision(normalized_decision):
    """Parse a normalized poker decision into action type and amount.
    
    Args:
        normalized_decision: A normalized poker decision string (e.g., "bet 50", "raise 100")
        
    Returns:
        Tuple of (action_type, amount) where amount is None for check/fold/call
    """
    parts = normalized_decision.split()
    action_type = parts[0] if parts else ""
    
    amount = None
    if action_type in ["bet", "raise"] and len(parts) > 1:
        try:
            amount = float(parts[1].replace("$", ""))
        except ValueError:
            pass
    return action_type, amount


# Analyze both splits
analyze_distribution(dataset["train"], "Train")
analyze_distribution(dataset["test"], "Test")

# Compare distributions
print("\nComparison of distributions:")
train_answers = Counter([item.get('answer') for item in dataset["train"]])
test_answers = Counter([item.get('answer') for item in dataset["test"]])

all_answers = set(list(train_answers.keys()) + list(test_answers.keys()))
for answer in sorted(all_answers):
    train_pct = train_answers.get(answer, 0) / len(dataset["train"]) * 100
    test_pct = test_answers.get(answer, 0) / len(dataset["test"]) * 100
    print(f"{answer}: Train {train_pct:.2f}%, Test {test_pct:.2f}%")