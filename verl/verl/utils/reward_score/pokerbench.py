import re
import random
import ast
import operator


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

def _normalize_poker_decision(decision):
    """Normalize poker decisions for comparison.
    
    For example, "raise 3bb" and "raise 3 bb" should be considered the same.
    This needs to be adapted based on the exact format in the PokerBench dataset.
    """
    # Remove extra whitespace and standardize common terms
    decision = decision.strip()
    decision = decision.replace(" bb", "bb")
    decision = decision.replace(" big blinds", "bb")
    
    # Handle common action formats
    if "fold" in decision:
        return "fold"
    elif "check" in decision:
        return "check"
    elif "call" in decision:
        return "call"
    
    # Handle raises with sizing
    if "raise" in decision or "bet" in decision:
        # Extract the action and size if present
        sizing_match = re.search(r'(raise|bet)\s*([\d.]+)\s*bb', decision)
        if sizing_match:
            action = sizing_match.group(1)
            size = sizing_match.group(2)
            return f"{action} {size}"
    
    # Return as-is if no normalization rules match
    return decision


def poker_gto_reward(solution_str, ground_truth, method, format_score, score):
    """Reward function that checks if the model's poker decision matches the GTO decision.
    Gives partial credit (0.5) when action type matches but amount differs.
    
    Args:
        completions: List of model completions (each containing a list of message dictionaries)
        solution: List of ground truth GTO decisions
        
    Returns:
        List of rewards where:
        - 1.0 means the model's decision fully matches the GTO decision
        - 0.5 means the action type matches but amount differs
        - 0.0 means the action type doesn't match or couldn't be parsed
    """
    contents = [solution_str]
    ground_truths = [ground_truth]
    rewards = []
    
    for content, gto_decision in zip(contents, ground_truths):
        try:
            # Extract the decision from the response (assuming it's in the <answer> section)
            answer_pattern = r"<answer>\s*([\s\S]*?)\s*</answer>"
            answer_match = re.search(answer_pattern, content, re.DOTALL)
            
            if not answer_match:
                rewards.append(0.0)
                continue
                
            answer_section = answer_match.group(1).strip().lower()
            
            # Normalize the model's answer and the reference for comparison
            print("answer_section", answer_section, "gto_decision", gto_decision)
            normalized_response = _normalize_poker_decision(answer_section)
            normalized_reference = _normalize_poker_decision(gto_decision.lower())
            
            # Parse the decisions to get action type and amount
            response_action, response_amount = _parse_poker_decision(normalized_response)
            reference_action, reference_amount = _parse_poker_decision(normalized_reference)
            
            # Determine reward based on match criteria
            if response_action == reference_action:
                if response_action in ["check", "fold", "call"]:
                    # For actions without amounts, exact match means full reward
                    reward = 1.0
                elif response_action in ["bet", "raise"]:
                    # For bet/raise, compare amounts if available
                    if response_amount is not None and reference_amount is not None:
                        # Full reward for exact amount match
                        if abs(response_amount - reference_amount) < 2:  # Small epsilon for float comparison
                            reward = 1.5
                        else:
                            # Partial reward (0.5) when only action type matches
                            reward = 1.0
                    else:
                        # If we can't parse one of the amounts, give partial credit
                        reward = 0.5
                else:
                    # Unknown action type
                    reward = 0.0
            else:
                # Action types don't match
                reward = 0.0
            
        except (IndexError, ValueError, AttributeError) as e:
            # If we can't parse the answer section properly, return 0
            print(f"Error in poker_gto_reward: {e}")
            reward = 0.0
            
        rewards.append(reward)
    return rewards[0]

def poker_tag_count_reward(solution_str, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of tags required by the poker system prompt.
    
    Checks for <analyze>, </analyze>, <plan>, </plan>, <calculation>, </calculation>, <answer>, </answer> tags.
    """
    def count_tags(text: str) -> float:
        count = 0.0
        # Required tags - each worth 0.125 (for total of 0.75 for required tags)
        if text.count("<analyze>") == 1:
            count += 0.125
        if text.count("</analyze>") == 1:
            count += 0.125
        if text.count("<plan>") >= 1:  # Allow multiple plan sections
            count += 0.125
        if text.count("</plan>") >= 1 and text.count("</plan>") == text.count("<plan>"):
            count += 0.125
        if text.count("<answer>") == 1:
            count += 0.125
        if text.count("</answer>") == 1:
            count += 0.125
            
        # Optional calculation tags - worth 0.25 together
        # Check if calculation tags exist and are balanced
        if text.count("<calculation>") > 0 and text.count("<calculation>") == text.count("</calculation>"):
            count += 0.25
            
        # Check sequence: each calculation must be followed by a plan
        proper_sequence = True
        
        # Extract all tag positions
        tag_positions = []
        for tag in ["<analyze>", "</analyze>", "<plan>", "</plan>", "<calculation>", "</calculation>", "<answer>", "</answer>"]:
            positions = [m.start() for m in re.finditer(re.escape(tag), text)]
            for pos in positions:
                tag_positions.append((pos, tag))
        
        # Sort tags by position
        tag_positions.sort()
        
        # Check if it starts with <think> and ends with </answer>
        if len(tag_positions) >= 2:
            if tag_positions[0][1] != "<analyze>" or tag_positions[-1][1] != "</answer>":
                proper_sequence = False
        
        # Check calculation-plan sequence
        for i in range(len(tag_positions) - 1):
            if tag_positions[i][1] == "</calculation>":
                next_tag = tag_positions[i+1][1]
                if next_tag != "<plan>":
                    proper_sequence = False
                    break
        
        # Add extra reward if proper sequence
        if proper_sequence:
            count += 0.125
            
        if count > 0:
            print("tag_count", count)
        return count

    return count_tags(solution_str)

def poker_format_reward(solution_str, **kwargs):
    """
    Reward function that checks if the response follows the structure required by the poker system prompt:
    1. Must start with <analyze> and </analyze> tags
    2. Followed by <plan> and </plan> tags
    3. Optional <calculation></calculation> sections that must be followed by <plan></plan>
    4. End with <answer></answer> tags containing a properly formatted poker decision
    
    Returns:
    - 1.0 if structure is completely correct and poker decision format is correct
    - 0.5 if structure is correct but poker decision format is incorrect
    - 0.3 if sequence is wrong (calculation not followed by plan)
    - 0.0 if basic structure is incorrect
    """
    # Check for proper tag ordering - starts with think, then at least one plan, then answer
    
    start_pattern = r"^<analyze>.*?</analyze>\s*<plan>"
    end_pattern = r"</plan>.*?<answer>.*?</answer>$"
    
    # Check that the answer contains a properly formatted poker decision
    poker_answer_pattern = r"<answer>\s*(check|call|fold|bet\s+\d+(\.\d+)?(\s*bb)?|raise\s+\d+(\.\d+)?(\s*bb)?)\s*</answer>"
    
    completion_contents = [solution_str]
    
    rewards = []
    for i, content in enumerate(completion_contents):
        print(f"--- Completion {i} ---")
        print(content)
        print("-------------------")        
        # Clean whitespace and normalize for easier pattern matching
        normalized_content = re.sub(r'\s+', ' ', content).strip()
        
        # Check if content starts with think and has at least one plan and ends with answer
        starts_correctly = re.match(start_pattern, normalized_content, re.DOTALL) is not None
        ends_correctly = re.search(end_pattern, normalized_content, re.DOTALL) is not None
        
        if not (starts_correctly and ends_correctly):
            # Basic structure is wrong - doesn't follow think → plan → answer
            rewards.append(0.0)
            continue
            
        # Check if every calculation is immediately followed by a plan section
        calculations = re.findall(r"<calculation>.*?</calculation>", content, re.DOTALL)
        all_calculations_followed_by_plan = True
        
        for calc in calculations:
            # Find where this calculation ends in the content
            calc_end_pos = content.find("</calculation>", content.find(calc)) + len("</calculation>")
            
            # Check if there's an immediate plan section after this calculation
            # Allow only whitespace between </calculation> and <plan>
            plan_after_calc = re.match(r"\s*<plan>", content[calc_end_pos:]) is not None
            
            if not plan_after_calc:
                all_calculations_followed_by_plan = False
                break
        
        # Check if answer format is correct
        poker_match = re.search(poker_answer_pattern, content, re.DOTALL | re.IGNORECASE)
        
        # Determine reward based on checks
        if not all_calculations_followed_by_plan:
            # Sequence is wrong - calculation not followed by plan
            rewards.append(0.3)
        elif poker_match:
            # Everything is correct
            rewards.append(1.0)
        else:
            # Structure is correct but poker format is wrong
            rewards.append(0.5)
        
    return rewards[0]

def compute_tag_count_reward(solution_str, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        if(count>0):
            print("tag_count", count)
        return count

    contents = []
    contents.append(solution_str)

    return count_tags(solution_str)

def compute_format_reward(solution_str, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, solution_str, re.DOTALL | re.MULTILINE)]
    # return [1.0 if match else 0.0 for match in matches]
    return 1.0 if re.match(pattern, solution_str, re.DOTALL | re.MULTILINE) else 0.0

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    accuracy_reward = poker_gto_reward(solution_str, ground_truth, method, format_score, score)
    format_reward = poker_format_reward(solution_str)
    tag_count_reward = poker_tag_count_reward(solution_str)
    # return 1 * accuracy_reward + 0.5 * format_reward + 0.5 * tag_count_reward
    # return accuracy_reward + 0 * format_reward + 0 * tag_count_reward
    return accuracy_reward