"""Reward functions for GRPO training."""

import json
import math
import re
from typing import Dict

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils import is_e2b_available


if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import Sandbox

    load_dotenv()


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
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

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]

def poker_tag_count_reward(completions, **kwargs) -> list[float]:
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

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]

def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def extract_code(completion: str) -> str:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def code_reward(completions, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    rewards = []
    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    try:
        """Returns a reward function that evaluates code snippets in a sandbox."""
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
        verification_info = kwargs["verification_info"]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"]))
            )
            for code, info in zip(code_snippets, verification_info)
        ]
        with Sandbox(timeout=30, request_timeout=3) as sbx:
            for script in scripts:
                execution = sbx.run_code(script, language=verification_info["language"])
                try:
                    output = float(execution.text)
                except (TypeError, ValueError):
                    output = 0.0
                rewards.append(output)
    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)
    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


# Add the following function to the existing rewards.py file

def poker_gto_reward(completions, solution, **kwargs):
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
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, gto_decision in zip(contents, solution):
        try:
            # Extract the decision from the response (assuming it's in the <answer> section)
            answer_pattern = r"<answer>\n(.*?)\n</answer>"
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
                        if abs(response_amount - reference_amount) < 0.01:  # Small epsilon for float comparison
                            reward = 1.0
                        else:
                            # Partial reward (0.5) when only action type matches
                            reward = 0.5
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
        
    return rewards

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
            return f"{action} {size}bb"
    
    # Return as-is if no normalization rules match
    return decision

def poker_format_reward(completions, **kwargs):
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
    
    completion_contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for content in completion_contents:
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
        
    return rewards
