import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
# from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.protocol import pad_dataproto_to_divisor
import shutil
import requests

import clubs
import random
import numpy as np
from collections import defaultdict
from codetiming import Timer
from torch.utils.data import Dataset, DataLoader
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from torch.utils.data import RandomSampler, SequentialSampler
import pandas as pd
from torchdata.stateful_dataloader import StatefulDataLoader
from clubs.configs import PokerConfig
from .tensor_helper import TensorHelper, TensorConfig
import tensordict
from tensordict import TensorDict
from verl.utils.reward_score.pokerbench import poker_tag_count_reward, poker_format_reward

def _timer(name, timing_dict=None):
    """Context manager for timing code blocks."""
    timer = Timer(name=name, text="{name}: {:.4f}s", logger=None)
    if timing_dict is not None:
        timer.add_callback(lambda x: timing_dict.update({name: x}))
    return timer
    
# class PokerSelfPlayDataset(Dataset):
#     """
#     Dummy dataset that generates poker game starting states.
#     Instead of loading real data, this creates placeholders for poker games.
#     """
#     def __init__(self, num_games, tokenizer, max_prompt_length=512, seed=42):
#         self.num_games = num_games
#         self.tokenizer = tokenizer
#         self.max_prompt_length = max_prompt_length
#         self.rng = random.Random(seed)
        
#         # Create a base poker instruction prompt
#         self.system_prompt = "You are a helpful AI Assistant that provides expert poker analysis. Your task is to analyze the poker situation and determine the optimal GTO decision. 1. First, conduct a step-by-step reasoning phase within <analyze> and </analyze> tags where you analyze the current game situation (e.g., board texture, positions, stack sizes, etc.). 2. Next, enter a planning phase inside <plan> and </plan> where you explore multiple potential strategies. During this phase, list possible actions without committing to a single decision. Highlight any uncertainties or multiple viable options. 3. If you require further knowledge to refine your planning, call a game theory calculation function inside <calculation> and </calculation>. These calculations can include, but are not limited to: - Range estimation for you and your opponents. - Detailed estimations of hand strengths. - Expected value calculations for different actions. - Mixed strategy considerations (since GTO may require playing the same hand in different ways with specific frequencies). - Application of the 4-2 Rule. 4. You may call the <calculation> function multiple times. After each calculation, return to the planning phase (<plan> ... </plan>) to reassess your options—either confirming your earlier considerations or revising them based on new insights. 5. Only once your iterative planning and calculation process has fully refined your decision should you provide your final action. Output your final decision in the following format: <answer> [Your final decision in the format: check/call/fold/bet X/raise X] </answer>. For bet or raise decisions, ensure that X is the only numerical value in your output, for example, <answer> raise 10 </answer> or <answer> bet 100 </answer>."
    
#     def __len__(self):
#         return self.num_games
    
#     def __getitem__(self, idx):
#         # Create a chat message in the format expected by the tokenizer's chat template
#         chat = [
#             {"role": "system", "content": self.system_prompt}
#         ]
        
#         # Apply the chat template
#         prompt_with_chat_template = self.tokenizer.apply_chat_template(
#             chat, 
#             add_generation_prompt=True, 
#             tokenize=False
#         )
        
#         # Tokenize the formatted prompt
#         inputs = self.tokenizer(
#             prompt_with_chat_template,
#             padding="max_length", 
#             max_length=self.max_prompt_length, 
#             truncation=True, 
#             return_tensors="pt"
#         )
        
#         # Add game ID and raw prompt for tracking
#         result = {
#             "raw_prompt_ids": inputs.input_ids[0],
#             "input_ids": inputs.input_ids[0],
#             "attention_mask": inputs.attention_mask[0],
#             "position_ids": torch.arange(len(inputs.input_ids[0])),
#             "game_id": f"game_{idx}",
#             "raw_chat": chat  # Store the original chat message for later use
#         }
        
#         return result
class PokerSelfPlayDataset(Dataset):
    """
    Dummy dataset that generates poker game starting states.
    Instead of loading real data, this creates placeholders for poker games.
    """
    def __init__(self, num_games, tokenizer, max_prompt_length=512, seed=42):
        self.num_games = num_games
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.rng = random.Random(seed)
        
        # Create a base poker instruction prompt
        self.system_prompt = "You are a helpful AI Assistant that provides expert poker analysis. Your task is to analyze the poker situation and determine the optimal GTO decision. 1. First, conduct a step-by-step reasoning phase within <analyze> and </analyze> tags where you analyze the current game situation (e.g., board texture, positions, stack sizes, etc.). 2. Next, enter a planning phase inside <plan> and </plan> where you explore multiple potential strategies. During this phase, list possible actions without committing to a single decision. Highlight any uncertainties or multiple viable options. 3. If you require further knowledge to refine your planning, call a game theory calculation function inside <calculation> and </calculation>. These calculations can include, but are not limited to: - Range estimation for you and your opponents. - Detailed estimations of hand strengths. - Expected value calculations for different actions. - Mixed strategy considerations (since GTO may require playing the same hand in different ways with specific frequencies). - Application of the 4-2 Rule. 4. You may call the <calculation> function multiple times. After each calculation, return to the planning phase (<plan> ... </plan>) to reassess your options—either confirming your earlier considerations or revising them based on new insights. 5. Only once your iterative planning and calculation process has fully refined your decision should you provide your final action. Output your final decision in the following format: <answer> [Your final decision in the format: check/call/fold/bet X/raise X] </answer>. For bet or raise decisions, ensure that X is the only numerical value in your output, for example, <answer> raise 10 </answer> or <answer> bet 100 </answer>."
        
        # Create sample dataframe to mimic RLHFDataset structure
        self.dataframe = pd.DataFrame({
            'prompt': [self._create_initial_chat() for _ in range(num_games)],
            'game_id': [f"game_{i}" for i in range(num_games)]
        })
    
    def _create_initial_chat(self):
        """Create an initial chat message with system prompt and starter user message."""
        return [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, item):
        # Get row from dataframe, similar to RLHFDataset
        row_dict = self.dataframe.iloc[item].to_dict()
        
        # Extract the chat message
        chat = row_dict.pop('prompt')
        
        # Apply the chat template (same as RLHFDataset)
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # Tokenize (following RLHFDataset pattern)
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="error"
        )
        
        # Calculate position IDs like RLHFDataset
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Build return dictionary similar to RLHFDataset
        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)
        
        # Add raw_chat for consistency with other code
        row_dict['raw_chat'] = chat
        
        # Add index for each prompt
        row_dict["index"] = item
        
        return row_dict

# def create_self_play_dataloader(config, tokenizer):
#     """Create a dataloader specifically for poker self-play."""
#     train_dataset = PokerSelfPlayDataset(
#         num_games=config.self_play.batch_size, 
#         tokenizer=tokenizer,
#         max_prompt_length=config.self_play.max_prompt_length,
#         seed=config.self_play.seed
#     )
    
#     train_dataloader = DataLoader(
#         dataset=train_dataset,
#         batch_size=config.self_play.batch_size,
#         shuffle=False,  # No need to shuffle since we generate random games
#         num_workers=2,
#         collate_fn=lambda batch: {k: [item[k] for item in batch] for k in batch[0]}
#     )
    
#     return train_dataloader

def create_self_play_dataloader(config, tokenizer):
    """Create a dataloader specifically for poker self-play."""
    
    # Create the dataset
    train_dataset = PokerSelfPlayDataset(
        num_games=config.self_play.batch_size * config.self_play.iterations_per_epoch,  # Create enough games for the whole epoch
        tokenizer=tokenizer,
        max_prompt_length=config.self_play.max_prompt_length,
        seed=config.self_play.seed
    )
    
    # Create a sampler for better reproducibility
    train_dataloader_generator = torch.Generator()
    train_dataloader_generator.manual_seed(config.self_play.seed)
    sampler = SequentialSampler(data_source=train_dataset)
    
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    # Define a proper collate function that works with DataProto
    # def collate_fn(batch):
    #     """
    #     Properly collate batch items from the dataset into tensors.
    #     Matches the format expected by DataProto.from_single_dict
    #     """
    #     # Create output dictionary
    #     batch_dict = {
    #         'input_ids': torch.stack([item['input_ids'] for item in batch]),
    #         'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
    #         'position_ids': torch.stack([item['position_ids'] for item in batch]),
    #     }
        
    #     # Add non-tensor items to a nested dict
    #     batch_dict['non_tensor_batch'] = {
    #         'raw_prompt_ids': [item['raw_prompt_ids'] for item in batch],
    #         'raw_chat': [item['raw_chat'] for item in batch],
    #         'index': [item['index'] for item in batch]
    #     }
        
    #     return batch_dict
    
    # Create the dataloader using StatefulDataLoader
    # from verl.utils.dataset.util import StatefulDataLoader
    
    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.self_play.batch_size,
        num_workers=2,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler
    )
    
    return train_dataloader


@dataclass
class PokerGameState:
    """Represents the state of a poker game for generation."""
    call: int
    min_raise: int
    max_raise: int
    hole_cards: List
    community_cards: List
    pot: int
    stacks: List
    active: List
    action: int  # Current player index
    street_commits: List
    is_first_action: bool
    betting_round: int  # 0=preflop, 1=flop, 2=turn, 3=river
    button: int  # Position of the dealer button
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "call": self.call,
            "min_raise": self.min_raise,
            "max_raise": self.max_raise,
            "hole_cards": self.hole_cards,
            "community_cards": self.community_cards,
            "pot": self.pot,
            "stacks": self.stacks,
            "active": self.active,
            "action": self.action,
            "street_commits": self.street_commits,
            "is_first_action": self.is_first_action,
            "betting_round": self.betting_round,
            "button": self.button
        }

@dataclass
class PokerGenerationConfig:
    """Configuration for poker generation."""
    max_turns: int 
    max_prompt_length: int
    max_response_length: int
    max_players: int = 6
    small_blind: int = 5
    big_blind: int = 10
    initial_stack: int = 1000
    num_rollouts: int = 4  # Number of alternative actions to consider
    main_player_idx: int = 0
    opponent_model_path: str = None  # Path to opponent model checkpoint
    update_opponent_every: int = 5   # Update opponent model every N iterations
    max_start_length: int = 512      # Max length for prompt context
    temperature: float = 0.7
    top_p: float = 0.95
    system_prompt: str = None
    num_gpus: int = 8
    format_reward_weight: float = 0.5
    tag_count_reward_weight: float = 0.5

class PokerGenerationManager:
    """Manages LLM interactions for poker self-play."""
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: PokerGenerationConfig,
        is_validation: bool = False,
        opponent_rollout_wg=None,  # Optional separate worker group for opponent
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.opponent_rollout_wg = opponent_rollout_wg or actor_rollout_wg  # Use main model if no opponent specified
        self.config = config
        self.is_validation = is_validation
        self.timing_raw = {}
        
        # Initialize poker environment
        self.config_dict = {
            'num_players': config.max_players,
            'start_stack': config.initial_stack,
            'small_blind': config.small_blind,
            'big_blind': config.big_blind,
            'ante': 0
        }
        
        # Initialize game tracking
        self.game_states = {}
        self.game_rewards = {}

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            # max_obs_length=config.max_obs_length,
            # max_start_length=config.max_start_length
        ))

        
    def format_state_to_prompt(self, state: PokerGameState, player_idx: int) -> str:
        """Format poker game state into a text prompt for the LLM."""
        # Card representation
        # def format_card(card):
        #     rank, suit = card
        #     rank_map = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 
        #                7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}
        #     suit_map = {0: '♠', 1: '♥', 2: '♦', 3: '♣'}
        #     return f"{rank_map[rank]}{suit_map[suit]}"
        
        # Format hole cards
        hole_cards = [str(card) for card in state.hole_cards]
        hole_cards_str = f"Your cards: {' '.join(hole_cards)}"
        
        # Format community cards
        community_cards = [str(card) for card in state.community_cards]
        community_cards_str = "Community cards: " + (
            ' '.join(community_cards) if community_cards else "None yet"
        )
        
        # Format betting information
        round_names = ["pre-flop", "flop", "turn", "river"]
        round_str = f"Current betting round: {round_names[state.betting_round]}"
        
        pot_str = f"Current pot: {state.pot} chips"
        
        # Format player stacks and bets
        players_info = []
        for i in range(len(state.stacks)):
            position = "You" if i == player_idx else f"Player {i+1}"
            status = "active" if state.active[i] else "folded"
            bet = state.street_commits[i]
            stack = state.stacks[i]
            players_info.append(f"{position}: {status}, bet {bet}, stack {stack}")
        
        players_str = "\n".join(players_info)
        
        # Format current options
        options_str = f"To call: {state.call} chips"
        if state.min_raise > 0:
            options_str += f"\nMinimum raise: {state.min_raise} chips"
            options_str += f"\nMaximum raise: {state.max_raise} chips"
        
        # Create the final prompt
        user_content = f"""You are playing Texas Hold'em Poker. Make a decision based on the current game state.

                    {hole_cards_str}
                    {community_cards_str}
                    {round_str}
                    {pot_str}

                    Player Information:
                    {players_str}

                    Your Options:
                    {options_str}

                    Choose one of these actions:
                    1. Fold - Give up your hand and forfeit any bets
                    2. Check - Pass the action to the next player (only if no one has bet)
                    3. Call - Match the current bet ({state.call} chips)
                    4. Bet X - Bet a specific amount (minimum {state.min_raise} chips, maximum {state.max_raise} chips)
                    5. Raise X - Increase the bet (minimum {state.min_raise} chips, maximum {state.max_raise} chips)

                    Provide your decision in this format: <answer>your decision</answer>
                    Example: <answer>raise 50</answer> or <answer>fold</answer>
                    """
        # Return a chat message with the user prompt
        # return [
        #     {"role": "user", "content": user_content}
        # ]
        return user_content


    def _normalize_poker_decision(self, decision: str) -> str:
        """
        Normalize poker decisions for comparison.
        
        For example, "raise 3bb" and "raise 3 bb" should be considered the same.
        """
        # Remove extra whitespace and standardize common terms
        decision = decision.strip()
        decision = decision.replace(" bb", "bb")
        decision = decision.replace(" big blinds", "bb")
        
        # Handle common action formats
        if "fold" in decision:
            return "fold"
        elif "check" in decision and "raise" not in decision:
            return "check"
        elif "call" in decision:
            return "call"
        
        # Handle raises with sizing
        if "raise" in decision or "bet" in decision:
            # Extract the action and size if present
            sizing_match = re.search(r'(raise|bet)\s*([\d.]+)\s*(?:bb|big blinds?)?', decision)
            if sizing_match:
                action = sizing_match.group(1)
                size = sizing_match.group(2)
                return f"{action} {size}"
        
        # Return as-is if no normalization rules match
        return decision

    def _parse_poker_decision(self, normalized_decision: str) -> Tuple[str, Optional[float]]:
        """
        Parse a normalized poker decision `into action type and amount.
        `
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
                amount = float(parts[1].replace("bb", "").replace("$", ""))
            except ValueError:
                pass
        return action_type, amount

    def parse_poker_decision_old(self, response_text: str) -> Tuple[str, Optional[int]]:
        """
        Parse LLM response into a valid poker action.
        Uses the normalization and parsing logic from pokerbench.
        
        Args:
            response_text: Raw text response from the LLM
            
        Returns:
            Tuple of (action_type, amount) where amount is None for check/fold/call
            or an integer for bet/raise
        """
        # Extract answer section if using tags
        answer_pattern = r"<answer>\s*([\s\S]*?)\s*</answer>"
        answer_match = re.search(answer_pattern, response_text, re.DOTALL)
        if answer_match:
            decision_text = answer_match.group(1).strip().lower()
        else:
            decision_text = response_text.strip().lower()
        
        # Normalize the decision text
        normalized_decision = self._normalize_poker_decision(decision_text)
        
        # Parse the normalized decision
        action_type, amount = self._parse_poker_decision(normalized_decision)
        
        
        # Convert amount to integer for clubs environment if present
        if amount is not None:
            # Convert to big blinds if needed
            if 'bb' in normalized_decision:
                # Amount is in big blinds, convert to chips
                amount = int(float(amount) * self.config.big_blind)
            else:
                # Amount is in chips
                amount = int(float(amount))
        
        # Default values for clubs environment
        if action_type == "fold":
            return action_type, 0
        elif action_type in ["check", "call"]:
            return action_type, 0  # Amount will be determined by the environment
        elif action_type in ["bet", "raise"]:
            if amount is None:
                # Default to minimum raise if no amount
                return action_type, 0  # Will be adjusted to min raise
            return action_type, amount
        
        # Default to fold if can't parse
        return "fold", 0

    def parse_poker_action(self, response_text: str, obs: Dict) -> Tuple[str, int]:
        """
        Parse a poker action from the LLM's response text into a format that clubs.poker.Dealer can use.
        
        Args:
            response_text: Raw text response from the LLM
            obs: Current observation containing call/min_raise/max_raise info
        
        Returns:
            An integer value representing the action:
            - 0: fold
            - call_amount: call
            - bet_amount: bet/raise (between min_raise and max_raise)
        """

        # Extract answer section if using tags
        answer_pattern = r"<answer>\s*([\s\S]*?)\s*</answer>"
        answer_match = re.search(answer_pattern, response_text, re.DOTALL)
        if answer_match:
            decision_text = answer_match.group(1).strip().lower()
        else:
            decision_text = response_text.strip().lower()
        
        # Normalize the decision text
        normalized_decision = self._normalize_poker_decision(decision_text)
        print(f"normalized_decision: {normalized_decision}")
        # Parse the normalized decision
        action_type, amount = self._parse_poker_decision(normalized_decision)
        print(f"action_type: {action_type}")
        print(f"amount: {amount}")
        
        call_amount = obs["call"]
        min_raise = obs["min_raise"]
        max_raise = obs["max_raise"]
        current_player_stack = obs["stacks"][obs["action"]]

        # Handle all action types with rule validation
        if action_type == "fold":
            # Can always fold
            return normalized_decision, 0
        
        elif action_type == "check":
            # Can only check if there's no bet to call
            if call_amount == 0:
                return normalized_decision, 0  # In clubs, check is represented by 0
            else:
                # If player tries to check when they need to call, default to fold
                print(f"Warning: Player tried to check when call required. Defaulting to fold.")
                return normalized_decision, 0  # Fold

        elif action_type == "call":
            # If there's nothing to call, treat as check
            if call_amount == 0:
                return normalized_decision, 0  # Check
            else:
                # Ensure player has enough chips to call
                '''
                TODO: add all-in check in RL
                '''
                if call_amount >= current_player_stack:
                    # All-in call
                    return normalized_decision, current_player_stack
                else:
                    return normalized_decision, call_amount

        elif action_type in ["bet", "raise"]:
            # Can only bet/raise if min_raise is defined and positive
            if min_raise <= 0 or min_raise > current_player_stack:
                print(f"Warning: Raise not allowed. Defaulting to call.")
                # Default to call if raise not allowed
                return normalized_decision, min(call_amount, current_player_stack)
            
            # Ensure amount is provided
            if amount is None or amount <= 0:
                # Default to minimum raise
                '''
                TODO: add amount check in RL
                '''
                amount = min_raise
            
            # Enforce bounds on raise amount
            if amount < min_raise:
                print(f"Warning: Raise amount {amount} below minimum {min_raise}. Using minimum.")
                amount = min_raise
            
            if amount > max_raise:
                print(f"Warning: Raise amount {amount} above maximum {max_raise}. Using maximum.")
                amount = max_raise
            
            # Ensure amount doesn't exceed player's stack
            if amount > current_player_stack:
                print(f"Warning: Raise amount {amount} exceeds stack {current_player_stack}. Going all-in.")
                amount = current_player_stack
            
            return normalized_decision, amount

        # Default to fold if we couldn't parse the action or it was invalid
        print(f"Warning: Could not parse valid action from '{response_text}'. Defaulting to fold.")
        return normalized_decision, 0

        # # Convert amount to integer for clubs environment if present
        # if amount is not None:
        #     # Amount is in chips
        #     amount = int(float(amount))
        # # Default values for clubs environment
        # if action_type == "fold":
        #     return action_type, 0
        # elif action_type in ["check", "call"]:
        #     if obs and obs["call"] > 0:
        #         return action_type, obs["call"]
        #     else:
        #         return action_type, 0
        # elif action_type in ["bet", "raise"]:
        #     return action_type, amount
        # return "fold", 0


    def normalize_poker_action(self, action_type: str, amount: int, state: PokerGameState) -> Tuple[str, int]:
        """Normalize poker action to be valid within the current game state."""
        if action_type == "fold":
            return "fold", 0
            
        if action_type == "check":
            # Convert check to call if there's something to call
            if state.call > 0:
                return "call", state.call
            return "check", 0
            
        if action_type == "call":
            return "call", state.call
            
        if action_type == "raise":
            # Ensure raise amount is within valid range
            if amount < state.min_raise:
                amount = state.min_raise
            if amount > state.max_raise:
                amount = state.max_raise
            return "raise", amount
            
        # Default to fold if unknown action
        return "fold", 0

    def _generate_random_action(self, state: PokerGameState) -> Tuple[str, int]:
        """Generate a random action for the opponent when no opponent model is available."""
        # Simple random opponent
        if random.random() < 0.3:  # 30% fold
            return "fold", 0
        elif random.random() < 0.6:  # 30% call/check
            if state.call == 0:
                return "check", 0
            else:
                return "call", 0
        else:  # 40% raise/bet
            if state.min_raise <= state.max_raise:
                amount = random.randint(state.min_raise, state.max_raise)
                if state.call == 0:
                    return "bet", amount
                else:
                    return "raise", amount
            else:
                # Can't raise, so call/check
                if state.call == 0:
                    return "check", 0
                else:
                    return "call", 0  

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def _generate_with_gpu_padding(self, active_batch: DataProto, if_main_player: bool) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        if if_main_player:
            num_gpus = self.config.num_gpus
            if num_gpus <= 1:
                return self.actor_rollout_wg.generate_sequences(active_batch)
                
            batch_size = active_batch.batch['input_ids'].shape[0]
            remainder = batch_size % num_gpus
            
            if remainder == 0:
                return self.actor_rollout_wg.generate_sequences(active_batch)
            
            # Add padding sequences
            padding_size = num_gpus - remainder
            padded_batch = {}
            
            for k, v in active_batch.batch.items():
                # Use first sequence as padding template
                pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
                padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

            padded_active_batch = DataProto.from_dict(padded_batch)

            # Generate with padded batch
            padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
            
            # Remove padding from output
            trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
            
            # Handle meta_info if present
            if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
                trimmed_meta = {}
                for k, v in padded_output.meta_info.items():
                    if isinstance(v, torch.Tensor):
                        trimmed_meta[k] = v[:-padding_size]
                    else:
                        trimmed_meta[k] = v
                padded_output.meta_info = trimmed_meta
                
            padded_output.batch = trimmed_batch
            return padded_output
        else:
            num_gpus = self.config.num_gpus
            if num_gpus <= 1:
                return self.opponent_rollout_wg.generate_sequences(active_batch)
                
            batch_size = active_batch.batch['input_ids'].shape[0]
            remainder = batch_size % num_gpus
            
            if remainder == 0:
                return self.opponent_rollout_wg.generate_sequences(active_batch)
            
            # Add padding sequences
            padding_size = num_gpus - remainder
            padded_batch = {}
            
            for k, v in active_batch.batch.items():
                # Use first sequence as padding template
                pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
                padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

            padded_active_batch = DataProto.from_dict(padded_batch)

            # Generate with padded batch
            padded_output = self.opponent_rollout_wg.generate_sequences(padded_active_batch)
            
            # Remove padding from output
            trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
            
            # Handle meta_info if present
            if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
                trimmed_meta = {}
                for k, v in padded_output.meta_info.items():
                    if isinstance(v, torch.Tensor):
                        trimmed_meta[k] = v[:-padding_size]
                    else:
                        trimmed_meta[k] = v
                padded_output.meta_info = trimmed_meta
                
            padded_output.batch = trimmed_batch
            return padded_output

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )



        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_user_prompt(self, user_prompt: str, game_idx: int, batch_size: int) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        user_prompt_strs = []

        for i in range(batch_size):
            if i == game_idx:
                user_prompt_strs.append(user_prompt)
            else:
                user_prompt_strs.append('')

        user_prompt_ids = self._batch_tokenize(user_prompt_strs)
        return user_prompt_ids, user_prompt_strs

    def compute_format_reward(self, response_text: str) -> float:
        return poker_format_reward(response_text)
    
    def compute_tag_count_reward(self, response_text: str) -> float:
        return poker_tag_count_reward(response_text)

    def run_llm_loop(self, gen_batch, initial_input_ids):
        """
        Run a batch of self-play poker games simultaneously using LLMs for decisions.
        
        Args:
            gen_batch: Initial batch containing prompts and tokenized inputs
            initial_input_ids: Initial tokens for the generation process
            
        Returns:
            DataProto object containing all trajectories with rewards
        """
        # Initialize rollout sequences
        batch_size = initial_input_ids.shape[0]
        device = initial_input_ids.device
        last_main_player_responses = [None] * batch_size
        reward_tensor = torch.zeros(batch_size, dtype=torch.float)
        
        # Setup for batch processing (similar to Search-R1)
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_prompt_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []], 
            'responses_with_info_mask': initial_input_ids[:, []]
        }
        
        # Initialize tracking variables
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        active_num_list = [active_mask.sum().item()]
        game_steps = torch.zeros(batch_size, dtype=torch.int, device=device)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int, device=device)
        total_action_stats = torch.zeros(batch_size, dtype=torch.int, device=device)
        
        # Setup rewards and metrics
        earnings = [0.0] * batch_size
        completed = [False] * batch_size
        
        # Extra data for validation reporting
        sample_states = [[] for _ in range(batch_size)]
        sample_decisions = [[] for _ in range(batch_size)]
        
        # Setup poker environment
        poker_config = {
            "num_players": self.config.max_players,
            "num_streets": 4,
            "blinds": [self.config.small_blind, self.config.big_blind] + [0] * (self.config.max_players - 2),
            "antes": 0,
            "raise_sizes": "inf",
            "num_raises": "inf",
            "num_suits": 4,
            "num_ranks": 13,
            "num_hole_cards": 2,
            "num_community_cards": [0, 3, 1, 1],
            "num_cards_for_hand": 5,
            "mandatory_num_hole_cards": 0,
            "start_stack": self.config.initial_stack,
            "low_end_straight": True,
        }
        
        # Initialize dealers and observations
        dealers = []
        observations = []
        chat_histories = [{} for _ in range(batch_size)]
        
        for i in range(batch_size):
            dealer = clubs.poker.Dealer(**poker_config)
            obs = dealer.reset()
            dealers.append(dealer)
            observations.append(obs)
        
        game_trajectories = [
            {
                'input_ids': None, 
                'attention_mask': None, 
                'responses': [], 
                'responses_with_info_mask': [],  # Keep track of both for _compose_final_output
                'actions': [],
                'rewards': 0,
                'player_indices': []
            } 
            for _ in range(batch_size)
        ]
        
        first_main_player_action = [True] * batch_size

        turns_remaining = True
        while turns_remaining:
            # Batches for main player and opponents
            main_player_prompts = []
            main_player_input_ids = []
            main_player_attention_masks = []
            main_player_position_ids = []
            main_player_raw_prompt_ids = []
            main_player_raw_prompt = []
            main_player_game_indices = []
            main_player_system_prompts = []
            main_player_user_prompts = []
            main_player_sys_attn_mask = []
            main_player_user_attn_mask = []
            main_player_sys_position_ids_list = []
            main_player_user_position_ids_list = []
            
            opponent_prompts = []
            opponent_game_indices = []
            opponent_input_ids = []
            opponent_attention_masks = []
            opponent_position_ids = []
            opponent_raw_prompt_ids = []
            opponent_raw_prompt = []

            # Check each game to see which players need to make decisions
            for game_idx in range(batch_size):
                if not active_mask[game_idx]:
                    continue  # Skip inactive games
            
                obs = observations[game_idx]
                player_idx = obs['action']
                is_main_player = (player_idx == self.config.main_player_idx)
                
                # Convert observation to state and format as user message
                state = self._convert_obs_to_state(obs)
                user_content = self.format_state_to_prompt(state, player_idx)

                # For each player, maintain separate conversation
                if player_idx not in chat_histories[game_idx]:
                    # Initialize conversation for this player with system prompt
                    chat_histories[game_idx][player_idx] = [
                        {"role": "system", "content": self.config.system_prompt}
                    ]

                # Get player's chat history and add the user message
                player_chat = chat_histories[game_idx][player_idx]
                user_message = {"role": "user", "content": user_content}
                system_message = {"role": "system", "content": self.config.system_prompt}
                current_chat = [system_message] + [user_message]
                chat_histories[game_idx][player_idx].append(user_message)
                
                # Apply the chat template to get formatted prompt
                prompt_with_template = self.tokenizer.apply_chat_template(
                    current_chat,
                    add_generation_prompt=True,
                    tokenize=False
                )

                # Create separate system prompt and user prompt for main player
                if is_main_player:
                    main_player_system_prompt = self.tokenizer.apply_chat_template(
                        [{"role": "system", "content": self.config.system_prompt}],
                        add_generation_prompt=False,
                        tokenize=False
                    )
                    
                    main_player_user_prompt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_content}],
                        add_generation_prompt=True,
                        tokenize=False
                    )

                # Tokenize the prompt
                input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation='error')
                position_ids = compute_position_id_with_mask(attention_mask)
                raw_prompt_ids = self.tokenizer.encode(prompt_with_template, add_special_tokens=False)
                # Tokenize system and user prompts separately for main player
                if is_main_player:
                    main_player_system_ids, main_player_system_mask = verl_F.tokenize_and_postprocess_data(
                        prompt=main_player_system_prompt,
                        tokenizer=self.tokenizer,
                        max_length=self.config.max_prompt_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        left_pad=True,
                        truncation='error'
                    )
                    main_player_system_position_ids = compute_position_id_with_mask(main_player_system_mask)

                    main_player_user_ids, main_player_user_mask = verl_F.tokenize_and_postprocess_data(
                        prompt=main_player_user_prompt,
                        tokenizer=self.tokenizer,
                        max_length=self.config.max_prompt_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        left_pad=True,
                        truncation='error'
                    )
                    main_player_user_position_ids = compute_position_id_with_mask(main_player_user_mask)
                    
                # Add to main player batches
                if is_main_player:

                    main_player_prompts.append(prompt_with_template)
                    main_player_input_ids.append(input_ids[0])
                    main_player_attention_masks.append(attention_mask[0])
                    main_player_position_ids.append(position_ids[0])
                    main_player_raw_prompt_ids.append(raw_prompt_ids)
                    main_player_raw_prompt.append(current_chat)
                    main_player_game_indices.append(game_idx)
                    main_player_system_prompts.append(main_player_system_ids[0])
                    main_player_user_prompts.append(main_player_user_ids[0])
                    main_player_sys_attn_mask.append(main_player_system_mask[0])
                    main_player_user_attn_mask.append(main_player_user_mask[0])
                    main_player_sys_position_ids_list.append(main_player_system_position_ids[0])
                    main_player_user_position_ids_list.append(main_player_user_position_ids[0])

                    # Sample for validation
                    if self.is_validation and len(sample_states[game_idx]) < 1:
                        sample_states[game_idx].append(user_content)
                    
                else:
                    opponent_prompts.append(prompt_with_template)
                    opponent_input_ids.append(input_ids[0])
                    opponent_attention_masks.append(attention_mask[0])
                    opponent_position_ids.append(position_ids[0])
                    opponent_raw_prompt_ids.append(raw_prompt_ids)
                    opponent_raw_prompt.append(current_chat)
                    opponent_game_indices.append(game_idx)
                
            if main_player_input_ids:
                batch_for_gen = DataProto()
                batch_for_gen.batch = TensorDict({
                    "input_ids": torch.stack(main_player_input_ids),
                    "attention_mask": torch.stack(main_player_attention_masks),
                    "position_ids": torch.stack(main_player_position_ids)
                }, batch_size=[len(main_player_prompts)])
                # batch_for_gen.non_tensor_batch = {"raw_prompt": main_player_prompts}
                batch_for_gen.meta_info = {"do_sample": True}
                
                gen_output = self._generate_with_gpu_padding(batch_for_gen, if_main_player=True)
                response_ids, response_texts = self._postprocess_responses(gen_output.batch['responses'])

                # Extract info mask (for search results if present)
                response_with_info_mask = gen_output.batch.get('responses_with_info_mask', response_ids)
                
                # Process each main player response
                for i, (game_idx, player_idx, response_id, response_text, response_info_mask) in enumerate(
                    zip(main_player_game_indices, 
                        [self.config.main_player_idx] * len(main_player_game_indices),
                        response_ids, 
                        response_texts,
                        response_with_info_mask)
                ):

                    # Compute format reward
                    format_reward = self.compute_format_reward(response_text)
                    # Compute tag count reward
                    tag_count_reward = self.compute_tag_count_reward(response_text)

                    # Update chat history
                    chat_histories[game_idx][player_idx].append({"role": "assistant", "content": response_text})

                    # Extract answer 
                    # Extract answer section if using tags
                    answer_pattern = r"<answer>\s*([\s\S]*?)\s*</answer>"
                    answer_match = re.search(answer_pattern, response_text, re.DOTALL)
                    if answer_match:
                        answer_text = answer_match.group(1).strip().lower()
                    else:
                        answer_text = response_text.strip().lower()
                    
                    # # Normalize the decision text
                    # normalized_decision = self._normalize_poker_decision(answer_text)
                    # print(f"normalized_decision: {normalized_decision}")
                    # # Parse the normalized decision
                    # action_type, amount = self._parse_poker_decision(normalized_decision)

                    ## Tokenize just the answer part
                    answer_ids, answer_mask = verl_F.tokenize_and_postprocess_data(
                        prompt=answer_text,
                        tokenizer=self.tokenizer,
                        max_length=self.config.max_prompt_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        left_pad=False,
                        truncation='error'
                    )
                    answer_position_ids = compute_position_id_with_mask(answer_mask)

                    print(f"main_player_system_prompts[i]: {main_player_system_prompts[i].shape}")
                    print(f"main_player_user_prompts[i]: {main_player_user_prompts[i].shape}")
                    print(f"answer_ids: {answer_ids.shape}")
                    print(f"main_player_sys_attn_mask[i]: {main_player_sys_attn_mask[i].shape}")
                    print(f"main_player_user_attn_mask[i]: {main_player_user_attn_mask[i].shape}")
                    print(f"answer_mask: {answer_mask.shape}")
                    print(f"main_player_sys_position_ids_list[i]: {main_player_sys_position_ids_list[i].shape}")
                    print(f"main_player_user_position_ids_list[i]: {main_player_user_position_ids_list[i].shape}")
                    print(f"answer_position_ids: {answer_position_ids.shape}")
                    
                    # Handle the concatenated trajectory building
                    if game_trajectories[game_idx]['input_ids'] is None:
                        # # This is the first prompt-response pair
                        # game_trajectories[game_idx]['input_ids'] = torch.cat([
                        #     main_player_input_ids[i].unsqueeze(0), response_id.unsqueeze(0)
                        # ], dim=1)
                        # game_trajectories[game_idx]['attention_mask'] = torch.cat([
                        #     main_player_attention_masks[i].unsqueeze(0), 
                        #     torch.ones_like(response_id).unsqueeze(0)
                        # ], dim=1)

                        # First action: Initialize with both system and user prompt
                        game_trajectories[game_idx]['input_ids'] = torch.cat([
                            main_player_system_prompts[i].unsqueeze(0),
                            main_player_user_prompts[i].unsqueeze(0),
                            answer_ids
                        ], dim=1)
                        
                        game_trajectories[game_idx]['attention_mask'] = torch.cat([
                            main_player_sys_attn_mask[i].unsqueeze(0),
                            main_player_user_attn_mask[i].unsqueeze(0),
                            answer_mask
                        ], dim=1)
                        game_trajectories[game_idx]['position_ids'] = torch.cat([
                            main_player_sys_position_ids_list[i].unsqueeze(0),
                            main_player_user_position_ids_list[i].unsqueeze(0),
                            answer_position_ids
                        ], dim=1)
                        
                        first_main_player_action[game_idx] = False
                    else:
                        # Append to existing trajectory
                        # We need only the new prompt part (not including previous history)
                        # This is complex and depends on tokenizer - for now just concatenate full inputs
                        # In a real implementation, you'd need to extract just the new user message part
                        print(f"game_trajectories[game_idx]['input_ids']: {game_trajectories[game_idx]['input_ids'].shape}")
                        print(f"main_player_input_ids[i]: {main_player_input_ids[i].shape}")
                        print(f"response_id: {response_id.shape}")
                        # game_trajectories[game_idx]['input_ids'] = torch.cat([
                        #     game_trajectories[game_idx]['input_ids'],
                        #     main_player_input_ids[i].unsqueeze(0),
                        #     response_id.unsqueeze(0)
                        # ], dim=1)
                        # game_trajectories[game_idx]['attention_mask'] = torch.cat([
                        #     game_trajectories[game_idx]['attention_mask'],
                        #     main_player_attention_masks[i].unsqueeze(0),
                        #     torch.ones_like(response_id).unsqueeze(0)
                        # ], dim=1)
                        game_trajectories[game_idx]['input_ids'] = torch.cat([
                            game_trajectories[game_idx]['input_ids'],
                            main_player_user_prompts[i].unsqueeze(0),
                            answer_ids
                        ], dim=1)
                        
                        game_trajectories[game_idx]['attention_mask'] = torch.cat([
                            game_trajectories[game_idx]['attention_mask'],
                            main_player_user_attn_mask[i].unsqueeze(0),
                            answer_mask
                        ], dim=1)
                        game_trajectories[game_idx]['position_ids'] = torch.cat([
                            game_trajectories[game_idx]['position_ids'],
                            main_player_user_position_ids_list[i].unsqueeze(0),
                            answer_position_ids
                        ], dim=1)

                    last_main_player_responses[game_idx] = {'response_ids': answer_ids, 'response_text': answer_text, 'format_reward': format_reward, 'tag_count_reward': tag_count_reward, 'response_len': answer_ids.shape[0]}
                    print(f"last_main_player_responses[game_idx]: {last_main_player_responses[game_idx]}")
                    # Record the main player trajectory
                    game_trajectories[game_idx]['responses'].append(answer_ids)
                    # game_trajectories[game_idx]['responses_with_info_mask'].append(response_info_mask)
                    game_trajectories[game_idx]['actions'].append(response_text)
                    game_trajectories[game_idx]['rewards'] += self.config.format_reward_weight * format_reward + self.config.tag_count_reward_weight * tag_count_reward  # Format reward
                    game_trajectories[game_idx]['player_indices'].append(player_idx)

                    # Sample for validation
                    if self.is_validation and len(sample_decisions[game_idx]) < 1:
                        sample_decisions[game_idx].append(response_text)

                    # Parse action and execute in environment
                    try:
                        print(f"response_text: {response_text}")
                        print(f"observations[game_idx]: {observations[game_idx]}")
                        poker_action_raw, poker_action = self.parse_poker_action(response_text, observations[game_idx])
                        print(f"poker_action_raw: {poker_action_raw}")
                        print(f"poker_action: {poker_action}")
                        
                        valid_action_stats[game_idx] += 1
                        total_action_stats[game_idx] += 1

                        # Step the environment
                        next_obs, rewards_list, done = dealers[game_idx].step(poker_action)
                        print(f"game_idx: {game_idx}, main_player_idx: {self.config.main_player_idx}")
                        print(f"dealers[game_idx]: {dealers[game_idx]}")
                        print(f"poker_action: {poker_action}")
                        print(f"next_obs: {next_obs}")
                        print(f"rewards_list: {rewards_list}")
                        print(f"done: {done}")
                        observations[game_idx] = next_obs
                        
                        # Check if game is complete
                        if all(done):
                            active_mask[game_idx] = False
                            main_player_reward = rewards_list[self.config.main_player_idx]
                            
                            # Record for validation
                            if self.is_validation:
                                earnings[game_idx] = main_player_reward
                                completed[game_idx] = True

                            if last_main_player_responses[game_idx] is not None:
                                print(f"Game {game_idx} ended on opponent turn - including last main player response")
                                
                                # # If this game hasn't recorded any main player responses yet, add the last one
                                # if not game_trajectories[game_idx]['responses']:
                                #     # Add the last main player response to the trajectory
                                #     last_resp = last_main_player_responses[game_idx]
                                #     game_trajectories[game_idx]['responses'].append(last_resp['response_id'])
                                #     game_trajectories[game_idx]['responses_with_info_mask'].append(last_resp['response_with_info_mask'])
                                #     # game_trajectories[game_idx]['actions'].append(last_resp['action_str'])
                                #     # game_trajectories[game_idx]['rewards'] += last_resp['format_reward']
                                #     # game_trajectories[game_idx]['player_indices'].append(last_resp['player_idx'])
                                    
                            
                            # Update all rewards for this game's main player actions
                            reward_bonus = main_player_reward 
                            game_trajectories[game_idx]['rewards'] += reward_bonus
                    except Exception as e:
                        print(f"Error in game {game_idx}, action execution: {e}")
                        print(f"Defaulting to fold for player {player_idx}")
                        
                        # Default to fold (action = 0)
                        next_obs, rewards_list, done = dealers[game_idx].step(0)
                        print(f"game_idx: {game_idx}, main_player_idx: {self.config.main_player_idx}")
                        print(f"dealers[game_idx]: {dealers[game_idx]}")
                        print(f"next_obs: {next_obs}")
                        print(f"rewards_list: {rewards_list}")
                        print(f"done: {done}")
                        observations[game_idx] = next_obs
                        
                        # Check if game is complete after folding
                        if all(done):
                            active_mask[game_idx] = False
                            main_player_reward = rewards_list[self.config.main_player_idx]
                            
                            # Record for validation
                            if self.is_validation:
                                earnings[game_idx] = main_player_reward
                                completed[game_idx] = True  
                            
                            # Update rewards with game outcome
                            reward_bonus = main_player_reward * self.config.win_reward_weight
                            game_trajectories[game_idx]['rewards'] += reward_bonus
            # Process opponent batch if any
            if opponent_prompts:
                batch_for_gen = DataProto()
                batch_for_gen.batch = TensorDict({
                    "input_ids": torch.stack(opponent_input_ids),
                    "attention_mask": torch.stack(opponent_attention_masks),
                    "position_ids": torch.stack(opponent_position_ids)
                }, batch_size=[len(opponent_prompts)])
                # batch_for_gen.non_tensor_batch = {"raw_prompt_ids": np.array(opponent_raw_prompt_ids)}
                batch_for_gen.meta_info = {"do_sample": True}
                
                gen_output = self._generate_with_gpu_padding(batch_for_gen, if_main_player=False)
                response_ids, response_texts = self._postprocess_responses(gen_output.batch['responses'])

                # Extract info mask (for search results if present)
                response_with_info_mask = gen_output.batch.get('responses_with_info_mask', response_ids)
                
                # Process each main player response
                for i, (game_idx, player_idx, response_id, response_text, response_info_mask) in enumerate(
                    zip(opponent_game_indices, 
                        [self.config.main_player_idx] * len(opponent_game_indices),
                        response_ids, 
                        response_texts,
                        response_with_info_mask)
                ):

                    # Sample for validation
                    if self.is_validation and len(sample_decisions[game_idx]) < 1:
                        sample_decisions[game_idx].append(response_text)

                    # Parse action and execute in environment
                    try:
                        print(f"response_text: {response_text}")
                        print(f"observations[game_idx]: {observations[game_idx]}")
                        poker_action_raw, poker_action = self.parse_poker_action(response_text, observations[game_idx])
                        # Step the environment
                        next_obs, rewards_list, done = dealers[game_idx].step(poker_action)
                        print(f"game_idx: {game_idx}, main_player_idx: {self.config.main_player_idx}")
                        print(f"dealers[game_idx]: {dealers[game_idx]}")
                        print(f"poker_action: {poker_action}")
                        print(f"next_obs: {next_obs}")
                        print(f"rewards_list: {rewards_list}")
                        print(f"done: {done}")
                        observations[game_idx] = next_obs
                        
                        # Check if game is complete
                        if all(done):
                            active_mask[game_idx] = False
                            main_player_reward = rewards_list[self.config.main_player_idx]

                            # Record for validation
                            if self.is_validation:
                                earnings[game_idx] = main_player_reward
                                completed[game_idx] = True
                            
                            # If this was an opponent's action that ended the game,
                            # update the last main player response with final rewards
                            if last_main_player_responses[game_idx] is not None:
                                if (game_trajectories[game_idx]['rewards']) > 0:
                                    # Add win/loss reward to the main player's last action
                                    reward_bonus = main_player_reward
                                    # Update the last reward with the game outcome
                                    game_trajectories[game_idx]['rewards'] += reward_bonus
                                else:
                                    # If no main player responses recorded yet, add the last main player response
                                    print(f"Game {game_idx}: No main player responses yet, adding the last one")
                                    last_resp = last_main_player_responses[game_idx]
                                    
                                    # Record the main player's last response
                                    game_trajectories[game_idx]['responses'].append(last_resp['response_ids'])
                                    game_trajectories[game_idx]['actions'].append(last_resp['response_text'])
                                    
                                    # Compute total reward (format + tag_count + win/loss)
                                    total_reward = (
                                        self.config.format_reward_weight * last_resp['format_reward'] +
                                        self.config.tag_count_reward_weight * last_resp['tag_count_reward'] +
                                        main_player_reward
                                    )
                                    game_trajectories[game_idx]['rewards'].append(total_reward)
                                    game_trajectories[game_idx]['player_indices'].append(self.config.main_player_idx)
                            
                    except Exception as e:
                        print(f"Error in game {game_idx}, action execution: {e}")
                        print(f"Defaulting to fold for player {player_idx}")  
                        # Default to fold (action = 0)
                        total_action_stats[game_idx] += 1
                        next_obs, rewards_list, done = dealers[game_idx].step(0)
                        print(f"game_idx: {game_idx}, main_player_idx: {self.config.main_player_idx}")
                        print(f"dealers[game_idx]: {dealers[game_idx]}")
                        print(f"next_obs: {next_obs}")
                        print(f"rewards_list: {rewards_list}")
                        print(f"done: {done}")
                        observations[game_idx] = next_obs
                        
                        # Check if game is complete after folding
                        if all(done):
                            active_mask[game_idx] = False
                            main_player_reward = rewards_list[self.config.main_player_idx]

                            # Record for validation
                            if self.is_validation:
                                earnings[game_idx] = main_player_reward
                                completed[game_idx] = True

                            if last_main_player_responses[game_idx] is not None and (game_trajectories[game_idx]['rewards']) > 0:
                                # Add win/loss reward to the main player's last action
                                reward_bonus = main_player_reward
                                # Update the last reward with the game outcome
                                game_trajectories[game_idx]['rewards'] += reward_bonus

            # Check if all games are inactive
            turns_remaining = active_mask.any().item()
            active_num_list.append(active_mask.sum().item()) 

        # Compose the final output for each game and then combine
        all_game_data_protos = []
        
        for game_idx in range(batch_size):
            if len(game_trajectories[game_idx]['input_ids']) == 0:
                continue  # Skip games with no data
            
            # Create DataProto for this game
            game_data_proto = DataProto()
            
            # We already have the full concatenated conversation
            full_input_ids = game_trajectories[game_idx]['input_ids']
            full_attention_mask = game_trajectories[game_idx]['attention_mask']
            
            # Create info mask (may need adjustment based on your specific needs)
            # info_mask = torch.ones_like(full_attention_mask)
            # info_mask = self.tensor_fn.create_attention_mask(full_input_ids)
            
            # Create position ids
            # position_ids = compute_position_id_with_mask(full_attention_mask)
            position_ids = self.tensor_fn.create_position_ids(full_attention_mask)

            # Remove the final response from the input
            last_response_len = last_main_player_responses[game_idx]['response_len']
            full_input_ids, full_attention_mask, position_ids = self._remove_final_response_from_input(full_input_ids, full_attention_mask, position_ids, last_response_len)
            
            
            # Set batch
            print(f"full_input_ids: {(full_input_ids).shape}")
            print(f"full_attention_mask: {(full_attention_mask).shape}")
            # print(f"info_mask: {(info_mask).shape}")
            print(f"position_ids: {(position_ids).shape}")
            game_data_proto.batch = TensorDict({
                'input_ids': (full_input_ids),
                'attention_mask': (full_attention_mask),
                # 'info_mask': (info_mask),
                'position_ids': (position_ids),
                'responses': (last_main_player_responses[game_idx]['response_ids']),
            }, batch_size=[1])
            
            # Set non_tensor_batch
            # game_data_proto.non_tensor_batch = {
            #     'actions': game_trajectories[game_idx]['actions'],
            #     'rewards': np.array([game_trajectories[game_idx]['rewards']], dtype=object),
            #     'player_indices': game_trajectories[game_idx]['player_indices']
            # }
            
            # Set meta_info
            game_data_proto.meta_info = {
                'game_idx': game_idx
            }
            
            all_game_data_protos.append(game_data_proto)

            # # Prepare data for _compose_final_output
            # left_side = {
            #     'input_ids': torch.stack(game_trajectories[game_idx]['input_ids'])
            # }
            
            # right_side = {
            #     'responses': torch.stack(game_trajectories[game_idx]['responses']),
            #     'responses_with_info_mask': torch.stack(game_trajectories[game_idx]['responses_with_info_mask'])
            # }
            
            # meta_info = {
            #     'actions': game_trajectories[game_idx]['actions'],
            #     'rewards': game_trajectories[game_idx]['rewards'],
            #     'player_indices': game_trajectories[game_idx]['player_indices'],
            #     'game_idx': game_idx
            # }
            
            # # Compose final output for this game
            # game_data_proto = self._compose_final_output(left_side, right_side, meta_info)
            # all_game_data_protos.append(game_data_proto)
        
        # Combine all game DataProtos
        padded_all_game_data_protos = []
        if all_game_data_protos:
            max_len = max([dp.batch['input_ids'].shape[1] for dp in all_game_data_protos])
            # Pad all DataProtos to the same maximum length
            for dp in all_game_data_protos:
                current_seq_len = dp.batch['input_ids'].shape[1]
                if current_seq_len < max_len:
                    pad_size = max_len - current_seq_len
                    # Pad input_ids
                    padding = torch.full(
                        (dp.batch['input_ids'].shape[0], pad_size),
                        self.tokenizer.pad_token_id,  # Use the proper padding token ID
                        dtype=dp.batch['input_ids'].dtype,
                        device=dp.batch['input_ids'].device
                    )
                    dp.batch['input_ids'] = torch.cat([dp.batch['input_ids'], padding], dim=1)

                    # Pad attention_mask (with zeros)
                    # Pad attention_mask (with zeros)
                    # attn_padding = torch.zeros(
                    #     (dp.batch['attention_mask'].shape[0], pad_size),
                    #     dtype=dp.batch['attention_mask'].dtype,
                    # )
                    attn_padding = self.tensor_fn.create_attention_mask(padding)
                    dp.batch['attention_mask'] = torch.cat([dp.batch['attention_mask'], attn_padding], dim=1)

                    # # Pad info_mask (with zeros)
                    # info_padding = torch.zeros(
                    #     (dp.batch['info_mask'].shape[0], pad_size),
                    #     dtype=dp.batch['info_mask'].dtype,
                    # )
                    # dp.batch['info_mask'] = torch.cat([dp.batch['info_mask'], info_padding], dim=1)

                    # Pad position_ids (with zeros)
                    # pos_padding = torch.zeros(
                    #     (dp.batch['position_ids'].shape[0], pad_size),
                    #     dtype=dp.batch['position_ids'].dtype,
                    # )
                    pos_padding = self.tensor_fn.create_position_ids(attn_padding)
                    dp.batch['position_ids'] = torch.cat([dp.batch['position_ids'], pos_padding], dim=1)
                    

                padded_all_game_data_protos.append(dp)
            print(f"padded_all_game_data_protos: {len(padded_all_game_data_protos)}")
            print(f"padded_all_game_data_protos[0].batch['input_ids'].shape: {padded_all_game_data_protos[0].batch['input_ids'].shape}")
            print(f"padded_all_game_data_protos[0].batch['attention_mask'].shape: {padded_all_game_data_protos[0].batch['attention_mask'].shape}")
            # print(f"padded_all_game_data_protos[0].batch['info_mask'].shape: {padded_all_game_data_protos[0].batch['info_mask'].shape}")
            print(f"padded_all_game_data_protos[0].batch['position_ids'].shape: {padded_all_game_data_protos[0].batch['position_ids'].shape}")
            combined_output = DataProto.concat(padded_all_game_data_protos)
            # Add additional information for validation
            # combined_output.non_tensor_batch.update({
            #     'earnings': earnings,
            #     'completed': completed
            # })
            
            if self.is_validation:
                # combined_output.non_tensor_batch.update({
                #     'sample_states': [states[0] if states else "No sample available" for states in sample_states],
                #     'sample_decisions': [decisions[0] if decisions else "No decision available" for decisions in sample_decisions]
                # })
                pass

            combined_output.meta_info.update({
                'valid_action_stats': valid_action_stats.tolist(),
                'total_action_stats': total_action_stats.tolist(),
                'active_mask': active_mask.tolist(),
                'active_num_list': active_num_list
            })
            

            reward_tensor = torch.zeros_like(combined_output.batch['responses'], dtype=torch.float32)

            # Pad all DataProtos to the same maximum length
            for i, dp in enumerate(all_game_data_protos):
                prompt_length = dp.batch['input_ids'].shape[1]
                valid_response_length = dp.batch['attention_mask'][prompt_length:].sum()
                game_rewards = game_trajectories[i]['rewards']
                reward_tensor[i, valid_response_length - 1] = game_rewards
                print(f"prompt_length: {prompt_length}, valid_response_length: {valid_response_length}, game_rewards: {game_rewards}")
                print(f"reward_tensor[i, valid_response_length - 1]: {reward_tensor[i, valid_response_length - 1]}, reward_tensor.shape: {reward_tensor.shape}")

            # Save the game trajectories to a file
            return combined_output, reward_tensor
        
    def _remove_final_response_from_input(self, input_ids, attention_mask, position_ids, last_response_len):
        """Remove the final response from the input."""
        input_ids = input_ids[:, :-last_response_len]
        attention_mask = attention_mask[:, :-last_response_len]
        position_ids = position_ids[:, :-last_response_len]
        return input_ids, attention_mask, position_ids
        
    def _convert_obs_to_state(self, obs) -> PokerGameState:
        """Convert clubs observation to poker game state."""
        if obs is None:
            return None
            
        player_idx = obs['action']
        
        # Infer current betting round (street) from community cards
        if len(obs['community_cards']) == 0:
            betting_round = 0  # preflop
        elif len(obs['community_cards']) == 3:
            betting_round = 1  # flop
        elif len(obs['community_cards']) == 4:
            betting_round = 2  # turn
        elif len(obs['community_cards']) == 5:
            betting_round = 3  # river
        else:
            betting_round = 0  # default
        
        # Calculate min/max raise amounts
        call_amount = obs['call']
        min_raise = obs['min_raise']
        # max_raise = min(min_raise + obs['stacks'][player_idx], obs['stacks'][player_idx])
        max_raise = obs['max_raise']
        
        # Determine if this is the first action in the street
        is_first_action = sum(obs['street_commits']) == 0
        # is_first_action = player_idx == 0 and all(commit == 0 for commit in obs['street_commits'])
        
        state = PokerGameState(
            call=call_amount,
            min_raise=min_raise,
            max_raise=max_raise,
            hole_cards=obs['hole_cards'],
            community_cards=obs['community_cards'],
            pot=obs['pot'],
            stacks=obs['stacks'],
            active=obs['active'],
            action=player_idx,
            street_commits=obs['street_commits'],
            is_first_action=is_first_action,
            betting_round=betting_round,
            button=obs['button']
        )
        
        return state

    def _format_state_to_user_message(self, state, player_idx):
        """Format the current poker state as a user message."""
        # Convert card representations to readable format
        hole_cards_str = " ".join([str(card) for card in state.hole_cards])
        community_cards_str = " ".join([str(card) for card in state.community_cards])
        
        # Determine street name
        streets = ["Preflop", "Flop", "Turn", "River"]
        street_name = streets[state.betting_round] if state.betting_round < len(streets) else "Unknown"
        
        # Format player positions (using simple terminology)
        positions = ["Button", "Small Blind", "Big Blind"]
        if len(state.stacks) > 3:
            positions.extend([f"Position {i+1}" for i in range(3, len(state.stacks))])
        
        # Adjust positions based on the button position
        adjusted_positions = positions[-(player_idx+1):] + positions[:-(player_idx+1)]
        your_position = adjusted_positions[0]
        
        # Format the message
        message = f"""Poker Game State:
            Street: {street_name}
            Your position: {your_position}
            Your hole cards: {hole_cards_str}
            Community cards: {community_cards_str if community_cards_str else 'None'}
            Current pot: {state.pot}
            Your stack: {state.stacks[player_idx]}

            Player stacks: {state.stacks}
            Active players: {state.active}
            Street commits: {state.street_commits}

            Available actions:
            - Check/Fold: 0 chips
            """

        if state.call > 0:
            message += f"- Call: {state.call} chips\n"
        
        if state.min_raise > 0:
            message += f"- Raise: minimum {state.min_raise} chips, maximum {state.max_raise} chips\n"
        
        message += "\nWhat action do you take? Provide your decision in the format <answer>action</answer>, where action is check, call, fold, bet X, or raise X."
        
        return message
    def update_opponent_model(self, new_model_state_dict=None):
        """
        Update the opponent model with the current main player model.
        This should be called by the trainer when global_steps % update_opponent_every == 0.
        
        Args:
            new_model_state_dict: Optional state dict to use for update.
                                If None, will use the current actor model.
        """
        if new_model_state_dict is not None:
            # Update from provided state dict
            self.opponent_rollout_wg.update_model(new_model_state_dict)
        elif self.opponent_rollout_wg != self.actor_rollout_wg:
            # Get current main player model and update opponent
            main_state_dict = self.actor_rollout_wg.get_model_state_dict()
            self.opponent_rollout_wg.update_model(main_state_dict)
        
        print(f"Opponent model updated with latest main player model")