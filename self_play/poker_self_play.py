# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
import ray
import uuid
import clubs
import random
import hydra
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from vllm import LLM, SamplingParams
from dataclasses import dataclass, field
import re
from verl.utils.reward_score.pokerbench import _normalize_poker_decision, _parse_poker_decision

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GameState:
    """Represents the state of a poker game."""
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
    # New field to track if this is the first action in the betting round
    is_first_action: bool

@dataclass
class Trajectory:
    """Represents a single decision in a game."""
    prompt: str
    response: str
    action_bet: int
    action_type: str  # Type of action (bet, call, raise, fold, check)
    reward: float = 0.0
    game_id: str = ""
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    finished: bool = False
    is_selected: bool = False  # Was this the action actually taken
    rollout_idx: int = 0  # Index within the rollout group
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary suitable for dataframe creation."""
        return {
            "uid": self.uid,
            "prompt": self.prompt,
            "response": self.response,
            "action_type": self.action_type,
            "action_bet": self.action_bet,
            "reward": self.reward,
            "game_id": self.game_id,
            "finished": self.finished,
            "is_selected": self.is_selected,
            "rollout_idx": self.rollout_idx
        }


class LLMPokerPlayer:
    """LLM-based poker player that uses a language model to make decisions."""
    
    def __init__(self, model_path: str, temperature: float = 0.8, top_p: float = 0.95, max_tokens: int = 100):
        """
        Initialize an LLM-based poker player.
        
        Args:
            model_path: Path to the model
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            max_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        
        logger.info(f"Initializing LLM player with model: {model_path}")
        
        # Initialize LLM
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,  # Adjust based on available GPUs
            dtype="bfloat16",
            gpu_memory_utilization=0.5,
        )
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
    
    def create_prompt(self, game_state: Dict[str, Any]) -> str:
        """
        Create a prompt describing the current game state for the LLM.
        
        Args:
            game_state: Dictionary containing the current game state
            
        Returns:
            A formatted prompt string
        """
        hole_cards_str = ", ".join([str(card) for card in game_state['hole_cards']])
        community_cards_str = ", ".join([str(card) for card in game_state['community_cards']]) if game_state['community_cards'] else "None"
        
        # Format player stacks and status
        players_info = []
        for i, (stack, active) in enumerate(zip(game_state['stacks'], game_state['active'])):
            status = "active" if active else "folded"
            if i == game_state['action']:
                status += " (you)"
            players_info.append(f"Player {i+1}: {stack} chips ({status})")
        players_info_str = "\n".join(players_info)
        
        # Determine if this is the first action in the betting round
        is_first_action = True
        for commit in game_state['street_commits']:
            if commit > 0:
                is_first_action = False
                break
        
        # Available actions depend on the state
        action_instructions = ""
        if is_first_action:
            action_instructions = f"""What is your action? Choose one of the following:
1. Fold (give up and lose what you've bet)
2. Check (make no bet, only available if there's no existing bet)
3. Bet X (place a bet of X chips, must be between {game_state['min_raise']} and {game_state['max_raise']})"""
        elif game_state['call'] == 0:
            action_instructions = f"""What is your action? Choose one of the following:
1. Fold (give up and lose what you've bet)
2. Check (make no bet, since there's no existing bet to call)
3. Bet X (place a bet of X chips, must be between {game_state['min_raise']} and {game_state['max_raise']})"""
        else:
            action_instructions = f"""What is your action? Choose one of the following:
1. Fold (give up and lose what you've bet)
2. Call (match the current bet of {game_state['call']} chips)
3. Raise X (increase the bet to X chips, must be between {game_state['min_raise']} and {game_state['max_raise']})"""
        
        prompt = f"""You are playing Texas Hold'em Poker. Make the optimal decision based on the current game state.

Your hole cards: {hole_cards_str}
Community cards: {community_cards_str}
Current pot: {game_state['pot']} chips
"""

        if game_state['call'] > 0:
            prompt += f"Current bet to call: {game_state['call']} chips\n"
        else:
            prompt += "No current bet to call (you can check)\n"
            
        prompt += f"""Minimum bet/raise: {game_state['min_raise']} chips
Maximum bet/raise: {game_state['max_raise']} chips
Your stack: {game_state['stacks'][game_state['action']]} chips

Players information:
{players_info_str}

{action_instructions}

Respond with just one word/phrase: "fold", "check", "call", "bet X", or "raise X" where X is a number.
"""
        return prompt
    
    def decide_action(self, game_state: Dict[str, Any]) -> Tuple[int, str, str, str]:
        """
        Decide on an action based on the current game state.
        
        Args:
            game_state: Dictionary containing the current game state
            
        Returns:
            Tuple of (bet_amount, action_type, raw_response, prompt)
        """
        prompt = self.create_prompt(game_state)
        
        # Get response from LLM
        outputs = self.llm.generate([prompt], self.sampling_params)
        raw_response = outputs[0].outputs[0].text.strip()
        
        # Parse the response to get the bet amount and action type
        bet_amount, action_type = self._parse_action(raw_response, game_state)
        
        return bet_amount, action_type, raw_response, prompt
    
    def _parse_action(self, response: str, game_state: Dict[str, Any]) -> Tuple[int, str]:
        """
        Parse the LLM's response into a valid bet amount and action type.
        
        Args:
            response: Raw response from the LLM
            game_state: Dictionary containing the current game state
            
        Returns:
            Tuple of (bet_amount, action_type)
        """
        # Game state parameters
        call_amount = game_state['call']
        min_raise = game_state['min_raise']
        max_raise = game_state['max_raise']
        
        # Determine if this is the first action in the betting round
        is_first_action = True
        for commit in game_state['street_commits']:
            if commit > 0:
                is_first_action = False
                break
        
        # Use the normalization and parsing functions from pokerbench.py
        normalized_response = _normalize_poker_decision(response.lower().strip())
        action_type, amount = _parse_poker_decision(normalized_response)
        
        # Default action if we can't parse properly
        default_action = "check" if call_amount == 0 else "call"
        default_amount = 0 if call_amount == 0 else call_amount
        
        # Handle different action types
        if action_type == "fold":
            return 0, "fold"
            
        elif action_type == "check":
            if call_amount == 0:
                return 0, "check"
            else:
                # If check is not valid, default to call
                return call_amount, "call"
                
        elif action_type == "call":
            if call_amount > 0:
                return call_amount, "call"
            else:
                # If call is not valid (nothing to call), interpret as check
                return 0, "check"
                
        elif action_type == "bet":
            # Bet is only valid as first action in betting round when no one has bet
            if is_first_action and call_amount == 0:
                if amount is not None:
                    # Ensure bet is within limits
                    bet_amount = int(max(min_raise, min(amount, max_raise)))
                    return bet_amount, "bet"
                else:
                    # Default to min raise if amount not specified
                    return min_raise, "bet"
            else:
                # If bet is not valid context, interpret as raise
                if amount is not None:
                    raise_amount = int(max(min_raise, min(amount, max_raise)))
                    return raise_amount, "raise"
                else:
                    return min_raise, "raise"
                    
        elif action_type == "raise":
            # Raise is only valid when there's an existing bet
            if call_amount > 0:
                if amount is not None:
                    # Ensure raise is within limits
                    raise_amount = int(max(min_raise, min(amount, max_raise)))
                    return raise_amount, "raise"
                else:
                    # Default to min raise if amount not specified
                    return min_raise, "raise"
            else:
                # If raise is not valid (nothing to raise), interpret as bet
                if amount is not None:
                    bet_amount = int(max(min_raise, min(amount, max_raise)))
                    return bet_amount, "bet"
                else:
                    return min_raise, "bet"
        
        # If we couldn't parse or unknown action type, use default
        return default_amount, default_action


class RandomPokerPlayer:
    """A player that makes random decisions, used for debugging or as a baseline."""
    
    def __init__(self, fold_prob: float = 0.1, call_prob: float = 0.7, raise_prob: float = 0.2):
        self.fold_prob = fold_prob
        self.call_prob = call_prob
        self.raise_prob = raise_prob
    
    def decide_action(self, game_state: Dict[str, Any]) -> Tuple[int, str, str, Optional[str]]:
        """Make a random decision based on probabilities."""
        choice = random.random()
        
        # Determine if this is the first action
        is_first_action = True
        for commit in game_state['street_commits']:
            if commit > 0:
                is_first_action = False
                break
                
        # Fold
        if choice < self.fold_prob:
            return 0, "fold", "fold", None
        
        # Check/Call
        elif choice < self.fold_prob + self.call_prob:
            if game_state['call'] == 0:
                return 0, "check", "check", None
            else:
                return game_state['call'], "call", "call", None
        
        # Bet/Raise
        else:
            if game_state['min_raise'] <= game_state['max_raise']:
                raise_amount = random.randint(game_state['min_raise'], game_state['max_raise'])
                if is_first_action and game_state['call'] == 0:
                    return raise_amount, "bet", f"bet {raise_amount}", None
                else:
                    return raise_amount, "raise", f"raise {raise_amount}", None
            else:
                # If min_raise > max_raise, just call/check
                if game_state['call'] == 0:
                    return 0, "check", "check", None
                else:
                    return game_state['call'], "call", "call", None


class TrajectoryCollector:
    """Collects and manages the trajectories from poker games for GRPO training."""
    
    def __init__(self):
        """Initialize a new trajectory collector."""
        self.trajectories = []
        self.alternative_trajectories = []  # For storing all alternative responses
        self.current_game_id = None
        self.current_game_trajectories = []
        self.current_alternatives = []  # For storing alternatives of current decision point
    
    def start_game(self):
        """Start a new game with a unique ID."""
        self.current_game_id = str(uuid.uuid4())
        self.current_game_trajectories = []
        self.current_alternatives = []  # Reset alternatives for new game
    
    def add_step(self, prompt: str, response: str, bet: int, action_type: str):
        """Add a step in the current game's trajectory."""
        trajectory = Trajectory(
            prompt=prompt,
            response=response,
            action_bet=bet,
            action_type=action_type,
            game_id=self.current_game_id
        )
        self.current_game_trajectories.append(trajectory)
    
    def add_alternative(self, prompt: str, response: str, 
                      action_bet: int, action_type: str, is_selected: bool):
        """Add an alternative response for GRPO training."""
        trajectory = Trajectory(
            prompt=prompt,
            response=response,
            action_bet=action_bet,
            action_type=action_type,
            game_id=self.current_game_id,
            is_selected=is_selected,  # Track which response was actually used
            rollout_idx=len(self.current_alternatives)  # Track rollout index
        )
        self.current_alternatives.append(trajectory)
        
        # If this is the selected action, also add it to the main trajectory list
        if is_selected:
            self.current_game_trajectories.append(trajectory)

    def end_game(self, rewards: List[float], player_idx: int):
        """End the current game and assign rewards."""
        player_reward = rewards[player_idx]
        
        # Assign rewards to all trajectories in the current game for this player
        for traj in self.current_game_trajectories:
            traj.reward = player_reward
            traj.finished = True
        
        # Add trajectories to the full list
        self.trajectories.extend(self.current_game_trajectories)
        
        # Also assign rewards to all alternative responses
        # All alternatives from the same decision point get the same reward
        for traj in self.current_alternatives:
            traj.reward = player_reward
            traj.finished = True
        
        # Add alternatives to a separate list for GRPO training
        self.alternative_trajectories.extend(self.current_alternatives)
        self.current_alternatives = []
    
    def get_trajectories(self) -> List[Trajectory]:
        """Get all collected trajectories."""
        return self.trajectories
    
    def save_trajectories(self, output_path: str):
        """Save trajectories to a parquet file in GRPO-compatible format."""
        # Format the data for GRPO training
        grpo_data = []
        
        # Group alternatives by their prompt (decision point)
        prompt_groups = {}
        for traj in self.alternative_trajectories:
            # Create a unique key for each decision point
            prompt_key = f"{traj.game_id}_{hash(traj.prompt)}"
            if prompt_key not in prompt_groups:
                prompt_groups[prompt_key] = []
            prompt_groups[prompt_key].append(traj)
        
        # Format each group for GRPO training
        for prompt_key, alternatives in prompt_groups.items():
            # Ensure we have at least 2 alternatives for GRPO
            if len(alternatives) >= 2:
                group_id = str(uuid.uuid4())
                for i, traj in enumerate(alternatives):
                    grpo_data.append({
                        "uid": traj.uid,
                        "prompt": traj.prompt,
                        "response": traj.response,
                        "action_type": traj.action_type,
                        "action_bet": traj.action_bet,
                        "reward": traj.reward,
                        "game_id": traj.game_id,
                        "group_id": group_id,  # Group ID for GRPO
                        "rollout_idx": i,      # Index within the group
                        "was_selected": getattr(traj, 'is_selected', False)
                    })
        
        # Create and save the dataframe
        df = pd.DataFrame(grpo_data)
        df.to_parquet(output_path)

            

class PokerSelfPlayEnv:
    """Environment for poker self-play."""
    
    def __init__(self, main_model_path: str, opponent_model_path: str,
                 main_player_idx: int = 0, temperature: float = 0.8,
                 top_p: float = 0.95, use_random_opponent: bool = False,
                 num_rollouts: int = 2):
        """
        Initialize the poker self-play environment.
        
        Args:
            main_model_path: Path to the main player model
            opponent_model_path: Path to the opponent model
            main_player_idx: Index of the main player (0-5)
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            use_random_opponent: Whether to use random opponents
            num_rollouts: Number of responses to generate for GRPO (min 2)
        """
        self.main_player_idx = main_player_idx
        self.use_random_opponent = use_random_opponent
        self.num_rollouts = max(2, num_rollouts)  # Minimum 2 for GRPO
        
        # Initialize players
        if use_random_opponent:
            self.opponent_player = RandomPokerPlayer()
        else:
            self.opponent_player = LLMPokerPlayer(
                model_path=opponent_model_path,
                temperature=temperature,
                top_p=top_p
            )
        
        self.main_player = LLMPokerPlayer(
            model_path=main_model_path,
            temperature=temperature,
            top_p=top_p
        )
        
        # Initialize trajectory collector
        self.collector = TrajectoryCollector()
    
    def _handle_player_turn(self, player_idx: int, game_state: Dict[str, Any]) -> Tuple[int, str, str, str]:
        """
        Handle a player's turn, with multiple rollouts for the main player.
        
        Args:
            player_idx: Index of the current player
            game_state: Current game state
            
        Returns:
            Tuple of (bet_amount, action_type, response, prompt)
        """
        if player_idx == self.main_player_idx:
            # For main player, use multiple rollouts
            prompt = self.main_player.create_prompt(game_state)
            
            # Generate multiple responses
            outputs = self.main_player.llm.generate(
                [prompt] * self.num_rollouts, 
                self.main_player.sampling_params
            )
            
            responses = [output.outputs[0].text.strip() for output in outputs]
            
            # Parse all responses to valid actions
            parsed_actions = []
            for response in responses:
                bet_amount, action_type = self.main_player._parse_action(response, game_state)
                parsed_actions.append((bet_amount, action_type, response))
            
            # Use the first action in the actual game (could also sample)
            selected_bet, selected_action_type, selected_response = parsed_actions[0]
            
            # Record all responses for GRPO training
            for i, (bet, action_type, response) in enumerate(parsed_actions):
                # Add to trajectory collector
                self.collector.add_alternative(
                    prompt=prompt,
                    response=response,
                    action_bet=bet,
                    action_type=action_type,
                    is_selected=(i == 0)  # Mark if this was the action actually taken
                )
            
            return selected_bet, selected_action_type, selected_response, prompt
        else:
            # For opponent players, use regular single response
            return self.opponent_player.decide_action(game_state)
    
    def collect_trajectories(self, num_games: int):
        """
        Collect trajectories from self-play poker games.
        
        Args:
            num_games: Number of games to play
        """
        config = clubs.configs.KUHN_POKER_GAME
        
        # Play multiple games
        for _ in tqdm(range(num_games), desc="Playing poker games"):
            # Start a new game
            self.collector.start_game()
            
            # Initialize dealer
            dealer = clubs.poker.Dealer(**config)
            
            # Play until the game is done
            while not dealer.done:
                # Get the current player's index
                player_idx = dealer.button
                
                # Skip if player is not active
                if not dealer.active[player_idx]:
                    continue
                
                # Get game state
                game_state = {
                    'call': dealer.call,
                    'min_raise': dealer.min_raise,
                    'max_raise': dealer.max_raise,
                    'hole_cards': dealer.hole_cards[player_idx],
                    'community_cards': dealer.community_cards,
                    'pot': dealer.pot,
                    'stacks': dealer.stacks,
                    'active': dealer.active,
                    'action': player_idx,
                    'street_commits': dealer.street_commits,
                    'is_first_action': all(commit == 0 for commit in dealer.street_commits)
                }
                
                # Handle the player's turn using our new method
                bet, action_type, response, prompt = self._handle_player_turn(player_idx, game_state)
                
                # Execute the action in the game
                if action_type == "fold":
                    dealer.fold(player_idx)
                elif action_type in ["check", "call"]:
                    dealer.call(player_idx)
                else:  # bet or raise
                    dealer.raise_to(player_idx, bet)
            
            # End the game and assign rewards to all trajectories
            self.collector.end_game(dealer.payouts, self.main_player_idx)
    
    def save_trajectories(self, output_path: str):
        """Save the collected trajectories to a file."""
        self.collector.save_trajectories(output_path)


@hydra.main(config_path='verl/trainer/config', config_name='ppo_trainer', version_base=None)
def run_self_play(config):
    """Main function to run poker self-play and then train with GRPO."""
    OmegaConf.resolve(config)
    
    # Set temporary model paths for self-play (will be replaced with actual paths)
    main_model_path = config.actor_rollout_ref.model.path
    opponent_model_path = config.actor_rollout_ref.model.path
    
    # Add self-play specific configuration
    with open_dict(config):
        # Generate timestamp for unique directory
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up the default output directory if not provided
        default_output_dir = os.path.join(
            config.trainer.default_local_dir, 
            "poker_trajectories",
            timestamp
        )
        config.self_play = {
            "num_games": 100,
            "main_player_idx": 0,
            "use_random_opponent": False,
            "temperature": 0.8,
            "top_p": 0.95,
            "num_rollouts": 2,  # Number of rollouts for GRPO
            "output_dir": default_output_dir
        }
        # Make sure GRPO is enabled
        config.algorithm.adv_estimator = "grpo"



    # Set up Ray for distributed computing
    if not ray.is_initialized():
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        })
    
    # Make sure the output directory exists
    os.makedirs(config.self_play.output_dir, exist_ok=True)
    
    # Determine output paths for trajectories
    train_output_path = os.path.join(config.self_play.output_dir, "train.parquet")
    val_output_path = os.path.join(config.self_play.output_dir, "val.parquet")

    # Run self-play to collect trajectories
    ray.get(collect_trajectories_task.remote(config))
    
    # Run GRPO training with the collected trajectories

    
    # Import and run the PPO trainer
    from verl.trainer.main_ppo import main_task
    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def collect_trajectories_task(config):
    """Ray task to collect trajectories through self-play."""
    from verl.utils.fs import copy_to_local
    
    # Get model paths
    # Get model paths
    main_model_path = copy_to_local(config.actor_rollout_ref.model.path)
    opponent_model_path = main_model_path  # Initially the same
    
    # Ensure output directory exists
    os.makedirs(config.self_play.output_dir, exist_ok=True)
    
    # Define output paths
    train_output_path = os.path.join(config.self_play.output_dir, "train.parquet")
    val_output_path = os.path.join(config.self_play.output_dir, "val.parquet")
    
    
    # Set up the self-play environment
    env = PokerSelfPlayEnv(
        main_model_path=main_model_path,
        opponent_model_path=opponent_model_path,
        main_player_idx=config.self_play.main_player_idx,
        temperature=config.self_play.temperature,
        top_p=config.self_play.top_p,
        use_random_opponent=config.self_play.use_random_opponent,
        num_rollouts=config.self_play.num_rollouts
    )
    
    # Calculate the split between training and validation sets
    num_games = config.self_play.num_games
    num_train_games = int(num_games * 0.9)  # 90% for training
    num_val_games = num_games - num_train_games  # 10% for validation
    
    # Collect training trajectories
    logger.info(f"Collecting training trajectories from {num_train_games} games...")
    env.collect_trajectories(num_train_games)
    env.save_trajectories(train_output_path)
    
    # Collect validation trajectories
    logger.info(f"Collecting validation trajectories from {num_val_games} games...")
    env.collector = TrajectoryCollector()  # Reset collector
    env.collect_trajectories(num_val_games)
    env.save_trajectories(val_output_path)
    
    # Update config with the new trajectory paths
    with open_dict(config):
        config.data.train_files = train_output_path
        config.data.val_files = val_output_path
    
    logger.info(f"Trajectories collected and saved to {output_dir}")
    return config


if __name__ == "__main__":
    run_self_play()