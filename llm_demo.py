import clubs
import random
from llm import LLMPlayer

if __name__ == '__main__':
    
    llm_player = LLMPlayer()

    config = clubs.configs.NO_LIMIT_HOLDEM_SIX_PLAYER
    dealer = clubs.poker.Dealer(**config)
    obs = dealer.reset()

    while True:
        # Get game state
        game_state = {
            'call': obs['call'],
            'min_raise': obs['min_raise'],
            'max_raise': obs['max_raise'],
            'hole_cards': obs['hole_cards'],
            'community_cards': obs['community_cards'],
            'pot': obs['pot'],
            'stacks': obs['stacks'],
        }

        # Get action from LLM
        bet = llm_player.decide_action(game_state)

        # 执行LLM决策
        obs, rewards, done = dealer.step(bet)
        if all(done):
            break

    print(rewards)