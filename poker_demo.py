import clubs
import random
from poker_player import LLMPlayer
import sys
import os
import ray

# os.environ["PYTHONPATH"] = "/workdir/project-search/rl4search/clubs:" + os.environ.get("PYTHONPATH", "")

# # Add the startup hook for Ray workers
# ray.init(
#     runtime_env={
#         "env_vars": {
#             "PYTHONPATH": "/workdir/project-search/rl4search/clubs:/workdir/project-search/rl4search/verl:" + os.environ.get("PYTHONPATH", ""),
#             "TOKENIZERS_PARALLELISM": "false",
#         },
#         "py_modules": [
#             "/workdir/project-search/rl4search/verl",
#             "/workdir/project-search/rl4search/clubs"
#         ]
#     }
# )

# import inspect

# clubs_path = '/workdir/project-search/rl4search/clubs'
# if clubs_path not in sys.path:
#     sys.path.insert(0, clubs_path)

# clubs_path = inspect.getfile(clubs)
# print("Clubs module path:", clubs_path)


# exit(0)

if __name__ == '__main__':
    
    # llm_player = LLMPlayer()

    config = clubs.configs.NO_LIMIT_HOLDEM_SIX_PLAYER
    dealer = clubs.poker.Dealer(**config)
    obs = dealer.reset()

    print("config", config)
    print("dealer", dealer)
    # Add the startup hook for clubs
    print("obs", obs)
    print("obs['call']", obs['call'])
    print("obs['min_raise']", obs['min_raise'])
    print("obs['max_raise']", obs['max_raise'])
    print("obs['hole_cards']", obs['hole_cards'])
    print("obs['community_cards']", obs['community_cards'])
    print("obs['pot']", obs['pot'])
    print("obs['stacks']", obs['stacks'])
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

        # # Get action from LLM
        # bet = llm_player.decide_action(game_state)

        rand = random.random()
        if rand < 0.1:
            bet = 0
        elif rand < 0.80:
            bet = obs['call']
        else:
            bet = random.randint(obs['min_raise'], obs['max_raise'])

        # TODO
        print("config", config)
        print("dealer", dealer)
        print("obs", obs)
        print("obs['call']", obs['call'])
        print("obs['min_raise']", obs['min_raise'])
        print("obs['max_raise']", obs['max_raise'])
        print("obs['hole_cards']", obs['hole_cards'])
        print("obs['community_cards']", obs['community_cards'])
        print("obs['pot']", obs['pot'])
        print("obs['stacks']", obs['stacks'])
        print("bet", bet)
        obs, rewards, done = dealer.step(bet)
        print("rewards", rewards, "done", done)   
        if all(done):
            break

    print(rewards)