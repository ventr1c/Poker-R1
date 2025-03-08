import clubs
import random

config = clubs.configs.NO_LIMIT_HOLDEM_SIX_PLAYER
dealer = clubs.poker.Dealer(**config)
obs = dealer.reset()

while True:
    call = obs['call']
    min_raise = obs['min_raise']
    max_raise = obs['max_raise']

    rand = random.random()
    if rand < 0.1:
        bet = 0
    elif rand < 0.80:
        bet = call
    else:
        bet = random.randint(min_raise, max_raise)

    obs, rewards, done = dealer.step(bet)
    if all(done):
        break

print(rewards)