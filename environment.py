import random
from treys import Card, Evaluator, Deck

# Constants for Texas Hold'em rules
SMALL_BLIND = 1  # Small blind amount
BIG_BLIND = 2  # Big blind amount
STARTING_CHIPS = 100  # Initial chips for each player
MIN_RAISE = BIG_BLIND  # Minimum raise amount


class LLMPlayer:
    """ LLM-based poker player (can be replaced with real LLM inference). """
    def __init__(self, name):
        self.name = name
        self.chips = STARTING_CHIPS
        self.hand = []
        self.active = True  # Whether the player is still in the game

    def make_decision(self, state):
        """ Make a decision based on the current game state. """
        legal_actions = state["legal_actions"]
        return random.choice(legal_actions)  # Using random actions (replace with LLM inference)

    def bet(self, amount):
        """ Perform a bet. """
        bet_amount = min(self.chips, amount)  # Cannot bet more than available chips
        self.chips -= bet_amount
        return bet_amount


class PokerGame:
    """ Texas Hold'em poker game manager. """
    def __init__(self, num_players=6):
        self.deck = Deck()
        self.players = [LLMPlayer(f"LLM-{i}") for i in range(num_players)]
        self.active_players = self.players[:]  # List of players still in the game
        self.board = []  # Community cards
        self.pot = 0  # Total chips in the pot
        self.current_bet = 0  # Current highest bet
        self.evaluator = Evaluator()

    def deal(self):
        """ Deal two hole cards to each player. """
        for player in self.players:
            player.hand = [self.deck.draw(1), self.deck.draw(1)]

    def post_blinds(self):
        """ Force small blind and big blind bets. """
        self.players[0].bet(SMALL_BLIND)
        self.players[1].bet(BIG_BLIND)
        self.pot += SMALL_BLIND + BIG_BLIND
        self.current_bet = BIG_BLIND  # Initial minimum bet

    def betting_round(self):
        """ Execute a full betting round. """
        highest_bet = self.current_bet
        while True:
            for player in self.active_players:
                if player.chips == 0:  # Skip players with no chips left
                    continue

                legal_actions = ["fold", "call"]
                if player.chips >= highest_bet:
                    legal_actions.append("raise")

                state = {"pot": self.pot, "current_bet": highest_bet, "legal_actions": legal_actions}
                action = player.make_decision(state)

                if action == "fold":
                    player.active = False
                    self.active_players.remove(player)
                elif action == "call":
                    bet_amount = highest_bet - (STARTING_CHIPS - player.chips)
                    self.pot += player.bet(bet_amount)
                elif action == "raise":
                    raise_amount = highest_bet + MIN_RAISE
                    self.pot += player.bet(raise_amount)
                    highest_bet = raise_amount  # Update current highest bet

            # Proceed to the next round only if all active players have matched the highest bet
            if all(player.chips == 0 or STARTING_CHIPS - player.chips == highest_bet for player in self.active_players):
                break

    def play(self):
        """ Run a complete game round. """
        self.deal()
        self.post_blinds()

        # Preflop betting round
        print("--- Preflop ---")
        self.betting_round()

        # Flop, Turn, and River rounds
        for round_name in ["Flop", "Turn", "River"]:
            if len(self.active_players) <= 1:
                break  # Only one player left, they win by default

            print(f"--- {round_name} ---")
            if round_name == "Flop":
                self.board.extend([self.deck.draw(1) for _ in range(3)])  # Append 3 separate cards
            else:
                self.board.append(self.deck.draw(1))  # Append a single card
            print(self.board)
            print(f"Community cards: {[Card.int_to_str(c) for c in self.board]}")

            self.betting_round()

        # Showdown: Determine the winner
        # self.determine_winner()

    def determine_winner(self):
        """ Use `treys` to evaluate the best hand and determine the winner. """
        best_score = float("inf")
        winner = None

        for player in self.active_players:
            hand = player.hand
            score = self.evaluator.evaluate(hand, self.board)

            if score < best_score:
                best_score = score
                winner = player

        print(f"Winner: {winner.name} (Pot: {self.pot} chips)")
        winner.chips += self.pot  # Winner takes the pot


# Start the game
game = PokerGame()
game.play()