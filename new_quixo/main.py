import random
from game import Game, Move, Player
from my_players import ReinforcementPlayer, HumanPlayer
from utils import Utils


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class MyPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


if __name__ == '__main__':
    random_player = RandomPlayer()
    reinforcement_player = ReinforcementPlayer()
    environment = Utils()

    # environment.train(reinforcement_player, random_player, 500_000, policy_name="500k", plot=True, decreasing_exp_rate=True)
    # environment.test(reinforcement_player, random_player, policy_name="500k")
    # environment.evaluate_player(reinforcement_player, "500k_1")

    environment.train(reinforcement_player, random_player, 1000)
    reinforcement_player.policy_to_txt()