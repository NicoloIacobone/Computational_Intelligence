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

    environment.train(reinforcement_player, random_player)
    environment.test(reinforcement_player, random_player)

    # reinforcement_player.set_random_move(0.5)
    # environment.train(reinforcement_player, random_player, 50_000, policy_name="rl_vs_rand_100k_decreasing05", plot=True, decreasing_exp_rate=True)
    # environment.evaluate_player(reinforcement_player, "rl_vs_rand_100k_decreasing05_1")
    # environment.train(reinforcement_player, random_player, 50_000, policy_name="rl_vs_rand_100k_decreasing05", plot=True, decreasing_exp_rate=True)
    # environment.evaluate_player(reinforcement_player, "rl_vs_rand_100k_decreasing05_1")
    # environment.test(reinforcement_player, random_player, policy_name="rl_vs_rand_100k_decreasing05")
    # environment.evaluate_player(reinforcement_player, "new_hash_function_1")


