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
    reinforcement_player.set_random_move(0.9)
    reinforcement_player.load_policy("/policies/test_policy_1")

    environment = Utils()

    environment.train(reinforcement_player, random_player, 1_000_000, 100, -100, decreasing_exp_rate=True)
    environment.test(reinforcement_player, random_player)
    # reinforcement_player.policy_to_txt()

    # for i in range(2):
    #     environment.train(reinforcement_player, random_player, 1_000_000, policy_name=("no_decr_1kk_" + str(i+1)))
    #     environment.test(reinforcement_player, random_player, policy_name=("no_decr_1kk_" + str(i+1)))
    #     environment.evaluate_player(reinforcement_player, ("no_decr_1kk_1_" + str(i+1)))
    #     reinforcement_player.policy_to_txt("no_decr_1kk_" + str(i+1))

    # reinforcement_player.set_random_move(0.5)
    # for i in range(3):
    #     environment.train(reinforcement_player, random_player, 1_000_000, policy_name=("1kk_" + str(i+1)), decreasing_exp_rate=True)
    #     environment.test(reinforcement_player, random_player, policy_name=("1kk_" + str(i+1)))
    #     environment.evaluate_player(reinforcement_player, ("1kk_1_" + str(i+1)))
    #     reinforcement_player.policy_to_txt("1kk_" + str(i+1))
    #     reinforcement_player.set_random_move(reinforcement_player.random_move * 5)
    #     reinforcement_player.set_learning_rate(reinforcement_player.learning_rate * 5)
