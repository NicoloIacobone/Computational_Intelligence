import random
from game import Game, Move, Player
from my_players import ReinforcementPlayer, HumanPlayer, MinimaxPlayer
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
    reinforcement_player = ReinforcementPlayer()
    minimax_player = MinimaxPlayer()
    human_player = HumanPlayer()
    random_player = RandomPlayer()
    players = [reinforcement_player, minimax_player, human_player, random_player]
    player_names = ["Reinforcement", "Minimax", "Human", "Random"]

    player_1 = int(input("Select player 1 (X):\n0 - Reinforcement\n1 - Minimax\n2 - Human\n3 - Random\n"))
    player_2 = int(input("\nSelect player 2 (O):\n0 - Reinforcement\n1 - Minimax\n2 - Human\n3 - Random\n"))

    game = Game()
    winner = game.play(players[player_1], players[player_2])

    if winner == 0:
        print(f"{player_names[player_1]} player X won!")
    elif winner == 1:
        print(f"{player_names[player_2]} player O won!")
