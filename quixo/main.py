import random
from game import Game, Move, Player
from collections import defaultdict


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        print(f"Player {game.current_player_idx} moves {from_pos} {move}")
        return from_pos, move


class MyPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        print(f"Player {game.current_player_idx} moves {from_pos} {move}")
        return from_pos, move
    
class ReinforcementPlayer(Player):
    def __init__(self, player_index, random_move = 0.0):
        self.value_dictionary = defaultdict(float) # state of the game and its value
        self.hit_state = defaultdict(int) # state of the game and how many times it was visited during the training phase
        self.epsilon = 0.1 # learning rate
        self.player_index = player_index # index of the player (1 or 2)
        self.random_move = random_move # a value between 0 and 1, used to choose a random move when training

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        pass

    


if __name__ == '__main__':
    g = Game()
    g.print()
    player1 = MyPlayer()
    player2 = RandomPlayer()
    winner = g.play(player1, player2)
    g.print()
    print(f"Winner: Player {winner}")
