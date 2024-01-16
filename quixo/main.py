import random
from game import Game, Move, Player
from my_players import ReinforcementPlayer, HumanPlayer

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        # print(f"Player {game.current_player_idx} moves {from_pos} {move}")
        return from_pos, move

if __name__ == '__main__':
    human_player = HumanPlayer()
    reinforcement_player = ReinforcementPlayer()
    game = Game()
    # game.play(reinforcement_player, human_player)
    reinforcement_player.compute_available_moves(game)
