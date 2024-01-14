import random
from game import Game, Move, Player
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import pickle
import numpy as np
import os


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        # print(f"Player {game.current_player_idx} moves {from_pos} {move}")
        return from_pos, move


class MyPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        # print(f"Player {game.current_player_idx} moves {from_pos} {move}")
        return from_pos, move
    
class ReinforcementPlayer(Player):
    def __init__(self, random_move = 0.3):
        self.value_dictionary = defaultdict(float) # state of the game and its value
        self.hit_state = defaultdict(int) # state of the game and how many times it was visited during the training phase
        self.epsilon = 0.1 # learning rate
        self.random_move = random_move # a value between 0 and 1, used to choose a random move when training

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        best_move_score = -10_000
        best_move = None
        available_moves = game.get_available_moves()

        if random.random() < self.random_move:
            best_move = random.choice(available_moves)
        else:
            for move in available_moves:
                new_state = deepcopy(game)
                from_pos, slide = move
                new_state.make_single_move(from_pos, slide, new_state.current_player_idx)
                hashable_state = np.array2string(new_state._board.flatten(), separator = '')
                actual_move_score = self.value_dictionary[hashable_state]

                # to solve the problem of unseen states that are added to the dictionary with a value of 0 while
                # looking for the best move, if actual_move_score is 0 it means that the state was never visited, in this case
                # we remove the state from the dictionary
                if actual_move_score == 0:
                    del self.value_dictionary[hashable_state]

                if actual_move_score > best_move_score:
                    best_move_score = actual_move_score
                    best_move = move

        from_pos, slide = best_move
        return from_pos, slide
    
    def give_reward(self, reward, trajectory):
        for state in reversed(trajectory):
            self.value_dictionary[state] += 0.2 * (0.9 * reward - self.value_dictionary[state])
            reward = self.value_dictionary[state]

    def print_reward(self):
        return sorted(self.value_dictionary.items(), key = lambda x: x[1], reverse = True)

    def set_random_move(self, random_move):
        self.random_move = random_move

    # creates the policy file where it is stored the value of each state
    def create_policy(self, policy_file):
        """Creates the policy file"""
        # fw = open('policy_' + str(self.player_index), 'wb')
        fw = open(policy_file, 'wb')
        pickle.dump(self.value_dictionary, fw)
        fw.close()

    # loads the policy file
    def load_policy(self, policy_file):
        """Loads the policy file"""
        fr = open(policy_file, 'rb')
        self.value_dictionary = pickle.load(fr)
        fr.close()

    # gets the policy size
    def get_policy_size(self, policy_file):
        """Gets the policy size"""
        size = os.path.getsize(policy_file)
        return size

if __name__ == '__main__':
    # g = Game()
    # g.print()
    # player1 = MyPlayer()
    player1 = ReinforcementPlayer()
    player2 = RandomPlayer()
    # winner = g.play(player1, player2)
    # g.print()
    # print(f"Winner: Player {winner}")

    # training phase
    for _ in tqdm(range(5_000)):
        game = Game()
        win = game.play(player1, player2)
        # print(win)
        if win == 0:
            player1.give_reward(1, game.trajectory)
        elif win == 1:
            player1.give_reward(-1, game.trajectory)
        else:
            player1.give_reward(0, game.trajectory)

    player1.create_policy()
        

    # testing phase
    player1.load_policy()
    player1.set_random_move(0)
    win_rate = 0
    lose_rate = 0
    draw_rate = 0

    for _ in tqdm(range(100)):
        # print("0")
        game = Game()
        # print("1")
        win = game.play(player1, player2)
        # print("A")
        if win == 0:
            # print("B")
            # game.print()
            win_rate += 1
        elif win == 1:
            # print("C")
            # game.print()
            lose_rate += 1
        else:
            # print("D")
            # game.print()
            draw_rate += 1

    print(f"Win rate: {win_rate/100}%")
    print(f"Lose rate: {lose_rate/100}%")
    print(f"Draw rate: {draw_rate/100}%")
