import random
import numpy as np
from collections import defaultdict
from game import Game, Move, Player
from copy import deepcopy
import pickle
import os

class HumanPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        x = int(input("Move: "))
        print("x: ", x)
        y = int(input("Move: "))
        print("y: ", y)
        move = Move(int(input("Move: ")))
        print("move: ", move)
        from_pos = (x, y)
        return from_pos, move

class ReinforcementPlayer(Player):
    def __init__(self, random_move = 0.3):
        self.value_dictionary = defaultdict(float) # state of the game and its value
        self.hit_state = defaultdict(int) # state of the game and how many times it was visited during the training phase
        self.epsilon = 0.1 # learning rate
        self.random_move = random_move # a value between 0 and 1, used to choose a random move when training
        self.trajectory = [] # list of states visited during the training phase
        self.available_moves = [] # list of available moves in a given state

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        best_move_score = -10_000
        best_move = None
        # available_moves = self.__available_moves(game)
        self.compute_available_moves(game)

        if random.random() < self.random_move:
            best_move = random.choice(self.available_moves)
        else:

            for move in self.available_moves:
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
        
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        best_move_score = -10_000
        best_move = None
        self.compute_available_moves(game)

        if random.random() < self.random_move:
            best_move = random.choice(self.available_moves)
        else:
            # old_board = game.get_board()
            old_board = game._board
            for move in self.available_moves:
                from_pos, slide = move
                _ = game.__move(from_pos, slide, game.current_player_idx)
                hashable_state = np.array2string(game._board.flatten(), separator = '')
                actual_move_score = self.value_dictionary[hashable_state]

                game._board = old_board

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

    def compute_available_moves(self, game: Game):
        ''' Compute all possible moves in this state '''
        self.available_moves = []
        # old_board = deepcopy(self._board)
        old_board = game.get_board()

        for row in range(5):
            for col in range(5):
                if row == 0 or row == 4 or col == 0 or col == 4:
                    for move in Move:
                        possible_move = game.__move((row, col), move, game.current_player_idx)
                        # self._board = deepcopy(old_board)
                        game._board = old_board
                        if possible_move:
                            self.available_moves.append(((row, col), move))

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