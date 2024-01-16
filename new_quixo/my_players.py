# this class contains the players that will be used in the game (human, reinforcement) that extend the Player class
from game import Game, Move, Player
from my_game import MyGame
from collections import defaultdict
import random
import pickle
import os
import numpy as np

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
    def __init__(self, random_move = 0.3, training = False):
        self.training = training # if we are training the model
        self.value_dictionary = defaultdict(float) # state of the game and its value
        # self.hit_state = defaultdict(int) # state of the game and how many times it was visited during the training phase
        # self.epsilon = 0.1 # learning rate
        self.random_move = random_move # a value between 0 and 1, used to choose a random move when training
        self.trajectory = [] # list of states visited during the training phase
        self.available_moves = [] # list of available moves in a given state

    def compute_available_moves(self, game: Game):
        ''' Compute all possible moves in this state '''
        # reset the old list of available moves
        self.available_moves = []

        # create a new board to test the moves
        test_board = MyGame(game)

        # call the function that computes all possible moves
        self.available_moves = test_board.compute_available_moves()

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        best_move_score = -10_000 # the best move score is initialized with a very low value
        best_move = None # the best move is initialized with None
        self.compute_available_moves(game) # compute all possible moves in the current state

        if random.random() < self.random_move: # if a random number is lower than the random_move value
            best_move = random.choice(self.available_moves) # choose a random move to do exploration
        else: # otherwise do exploitation
            test_board = MyGame(game) # create a new board to test the moves
            old_board = test_board._board.copy() # save the old board to restore it after testing a move
            for move in self.available_moves: # for each possible move
                from_pos, slide = move # get the move
                _ = test_board._Game__move(from_pos, slide, test_board.current_player_idx)
                # hashable_state = str(test_board._board.flatten()) # get the hashable state
                hashable_state = np.array2string(game._board.flatten(), separator = '')
                actual_move_score = self.value_dictionary[hashable_state] # get the value of the state
                
                test_board._board = old_board.copy() # restore the old board after testing a move

                # to solve the problem of unseen states that are added to the dictionary with a value of 0 while
                # looking for the best move, if actual_move_score is 0 it means that the state was never visited, in this case
                # we remove the state from the dictionary
                if actual_move_score == 0:
                    del self.value_dictionary[hashable_state]

                if actual_move_score > best_move_score:
                    best_move_score = actual_move_score
                    best_move = move

        from_pos, slide = best_move

        if self.training: # if we are training the model
            self.update_trajectory(game, from_pos, slide)

        return from_pos, slide
    
    def update_trajectory(self, game, from_pos, slide):
        test_board = MyGame(game) # create a new board to apply the move
        # count the number of -1 in the board, if it is 25 or 24, it means the game is started and we have to reset the trajectory
        if np.count_nonzero(test_board._board == -1) == 25 or np.count_nonzero(test_board._board == -1) == 24:
            self.trajectory = []
        test_board._Game__move(from_pos, slide, test_board.current_player_idx) # apply the move
        # hashable_state = str(test_board._board.flatten())
        hashable_state = np.array2string(game._board.flatten(), separator = '') # get the hashable state
        self.trajectory.append(hashable_state) # add the state to the trajectory
    
    def give_reward(self, reward):
        for state in reversed(self.trajectory):
            self.value_dictionary[state] += 0.2 * (0.9 * reward - self.value_dictionary[state])
            reward = self.value_dictionary[state]

    def get_rewards(self):
        return sorted(self.value_dictionary.items(), key = lambda x: x[1], reverse = True)

    def set_random_move(self, random_move):
        self.random_move = random_move

    # creates the policy file where it is stored the value of each state
    def create_policy(self, policy_file):
        """Creates the policy file"""
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