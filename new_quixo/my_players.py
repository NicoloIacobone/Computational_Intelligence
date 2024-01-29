# this class contains the players that will be used in the game (human, reinforcement) that extend the Player class
from game import Game, Move, Player
from my_game import MyGame
from collections import defaultdict
import random
import pickle
import os
import numpy as np
import tkinter as tk

class HumanPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self.chosen_row = None
        self.chosen_col = None
        self.chosen_move = None
        self.available_moves = None

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:

        self.create_first_grid(game) # create the grid

        from_pos = (self.chosen_row, self.chosen_col)
        return from_pos, self.chosen_move
    
    def create_first_grid(self, game : Game):

        row_col_dict = set() # create a set to store the coordinates of the available moves, we use the set to avoid duplicates
        self.compute_available_moves(game) # compute all possible moves in the current state
        game_board = game._board # get the board

        for move in self.available_moves: # for each possible move
            row, col = move[0]
            row_col_dict.add((col, row)) # add the coordinates of the move to the set (inverted because of the __move function logic)

        root = tk.Tk() # create the window
        root.title("Quixo") # set the title
        root.geometry("325x325") # set the size

        i = 0 # row counter
        j = 0 # column counter
        buttons = []  # list to store the buttons and let me attach the function later
        for row in game_board: # for each row
            button_row = []  # list to store the buttons in the row and let me attach the function later
            for cell in row: # for each column
                if cell == -1: # if the cell is empty
                    button = tk.Button(root, height=3, width=3, text="⬜️") # create a button with the empty symbol
                    button.grid(row=i, column=j)
                elif cell == 0: # if the cell is occupied by a cross
                    button = tk.Button(root, height=3, width=3, text="❌") # create a button with the cross symbol
                    button.grid(row=i, column=j)
                else: # if the cell is occupied by a circle
                    button = tk.Button(root, height=3, width=3, text="⭕️") # create a button with the circle symbol
                    button.grid(row=i, column=j)
                button_row.append(button) # add the button to the row list
                j += 1 # increase the column counter
            buttons.append(button_row) # add the entire row to the buttons list
            i += 1 # increase the row counter
            j = 0 # reset the column counter

        # assignment of the function to each button
        for i, button_row in enumerate(buttons): # for each row
            for j, button in enumerate(button_row): # for each column
                if (i, j) in row_col_dict: # if the coordinates are in the set of available moves
                    button.config(command=lambda row=i, col=j: self.on_button_click(row=col, col=row, valid=True, root=root)) # assign the function to the button passing the coordinates in inverted order
                else:
                    button.config(command=lambda row=i, col=j: self.on_button_click(row=col, col=row, valid=False)) # assign the function to the button passing the coordinates in inverted order

        root.mainloop() # start the window

    def create_second_grid(self):
        ordered_moves = [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT] # list of possible moves
        slide_text = ["⬆️", "⬇️", "⬅️", "➡️"] # list of symbols to show on the buttons
        available_slides = [] # list of available slides

        root = tk.Tk() # create the window
        root.title("Quixo") # set the title
        root.geometry("325x325") # set the size

        for move in self.available_moves: # for each possible move
            if (self.chosen_row, self.chosen_col) == move[0]: # if the coordinates are the same
                available_slides.append(move[1]) # add the slide to the list of available slides

        for i in range(4): # for each possible move
            if ordered_moves[i] in available_slides: # if the slide is available
                button = tk.Button(root, height=3, width=3, text=slide_text[i], fg='black', command=lambda slide=ordered_moves[i]: self.on_button_click(slide=slide, valid=True, root=root)) # create a button with the symbol
                button.grid(row=0, column=i) # place the button
            else: # if the slide is not available
                button = tk.Button(root, height=3, width=3, text="🚫", command=lambda slide=ordered_moves[i]: self.on_button_click(slide, False)) # create a button with the symbol
                button.grid(row=0, column=i) # place the button

        root.mainloop() # start the window

    def on_button_click(self, row = None, col = None, slide = None, valid : bool = None, root : tk.Tk = None):
        if valid:
            if row is not None and col is not None: # if the coordinates are valid
                # print("Row: ", row)
                self.chosen_row = row # save the row
                # print("Col: ", col)
                self.chosen_col = col # save the column
                root.destroy() # destroy the window
                self.create_second_grid() # create the second grid
            elif slide is not None: # if the slide is valid
                # print("Slide: ", slide)
                self.chosen_move = slide # save the slide
                root.destroy() # destroy the window
        else:
            if row is not None and col is not None: # if the coordinates are not valid
                print("Invalid move")
            elif slide is not None: # if the slide is not valid
                print("Invalid slide")

    def compute_available_moves(self, game: Game):
        ''' Compute all possible moves in this state '''
        # reset the old list of available moves just in case
        self.available_moves = []

        # create a new board to test the moves
        test_board = MyGame(game)

        # call the function that computes all possible moves
        self.available_moves = test_board.compute_available_moves()
    
class ReinforcementPlayer(Player):
    def __init__(self, random_move = 0.9, learning_rate = 0.1, training = False):
        self.training = training # if we are training the model we need to update the trajectory
        self.value_dictionary = defaultdict(float) # state of the game and its value
        self.learning_rate = learning_rate # used in the reward distribution function
        self.random_move = random_move # a value between 0 and 1, used to choose a random move when training, to favor exploration
        self.trajectory = [] # list of states visited during the training phase
        self.available_moves = [] # list of available moves in a given state
        self.player_index = None # index of the player (0 or 1)

    def compute_available_moves(self, game: Game):
        ''' Compute all possible moves in this state '''
        # reset the old list of available moves just in case
        self.available_moves = []

        # create a new board to test the moves
        test_board = MyGame(game)

        # call the function that computes all possible moves
        self.available_moves = test_board.compute_available_moves()

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        if self.player_index is None: # if this is the first move of the player
            self.player_index = game.current_player_idx # set the index of the player
            if not self.training: # if we are not training the model
                self.load_policy("policies/policy_" + str(self.player_index)) # load the policy file

        best_move_score = None # the best move score is initialized with a very low value
        best_move = None # the best move is initialized with None
        never_visited = True # flag to check if the state was never visited
        self.compute_available_moves(game) # compute all possible moves in the current state

        if random.random() < self.random_move: # if a random number is lower than the random_move value
            best_move = random.choice(self.available_moves) # choose a random move to do exploration
        else: # otherwise do exploitation
            test_board = MyGame(game) # create a new board to test the moves
            old_board = test_board._board.copy() # save the old board to restore it after testing a move
            actual_move_score = None # the actual move score is initialized with None
            for move in self.available_moves: # for each possible move
                from_pos, slide = move # get the move
                _ = test_board._Game__move(from_pos, slide, test_board.current_player_idx) # apply the move
                # hashable_state = np.array2string(game._board.flatten(), separator = '')
                # hashable_state = tuple(game._board.flatten()) # get the hashable state (use the board as key)

                ######### TESTS ROTATION #########
                # I need a total of 4 hashed states, one for each rotation
                # I need to rotate the board 3 times to get the other 3 hashed states
                # hashable_state = test_board._board.flatten().tobytes() # get the hashable state

                symmetries_set = set() # set of symmetries

                for i in range(4):
                    board_to_hash_rotated = np.rot90(test_board._board, k=i)
                    # board_to_hash_flipped = np.flip(board_to_hash_rotated)
                    board_to_hash_flipped = np.flip(board_to_hash_rotated, axis=0)

                    hashable_state_rotated = board_to_hash_rotated.astype(np.int8).flatten().tobytes() # get the hashable state
                    hashable_state_flipped = board_to_hash_flipped.astype(np.int8).flatten().tobytes()

                    symmetries_set.add(hashable_state_rotated)
                    symmetries_set.add(hashable_state_flipped)

                counter = 1
                for hashable_state in symmetries_set:
                    actual_move_score = self.value_dictionary[hashable_state]
                    if actual_move_score == 0:
                        del self.value_dictionary[hashable_state]
                    else:
                        never_visited = False
                        break
                    if counter == len(symmetries_set):
                        never_visited = False
                    counter += 1

                # for i in range (4):
                #     board_to_hash_rotated = np.rot90(test_board._board, k=i) # rotate the board
                #     board_to_hash_flipped = np.flip(board_to_hash_rotated)

                #     hashable_state_rotated = board_to_hash_rotated.astype(np.int8).flatten().tobytes() # get the hashable state
                #     hashable_state_flipped = board_to_hash_flipped.astype(np.int8).flatten().tobytes()

                #     if hashable_state_rotated == hashable_state_flipped:
                #         actual_move_score_rotated = self.value_dictionary[hashable_state_rotated]

                #         if actual_move_score_rotated == 0:
                #             del self.value_dictionary[hashable_state_rotated]
                #         else:
                #             never_visited = False
                #             actual_move_score = actual_move_score_rotated
                #             break
                #     else:

                #         actual_move_score_rotated = self.value_dictionary[hashable_state_rotated]
                #         actual_move_score_flipped = self.value_dictionary[hashable_state_flipped]

                #         if actual_move_score_rotated != 0 or actual_move_score_flipped != 0:
                #             never_visited = False
                #             if actual_move_score_rotated != 0:
                #                 actual_move_score = actual_move_score_rotated
                #                 del self.value_dictionary[hashable_state_flipped]
                #                 break
                #             else:
                #                 actual_move_score = actual_move_score_flipped
                #                 del self.value_dictionary[hashable_state_rotated]
                #                 break
                #         else:
                #             del self.value_dictionary[hashable_state_rotated]
                #             del self.value_dictionary[hashable_state_flipped]
                        
                
                test_board._board = old_board.copy() # restore the old board after testing a move

                # if actual_move_score == None: # if the state was never visited
                #     actual_move_score = 0 # set the score to 0

                if best_move is None or actual_move_score > best_move_score: # if the actual move score is better than the best move score
                    best_move_score = actual_move_score # update the best move score
                    best_move = move # update the best move

            if never_visited: # no state was visited, so we have to choose a random move
                best_move = random.choice(self.available_moves) # choose a random move

        from_pos, slide = best_move

        if self.training: # if we are training the model
            self.update_trajectory(game, from_pos, slide) # update the trajectory

        return from_pos, slide
    
    def update_trajectory(self, game, from_pos, slide):
        test_board = MyGame(game) # create a new board to apply the move
        # count the number of -1 in the board, if it is 25 or 24, it means the game is started and we have to reset the trajectory
        if np.count_nonzero(test_board._board == -1) == 25 or np.count_nonzero(test_board._board == -1) == 24:
            self.trajectory = []
        test_board._Game__move(from_pos, slide, test_board.current_player_idx) # apply the move
        # hashable_state = np.array2string(test_board._board.flatten(), separator = '') # get the hashable state
        # hashable_state = tuple(test_board._board.flatten()) # get the hashable state (use the board as key)
        hashable_state = test_board._board.astype(np.int8).flatten().tobytes()
        self.trajectory.append(hashable_state) # add the state to the trajectory

    def give_reward(self, reward, suicide: bool = False):
            count = 0
            decrease_point = len(self.trajectory) // 5 # every 1/5 of the elements in the trajectory, the reward is halved
            if decrease_point == 0: # this can happen only when RL player plays second, and the minimum length of trajectory is 4
                decrease_point += 1 # I increase it by 1 because later on there is count % decrease_point and it gives error if the second value is 0


            ######### TESTS ROTATION #########
            if reward > 0 or suicide:
                state_to_check = self.trajectory[-1] # already hashed
                original_state = np.frombuffer(state_to_check, dtype=np.int8).reshape(5, 5) # need to be restored to rotate and flip

                symmetries_set_suicide = set()

                for i in range(4):
                    state_rotated = np.rot90(original_state, k = i) # rotate the board
                    # state_flipped = np.flip(state_rotated)
                    state_flipped = np.flip(state_rotated, axis = 0)

                    hashed_state_rotated = state_rotated.astype(np.int8).flatten().tobytes() # get the hashable state of the rotated state
                    hashed_state_flipped = state_flipped.astype(np.int8).flatten().tobytes() # get the hashable state of the flipped state

                    symmetries_set_suicide.add(hashed_state_rotated) # add the two states to the set
                    symmetries_set_suicide.add(hashed_state_flipped)

                counter = 1
                for hashable_state in symmetries_set_suicide:
                    actual_move_score = self.value_dictionary[hashable_state]
                    if actual_move_score == 0:
                        del self.value_dictionary[hashable_state]
                    else:
                        self.value_dictionary[hashable_state] += 100_000_000
                        break
                    if counter == len(symmetries_set_suicide):
                        self.value_dictionary[hashable_state] += 100_000_000
                        break           
                    counter += 1

                # print("state_to_check:\t", state_to_check)
                # for i in range(4):
                #     original_state = np.frombuffer(state_to_check, dtype=np.int8).reshape(5, 5)

                #     state_rotated = np.rot90(original_state, k = i) # rotate the board
                #     state_flipped = np.flip(state_rotated)

                #     hashed_state_rotated = state_rotated.astype(np.int8).flatten().tobytes() # get the hashable state
                #     hashed_state_flipped = state_flipped.astype(np.int8).flatten().tobytes()

                #     if hashed_state_rotated == hashed_state_flipped:
                #         state_score_to_check_rotated = self.value_dictionary[hashed_state_rotated]
                #         if state_score_to_check_rotated == 0 and i != 3:
                #             del self.value_dictionary[hashed_state_rotated]
                #         else:
                #             self.value_dictionary[hashed_state_rotated] += 100_000_000
                #             break

                #     else:
                #         state_score_to_check_rotated = self.value_dictionary[hashed_state_rotated]
                #         state_score_to_check_flipped = self.value_dictionary[hashed_state_flipped]

                #         if i != 3:
                #             if state_score_to_check_rotated == 0 and state_score_to_check_flipped == 0:
                #                 del self.value_dictionary[hashed_state_rotated]
                #                 del self.value_dictionary[hashed_state_flipped]
                #             elif state_score_to_check_rotated != 0:
                #                 del self.value_dictionary[hashed_state_flipped]
                #                 self.value_dictionary[hashed_state_rotated] += 100_000_000
                #                 break
                #             else:
                #                 del self.value_dictionary[hashed_state_rotated]
                #                 self.value_dictionary[hashed_state_flipped] += 100_000_000
                #                 break
                #         else:
                #             if state_score_to_check_flipped != 0:
                #                 del self.value_dictionary[hashed_state_rotated]
                #                 self.value_dictionary[hashed_state_flipped] += 100_000_000
                #                 break
                #             else:
                #                 del self.value_dictionary[hashed_state_flipped]
                #                 self.value_dictionary[hashed_state_rotated] += 100_000_000
                #                 break
            if not suicide:
                for state in reversed(self.trajectory):
                    symmetries_set_not_suicide = set()
                    for i in range(4):
                        original_state = np.frombuffer(state, dtype=np.int8).reshape(5, 5) # obtain the original array

                        state_rotated = np.rot90(original_state, k = i) # rotate it
                        # state_flipped = np.flip(state_rotated)
                        state_flipped = np.flip(state_rotated, axis = 0)

                        hashed_state_rotated = state_rotated.astype(np.int8).flatten().tobytes() # hash it again (otherwise it doesn't work as key for defaultdict)
                        hashed_state_flipped = state_flipped.astype(np.int8).flatten().tobytes()

                        symmetries_set_not_suicide.add(hashed_state_rotated)
                        symmetries_set_not_suicide.add(hashed_state_flipped)

                    counter = 1
                    for hashable_state in symmetries_set_not_suicide:
                        actual_move_score = self.value_dictionary[hashable_state]
                        if actual_move_score == 0:
                            del self.value_dictionary[hashable_state]
                        else:
                            self.value_dictionary[hashable_state] += reward
                            if self.value_dictionary[hashable_state] == 0:
                                self.value_dictionary[hashable_state] += 1e-50
                            break

                        if counter == len(symmetries_set_not_suicide):
                            self.value_dictionary[hashable_state] += reward
                            if self.value_dictionary[hashable_state] == 0:
                                self.value_dictionary[hashable_state] += 1e-50
                            break
                        counter+=1

                    # for i in range(4):
                    #     original_state = np.frombuffer(state, dtype=np.int8).reshape(5, 5) # obtain the original array

                    #     state_rotated = np.rot90(original_state, k = i) # rotate it
                    #     state_flipped = np.flip(state_rotated)

                    #     hashed_state_rotated = state_rotated.astype(np.int8).flatten().tobytes() # hash it again (otherwise it doesn't work as key for defaultdict)
                    #     hashed_state_flipped = state_flipped.astype(np.int8).flatten().tobytes()

                    #     if hashed_state_rotated == hashed_state_flipped:
                    #         state_score_to_check_rotated = self.value_dictionary[hashed_state_rotated]
                    #         if state_score_to_check_rotated == 0 and i != 3:
                    #             del self.value_dictionary[hashed_state_rotated]
                    #         else:
                    #             self.value_dictionary[hashed_state_rotated] += reward
                    #             if self.value_dictionary[hashed_state_rotated] == 0:
                    #                 self.value_dictionary[hashed_state_rotated] += 1e-50
                    #             break

                    #     else:
                    #         state_score_to_check_rotated = self.value_dictionary[hashed_state_rotated]
                    #         state_score_to_check_flipped = self.value_dictionary[hashed_state_flipped]

                    #         if i != 3:
                    #             if state_score_to_check_rotated == 0 and state_score_to_check_flipped == 0:
                    #                 del self.value_dictionary[hashed_state_rotated]
                    #                 del self.value_dictionary[hashed_state_flipped]
                    #             elif state_score_to_check_rotated != 0:
                    #                 del self.value_dictionary[hashed_state_flipped]
                    #                 self.value_dictionary[hashed_state_rotated] += reward
                    #                 if self.value_dictionary[hashed_state_rotated] == 0:
                    #                     self.value_dictionary[hashed_state_rotated] += 1e-50
                    #                 break
                    #             else:
                    #                 del self.value_dictionary[hashed_state_rotated]
                    #                 self.value_dictionary[hashed_state_flipped] += reward
                    #                 if self.value_dictionary[hashed_state_flipped] == 0:
                    #                     self.value_dictionary[hashed_state_flipped] += 1e-50
                    #                 break

                    #         else:
                    #             if state_score_to_check_flipped != 0:
                    #                 del self.value_dictionary[hashed_state_rotated]
                    #                 self.value_dictionary[hashed_state_flipped] += reward
                    #                 if self.value_dictionary[hashed_state_flipped] == 0:
                    #                     self.value_dictionary[hashed_state_flipped] += 1e-50
                    #                 break
                    #             else:
                    #                 del self.value_dictionary[hashed_state_flipped]
                    #                 self.value_dictionary[hashed_state_rotated] += reward
                    #                 if self.value_dictionary[hashed_state_rotated] == 0:
                    #                     self.value_dictionary[hashed_state_rotated] += 1e-50
                    #                 break
                    
                    reward *= 0.95

                    count += 1
                    if count % decrease_point == 0:
                        reward *= 0.5


    def get_rewards(self):
        # order the dictionary by value (rewards)
        return sorted(self.value_dictionary.items(), key = lambda x: x[1], reverse = True)

    def set_random_move(self, random_move):
        # set the value of random_move to move between exploration and exploitation
        self.random_move = random_move

    def set_learning_rate(self, learning_rate):
        # set the value of learning_rate to distribute the reward
        self.learning_rate = learning_rate

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
    
    # print the first 20 and the last 20 elements of the ordered dictionary
    def print_policy(self):
        """Prints the policy"""
        sorted_dict = self.get_rewards()
        for i in range(20):
            print(sorted_dict[i])
        print("...")
        for i in range(20):
            print(sorted_dict[-i])


    # transform the ordered policy into a txt file
    def policy_to_txt(self, policy_name = "test_policy_1.txt"):
        """Transforms the policy into a txt file"""
        # sort the dictionary
        sorted_dict = self.get_rewards()

        # create the txt file
        fw = open(policy_name, 'w')
        for key, value in sorted_dict:
            fw.write(str(key) + ' ' + str(value) + '\n')
        fw.close()


class MinimaxPlayer(Player):
    def __init__(self, max_depth : int = 3, eval_function : int = 4) -> None:
        super().__init__()
        self.eval_function = eval_function # the evaluation function to use
        self.max_depth = max_depth # the maximum depth of the tree
        self.player_index = None # the index of the player (0 or 1)

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        if self.player_index is None:
            self.player_index = game.current_player_idx # set the index of the player if this is the first move it makes
        _, move = self.minimax(game) # call the minimax function to obtain the best move
        return move # return the best move
    
    def minimax(self, state: Game):
        best_eval = float('-inf') # initialize the best evaluation with a very low value
        alpha = float('-inf') # initialize alpha with a very low value (used for pruning)
        beta = float('inf') # initialize beta with a very high value (used for pruning)

        available_moves = self.compute_available_moves(state) # compute all possible moves in the current state
        best_move = random.choice(available_moves) # choose a random move to avoid selecting always the same move

        for move in available_moves: # for each possible move
            from_pos, slide = move # get the move
            prev_board = state._board.copy() # save the old board to restore it after testing a move
            prev_player = state.current_player_idx # save the old player to restore it after testing a move
            _ = state._Game__move(from_pos, slide, state.current_player_idx) # apply the move
            state.current_player_idx += 1 # change the current player
            state.current_player_idx %= 2
            child_eval = self.child_min(state, alpha, beta, 1) # call the child_min function to get the evaluation of the move
            state._board = prev_board.copy() # restore the old board
            state.current_player_idx = prev_player # restore the old player
            if child_eval > best_eval: # if the evaluation is better than the best evaluation
                best_eval = child_eval # update the best evaluation
                best_move = move # update the best move

            alpha = max(alpha, best_eval) # update alpha

        return best_eval, best_move
    
    def child_min(self, state: Game, alpha : float, beta : float, depth : int): 
        winner = state.check_winner() # check if there is a winner

        if winner != -1: # if there is a winner
            if winner == self.player_index: # if the winner is the player
                return 999999 # return a very high value
            else:
                return -999999 # return a very low value

        if depth == self.max_depth: # if we have reached the maximum depth
            return self.evaluate(state, winner, depth) # call the evaluation function
        
        best_eval = float('inf') # initialize the best evaluation with a very high value
        possible_moves = self.compute_available_moves(state) # compute all possible moves in the current state

        for move in possible_moves: # for each possible move
            from_pos, slide = move # get the move
            prev_board = state._board.copy() # save the old board to restore it after testing a move
            prev_player = state.current_player_idx # save the old player to restore it after testing a move
            _ = state._Game__move(from_pos, slide, state.current_player_idx) # apply the move
            state.current_player_idx += 1 # change the current player
            state.current_player_idx %= 2
            child_eval = self.child_max(state, alpha, beta, depth + 1) # call the child_max function to get the evaluation of the move
            best_eval = min(best_eval, child_eval) # update the best evaluation
            state._board = prev_board.copy() # restore the old board
            state.current_player_idx = prev_player # restore the old player

            beta = min(beta, best_eval) # update beta

            if beta <= alpha: # if beta is lower than alpha
                break # prune the tree
            
        return best_eval
    
    def child_max(self, state: Game, alpha : float, beta : float, depth : int):
        winner = state.check_winner() # check if there is a winner
        if winner != -1: # if there is a winner
            if winner == self.player_index: # if the winner is the player
                return 999999 # return a very high value
            else:
                return -999999 # return a very low value

        if depth == self.max_depth: # if we have reached the maximum depth
            return self.evaluate(state, winner, depth) # call the evaluation function
        
        best_eval = float('-inf') # initialize the best evaluation with a very low value
        possible_moves = self.compute_available_moves(state) # compute all possible moves in the current state

        for move in possible_moves: # for each possible move
            from_pos, slide = move # get the move
            prev_board = state._board.copy() # save the old board to restore it after testing a move
            prev_player = state.current_player_idx # save the old player to restore it after testing a move
            _ = state._Game__move(from_pos, slide, state.current_player_idx) # apply the move
            state.current_player_idx += 1 # change the current player
            state.current_player_idx %= 2
            child_eval = self.child_min(state, alpha, beta, depth + 1) # call the child_min function to get the evaluation of the move
            best_eval = max(best_eval, child_eval) # update the best evaluation
            state._board = prev_board.copy() # restore the old board
            state.current_player_idx = prev_player # restore the old player

            alpha = max(alpha, best_eval) # update alpha

            if beta <= alpha: # if beta is lower than alpha
                break # prune the tree
            
        return best_eval

    def compute_available_moves(self, game: Game):
        ''' Compute all possible moves in this state '''
        test_board = MyGame(game) # create a new board to test the moves
        available_moves = test_board.compute_available_moves() # call the function that computes all possible moves
        return available_moves # return the list of available moves
    
    def evaluate(self, game : Game, winner : int, depth : int = 0):
        if self.eval_function == 0:
            # The simplest evaluation function is to give a positive value to a winning position and a negative value to a losing position.
            return 0
        
        elif self.eval_function == 1:
            # A slightly more sophisticated evaluation function is to give a score based on the number of pieces owned on the board
            caselle_X = 0 # number of pieces owned by player 0
            caselle_O = 0 # number of pieces owned by player 1

            for i in range(5): # for each row
                for j in range(5): # for each column
                    if game._board[i][j] == 0: # if the element is 0
                        caselle_X += 1 # increase the number of pieces owned by player 0
                    elif game._board[i][j] == 1: # if the element is 1
                        caselle_O += 1 # increase the number of pieces owned by player 1

            if self.player_index == 0: # if the player is 0
                return caselle_X # return the number of pieces owned by player 0
            else: # if the player is 1
                return caselle_O # return the number of pieces owned by player 1
            
        elif self.eval_function == 2:
            # A more sophisticated evaluation function is to give a score based on the number of pieces on winning lines

            scores = [0, 0, 0, 157, 1885, 22621]
            # scores calculated as:
            # scores[0] = 0
            # scores[1] = 1 but set to 0 because it is not useful in the score calculation
            # scores[i] = scores[i - 1] * 12 + 1 for i > 1 where 12 is the size of score_n_concatenated
            # I chose this to be sure that, even in an impossible worse case, the importance of having a single line of length x is always greater than having all lines of length x - 1

            board = game._board # get the board

            if self.player_index == 0:
                score_0_row = [0, 0, 0, 0, 0] # for player 0, the number of elements on each row
                score_0_col = [0, 0, 0, 0, 0] # for player 0, the number of elements on each column
                score_0_diag = [0, 0] # for player 0, the number of elements on each diagonal

                for i in range(5): # for each row
                    for j in range(5): # for each column
                        if board[i][j] == 0: # if the element is 0
                            score_0_row[i] += 1 # increase the number of elements on the i-th row
                            score_0_col[j] += 1 # increase the number of elements on the j-th column

                            if i == j: # on the first diagonal
                                score_0_diag[0] += 1

                            if i + j == 4: # on the second diagonal
                                score_0_diag[1] += 1

                score_0_concatenated = score_0_row + score_0_col + score_0_diag # concatenate the lists of rows, columns and diagonals for player 0
                score_0_final = 0 # initialize the score for player 0

                for value in score_0_concatenated: # for each element in the list of rows, columns and diagonals for player 0
                    score_0_final += scores[value] # add the score of the element to the final score

                return score_0_final # return the score for player 0
            
            else:
                score_1_row = [0, 0, 0, 0, 0] # for player 1, the number of elements on each row
                score_1_col = [0, 0, 0, 0, 0] # for player 1, the number of elements on each column 
                score_1_diag = [0, 0] # for player 1, the number of elements on each diagonal

                for i in range(5): # for each row
                    for j in range(5): # for each column
                        if board[i][j] == 1: # if the element is 1
                            score_1_row[i] += 1 # increase the number of elements on the i-th row
                            score_1_col[j] += 1 # increase the number of elements on the j-th column

                            if i == j: # on the first diagonal
                                score_1_diag[0] += 1

                            if i + j == 4: # on the second diagonal
                                score_1_diag[1] += 1

                score_1_concatenated = score_1_row + score_1_col + score_1_diag # concatenate the lists of rows, columns and diagonals for player 1
                score_1_final = 0 # initialize the score for player 1

                for value in score_1_concatenated: # for each element in the list of rows, columns and diagonals for player 1
                    score_1_final += scores[value] # subtract the score of the element to the final score

                return score_1_final # return the score for player 1
                

        elif self.eval_function == 3: 
            # very similar to the previous one, but it gives a higher score to faster wins

            scores = [0, 0, 0, 196, 2356, 28276]
            # scores calculated as:
            # scores[0] = 0
            # scores[1] = 1 but set to 0 because it is not useful in the score calculation
            # scores[i] = scores[i - 1] * 12 + 1 + 3 for i > 1 where 12 is the size of score_n_concatenated and 3 is the depth I usually use (4 is too computational expensive)
            # I chose this to be sure that, even in an impossible worse case, the importance of having a single line of length x is always greater than having all lines of length x - 1

            board = game._board # get the board

            if self.player_index == 0:
                score_0_row = [0, 0, 0, 0, 0] # for player 0, the number of elements on each row
                score_0_col = [0, 0, 0, 0, 0] # for player 0, the number of elements on each column
                score_0_diag = [0, 0] # for player 0, the number of elements on each diagonal

                for i in range(5): # for each row
                    for j in range(5): # for each column
                        if board[i][j] == 0: # if the element is 0
                            score_0_row[i] += 1 # increase the number of elements on the i-th row
                            score_0_col[j] += 1 # increase the number of elements on the j-th column

                            if i == j: # on the first diagonal
                                score_0_diag[0] += 1

                            if i + j == 4: # on the second diagonal
                                score_0_diag[1] += 1

                score_0_concatenated = score_0_row + score_0_col + score_0_diag # concatenate the lists of rows, columns and diagonals for player 0
                score_0_final = 0 # initialize the score for player 0

                for value in score_0_concatenated: # for each element in the list of rows, columns and diagonals for player 0
                    score_0_final += scores[value] # add the score of the element to the final score

                return score_0_final - depth # return the score for player 0
            
            else:
                score_1_row = [0, 0, 0, 0, 0] # for player 1, the number of elements on each row
                score_1_col = [0, 0, 0, 0, 0] # for player 1, the number of elements on each column 
                score_1_diag = [0, 0] # for player 1, the number of elements on each diagonal

                for i in range(5): # for each row
                    for j in range(5): # for each column
                        if board[i][j] == 1: # if the element is 1
                            score_1_row[i] += 1 # increase the number of elements on the i-th row
                            score_1_col[j] += 1 # increase the number of elements on the j-th column

                            if i == j: # on the first diagonal
                                score_1_diag[0] += 1

                            if i + j == 4: # on the second diagonal
                                score_1_diag[1] += 1

                score_1_concatenated = score_1_row + score_1_col + score_1_diag # concatenate the lists of rows, columns and diagonals for player 1
                score_1_final = 0 # initialize the score for player 1

                for value in score_1_concatenated: # for each element in the list of rows, columns and diagonals for player 1
                    score_1_final += scores[value] # subtract the score of the element to the final score

                return score_1_final - depth # return the score for player 1
                
        elif self.eval_function == 4: 
            # very similar to the previous one, but it gives a higher score to faster wins and it takes into account the number of pieces owned by each player

            scores = [0, 0, 41, 521, 6281, 75401]
            # scores calculated as:
            # scores[0] = 0
            # scores[1] = 1 but set to 0 because it is not useful in the score calculation
            # scores[i] = scores[i - 1] * 12 + 1 + 3 + 25 for i > 1 where 12 is the size of score_n_concatenated where 12, 3 is the depth I usually use (4 is too computational expensive), 25 is the number of pieces of the board
            # I chose this to be sure that, even in an impossible worse case, the importance of having a single line of length x is always greater than having all lines of length x - 1

            board = game._board # get the board

            if self.player_index == 0:
                score_0_row = [0, 0, 0, 0, 0] # for player 0, the number of elements on each row
                score_0_col = [0, 0, 0, 0, 0] # for player 0, the number of elements on each column
                score_0_diag = [0, 0] # for player 0, the number of elements on each diagonal
                pieces_0 = 0 # number of pieces owned by player 0

                for i in range(5): # for each row
                    for j in range(5): # for each column
                        if board[i][j] == 0: # if the element is 0
                            pieces_0 += 1 # increase the number of pieces owned by player 0

                            score_0_row[i] += 1 # increase the number of elements on the i-th row
                            score_0_col[j] += 1 # increase the number of elements on the j-th column

                            if i == j: # on the first diagonal
                                score_0_diag[0] += 1

                            if i + j == 4: # on the second diagonal
                                score_0_diag[1] += 1

                score_0_concatenated = score_0_row + score_0_col + score_0_diag # concatenate the lists of rows, columns and diagonals for player 0
                score_0_final = 0 # initialize the score for player 0

                for value in score_0_concatenated: # for each element in the list of rows, columns and diagonals for player 0
                    score_0_final += scores[value] # add the score of the element to the final score

                return score_0_final - depth + pieces_0 # return the score for player 0
            
            else:
                score_1_row = [0, 0, 0, 0, 0] # for player 1, the number of elements on each row
                score_1_col = [0, 0, 0, 0, 0] # for player 1, the number of elements on each column 
                score_1_diag = [0, 0] # for player 1, the number of elements on each diagonal
                pieces_1 = 0 # number of pieces owned by player 1

                for i in range(5): # for each row
                    for j in range(5): # for each column
                        if board[i][j] == 1: # if the element is 1
                            pieces_1 += 1 # increase the number of pieces owned by player 1

                            score_1_row[i] += 1 # increase the number of elements on the i-th row
                            score_1_col[j] += 1 # increase the number of elements on the j-th column

                            if i == j: # on the first diagonal
                                score_1_diag[0] += 1

                            if i + j == 4: # on the second diagonal
                                score_1_diag[1] += 1

                score_1_concatenated = score_1_row + score_1_col + score_1_diag # concatenate the lists of rows, columns and diagonals for player 1
                score_1_final = 0 # initialize the score for player 1

                for value in score_1_concatenated: # for each element in the list of rows, columns and diagonals for player 1
                    score_1_final += scores[value] # subtract the score of the element to the final score

                return score_1_final - depth + pieces_1 # return the score for player 1
                    