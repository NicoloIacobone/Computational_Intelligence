# this class contains the players that will be used in the game (human, reinforcement) that extend the Player class
from game import Game, Move, Player
from my_game import MyGame
from collections import defaultdict
import random
import pickle
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

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
    def __init__(self, random_move = 0.3, learning_rate = 0.1, training = False):
        self.training = training # if we are training the model we need to update the trajectory
        self.value_dictionary = defaultdict(float) # state of the game and its value
        self.learning_rate = learning_rate # used in the reward distribution function
        self.random_move = random_move # a value between 0 and 1, used to choose a random move when training, to favor exploration
        self.trajectory = [] # list of states visited during the training phase
        self.neutral_moves = [] # list of available moves in a given state
        # self.winning_moves = [] # list of winning moves in a given state
        # self.losing_moves = [] # list of losing moves in a given state

    def compute_available_moves(self, game: Game):
        ''' Compute all possible moves in this state '''
        # reset the old list of available moves
        self.neutral_moves = []
        # self.winning_moves = []
        # self.losing_moves = []

        # create a new board to test the moves
        test_board = MyGame(game)

        # call the function that computes all possible moves
        # self.neutral_moves, self.winning_moves, self.losing_moves = test_board.compute_available_moves()
        self.neutral_moves = test_board.compute_available_moves()

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        best_move_score = None # the best move score is initialized with a very low value
        best_move = None # the best move is initialized with None
        never_visited = True # flag to check if the state was never visited
        self.compute_available_moves(game) # compute all possible moves in the current state

        # if len(self.winning_moves) > 0: # if there is a winning move
        #     best_move = random.choice(self.winning_moves)
        # else:
        if random.random() < self.random_move: # if a random number is lower than the random_move value
            best_move = random.choice(self.neutral_moves) # choose a random move to do exploration
        else: # otherwise do exploitation
            test_board = MyGame(game) # create a new board to test the moves
            old_board = test_board._board.copy() # save the old board to restore it after testing a move
            for move in self.neutral_moves: # for each possible move
                from_pos, slide = move # get the move
                _ = test_board._Game__move(from_pos, slide, test_board.current_player_idx)
                # hashable_state = np.array2string(game._board.flatten(), separator = '')
                # hashable_state = tuple(game._board.flatten()) # get the hashable state (use the board as key)
                hashable_state = test_board._board.flatten().tobytes()
                actual_move_score = self.value_dictionary[hashable_state] # get the value of the state
                
                test_board._board = old_board.copy() # restore the old board after testing a move

                # to solve the problem of unseen states that are added to the dictionary with a value of 0 while
                # looking for the best move, if actual_move_score is 0 it means that the state was never visited, in this case
                # we remove the state from the dictionary
                if actual_move_score == 0: # if the state was never visited
                    del self.value_dictionary[hashable_state] # remove the state from the dictionary because it was created from the access to read the value
                else:
                    never_visited = False

                if best_move is None or actual_move_score > best_move_score:
                    best_move_score = actual_move_score
                    best_move = move

            if never_visited: # no state was visited, so we have to choose a random move
                best_move = random.choice(self.neutral_moves) # choose a random move

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
        # hashable_state = np.array2string(test_board._board.flatten(), separator = '') # get the hashable state
        # hashable_state = tuple(test_board._board.flatten()) # get the hashable state (use the board as key)
        hashable_state = test_board._board.flatten().tobytes()
        self.trajectory.append(hashable_state) # add the state to the trajectory

    def give_reward(self, reward, suicide: bool = False, debug = True):
        if debug:
            count = 0
            decrease_point = len(self.trajectory) // 5

            if reward > 0 or suicide:
                self.value_dictionary[self.trajectory[-1]] += reward * 100_000_000
                # huge value assigned to the winning move to not insert an if inside the for that is true only one time

            if not suicide:
                for state in reversed(self.trajectory):
                    self.value_dictionary[state] += reward
                    reward *= 0.95
                    if self.value_dictionary[state] == 0:
                        self.value_dictionary[state] += 1e-50 # a value can't be zero, otherwise it is confused with a never visited state

                    count += 1
                    if count % decrease_point == 0:
                        reward *= 0.5
        else:
            step_penalty = -abs(reward / 100) # 1% of reward is subtracted from each step to favor faster wins
            first_move = True
            if not suicide: # if I haven't made the move that let me lose
                for state in reversed(self.trajectory): # for each move I made, starting from the last one
                    if first_move and reward > 0: # if the reward is positive (i.e. the agent won), give a big reward to the last move 
                        self.value_dictionary[state] += reward * 1_000
                        first_move = False
                    else:
                        self.value_dictionary[state] += self.learning_rate * (0.9 * reward - self.value_dictionary[state]) + step_penalty # distribute the reward
                        reward = self.value_dictionary[state] # reduce the reward that will be used for the next step's reward
                        first_move = False
            else: # if I made the move that let me lose
                self.value_dictionary[self.trajectory[-1]] += reward # apply the negative reward (it is huge) only on the last move
            if self.value_dictionary[state] == 0:
                    self.value_dictionary[state] += 1e-50 # a value can't be zero, otherwise it is confused with a never visited state


    def get_rewards(self):
        return sorted(self.value_dictionary.items(), key = lambda x: x[1], reverse = True)

    def set_random_move(self, random_move):
        self.random_move = random_move

    def set_learning_rate(self, learning_rate):
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
    def __init__(self, depth : int = 3, eval_function : int = 4, maximizing_player : bool = True) -> None:
        super().__init__()
        self.depth = depth
        self.eval_function = eval_function
        self.maximizing_player = maximizing_player

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        _, move = self.minimax(game, self.depth, float('-inf'), float('inf'), self.maximizing_player)
        return move
        
    def minimax(self, state : Game, depth : int, alpha : float, beta : float, maximizing_player : bool):
        winner = state.check_winner()
        if depth == 0 or winner != -1:
            return self.evaluate(state, winner, depth), None
        
        available_moves = self.compute_available_moves(state)

        # shuffle the moves to avoid always choosing the same patterns 
        random.shuffle(available_moves)

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            

            for move in available_moves: # for each possible move --> in minimax lingo, this is "for each child of position"
                child_state = MyGame(state) # create a new board to test the moves
                from_pos, slide = move # get the move
                _ = child_state._Game__move(from_pos, slide, child_state.current_player_idx) # apply the move = create a new state

                eval, _ = self.minimax(child_state, depth - 1, alpha, beta, False) # get the value of the state

                max_eval = max(max_eval, eval) # get the maximum value

                alpha = max(alpha, eval) # update alpha

                if beta <= alpha:
                    break

                # check if the current move is the best move
                if max_eval == eval:
                    best_move = move
            return max_eval, best_move
        
        else:
            min_eval = float('inf')
            best_move = None

            for move in available_moves:
                child_state = MyGame(state)
                from_pos, slide = move
                _ = child_state._Game__move(from_pos, slide, child_state.current_player_idx)

                eval, _ = self.minimax(child_state, depth - 1, alpha, beta, True)
 

                min_eval = min(min_eval, eval)

                beta = min(beta, eval)

                if beta <= alpha:
                    break

                # check if the current move is the best move
                if min_eval == eval:
                    best_move = move
            return min_eval, best_move

    def compute_available_moves(self, game: Game):
        ''' Compute all possible moves in this state '''
        test_board = MyGame(game) # create a new board to test the moves
        available_moves = test_board.compute_available_moves() # call the function that computes all possible moves
        return available_moves # return the list of available moves

    def evaluate(self, game : Game, winner : bool, depth : int = 0):
        if self.eval_function == 1:
            # The simplest evaluation function is to give a positive value to a winning position and a negative value to a losing position.
            if winner == 0:
                return 1
            elif winner == 1:
                return -1
            else:
                return 0
            
        elif self.eval_function == 2:
            # A more sophisticated evaluation function is to give a score based on the number of pieces on winning lines

            scores = [0, 0, 0, 157, 1885, 22621]
            # scores calculated as:
            # scores[0] = 0
            # scores[1] = 1 but set to 0 because it is not useful in the score calculation
            # scores[i] = scores[i - 1] * 12 + 1 for i > 1 where 12 is the size of score_n_concatenated
            # I chose this to be sure that, even in an impossible worse case, the importance of having a single line of length x is always greater than having all lines of length x - 1

            if winner == 0:
                return scores[5]
            elif winner == 1:
                return -scores[5]
            else:
                board = game._board # get the board

                score_0_row = [0, 0, 0, 0, 0] # for player 0, the number of elements on each row
                score_1_row = [0, 0, 0, 0, 0] # for player 1, the number of elements on each row
                score_0_col = [0, 0, 0, 0, 0] # for player 0, the number of elements on each column
                score_1_col = [0, 0, 0, 0, 0] # for player 1, the number of elements on each column
                score_0_diag = [0, 0] # for player 0, the number of elements on each diagonal
                score_1_diag = [0, 0] # for player 1, the number of elements on each diagonal

                for i in range(5): # for each row
                    for j in range(5): # for each column
                        if board[i][j] == 0: # if the element is 0
                            score_0_row[i] += 1 # increase the number of elements on the i-th row
                            score_0_col[j] += 1 # increase the number of elements on the j-th column
                        elif board[i][j] == 1: # if the element is 1
                            score_1_row[i] += 1 # increase the number of elements on the i-th row
                            score_1_col[j] += 1 # increase the number of elements on the j-th column

                        if i == j: # on the first diagonal
                            if board[i][j] == 0:
                                score_0_diag[0] += 1
                            elif board[i][j] == 1:
                                score_1_diag[0] += 1

                        if i + j == 4: # on the second diagonal
                            if board[i][j] == 0:
                                score_0_diag[1] += 1
                            elif board[i][j] == 1:
                                score_1_diag[1] += 1

                score_0_concatenated = score_0_row + score_0_col + score_0_diag # concatenate the lists of rows, columns and diagonals for player 0
                score_1_concatenated = score_1_row + score_1_col + score_1_diag # concatenate the lists of rows, columns and diagonals for player 1

                score_maximising_final = 0 # initialize the score for player 0
                score_minimising_final = 0 # initialize the score for player 1

                for value in score_0_concatenated: # for each element in the list of rows, columns and diagonals for player 0
                    score_maximising_final += scores[value] # add the score of the element to the final score
                for value in score_1_concatenated: # for each element in the list of rows, columns and diagonals for player 1
                    score_minimising_final -= scores[value] # subtract the score of the element to the final score

                if self.maximizing_player: # if the player is the maximizing player
                    return score_maximising_final # return the score for player 0
                else: # if the player is the minimizing player
                    return score_minimising_final # return the score for player 1
                

        elif self.eval_function == 3: 
            # very similar to the previous one, but it gives a higher score to faster wins

            scores = [0, 0, 0, 157, 1885, 22621]
            # scores calculated as:
            # scores[0] = 0
            # scores[1] = 1 but set to 0 because it is not useful in the score calculation
            # scores[i] = scores[i - 1] * 12 + 1 + 3 for i > 1 where 12 is the size of score_n_concatenated and 3 is the depth I usually use (4 is too computational expensive)
            # I chose this to be sure that, even in an impossible worse case, the importance of having a single line of length x is always greater than having all lines of length x - 1

            if winner == 0:
                return scores[5] + depth
            elif winner == 1:
                return -scores[5] - depth
            else:
                board = game._board

                score_0_row = [0, 0, 0, 0, 0]
                score_1_row = [0, 0, 0, 0, 0]
                score_0_col = [0, 0, 0, 0, 0]
                score_1_col = [0, 0, 0, 0, 0]
                score_0_diag = [0, 0]
                score_1_diag = [0, 0]

                for i in range(5):
                    for j in range(5):
                        if board[i][j] == 0:
                            score_0_row[i] += 1
                            score_0_col[j] += 1
                        elif board[i][j] == 1:
                            score_1_row[i] += 1
                            score_1_col[j] += 1

                        if i == j: # on the first diagonal
                            if board[i][j] == 0:
                                score_0_diag[0] += 1
                            elif board[i][j] == 1:
                                score_1_diag[0] += 1

                        if i + j == 4: # on the second diagonal
                            if board[i][j] == 0:
                                score_0_diag[1] += 1
                            elif board[i][j] == 1:
                                score_1_diag[1] += 1

                score_0_concatenated = score_0_row + score_0_col + score_0_diag
                score_1_concatenated = score_1_row + score_1_col + score_1_diag

                score_maximising_final = 0
                score_minimising_final = 0

                for value in score_0_concatenated:
                    score_maximising_final += scores[value]
                for value in score_1_concatenated:
                    score_minimising_final -= scores[value]

                if self.maximizing_player:
                    return score_maximising_final + depth # the same as before but take into account the depth (it favors faster wins)
                else:
                    return score_minimising_final - depth
                
        elif self.eval_function == 4: 
            # very similar to the previous one, but it gives a higher score to faster wins and it takes into account the number of pieces owned by each player

            scores = [0, 0, 0, 521, 6281, 75401]
            # scores calculated as:
            # scores[0] = 0
            # scores[1] = 1 but set to 0 because it is not useful in the score calculation
            # scores[i] = scores[i - 1] * 12 + 1 + 3 + 25 for i > 1 where 12 is the size of score_n_concatenated where 12, 3 is the depth I usually use (4 is too computational expensive), 25 is the number of pieces of the board
            # I chose this to be sure that, even in an impossible worse case, the importance of having a single line of length x is always greater than having all lines of length x - 1

            if winner == 0:
                return scores[5] + depth
            elif winner == 1:
                return -scores[5] - depth
            else:
                board = game._board

                score_0_row = [0, 0, 0, 0, 0]
                score_1_row = [0, 0, 0, 0, 0]
                score_0_col = [0, 0, 0, 0, 0]
                score_1_col = [0, 0, 0, 0, 0]
                score_0_diag = [0, 0]
                score_1_diag = [0, 0]
                pieces_0 = 0
                pieces_1 = 0

                for i in range(5):
                    for j in range(5):
                        if board[i][j] == 0:
                            pieces_0 += 1
                            score_0_row[i] += 1
                            score_0_col[j] += 1
                        elif board[i][j] == 1:
                            pieces_1 += 1
                            score_1_row[i] += 1
                            score_1_col[j] += 1

                        if i == j: # on the first diagonal
                            if board[i][j] == 0:
                                score_0_diag[0] += 1
                            elif board[i][j] == 1:
                                score_1_diag[0] += 1

                        if i + j == 4: # on the second diagonal
                            if board[i][j] == 0:
                                score_0_diag[1] += 1
                            elif board[i][j] == 1:
                                score_1_diag[1] += 1

                score_0_concatenated = score_0_row + score_0_col + score_0_diag
                score_1_concatenated = score_1_row + score_1_col + score_1_diag

                score_maximising_final = 0
                score_minimising_final = 0

                for value in score_0_concatenated:
                    score_maximising_final += scores[value]
                for value in score_1_concatenated:
                    score_minimising_final -= scores[value]

                if self.maximizing_player:
                    return score_maximising_final + depth + pieces_0 # the same as before but take into account the depth (it favors faster wins) and the number of pieces owned by the player
                else:
                    return score_minimising_final - depth - pieces_1

# class ParallelMinimaxPlayer(MinimaxPlayer):
#     def __init__(self, depth: int = 3, eval_function: int = 4, maximizing_player: bool = True, num_processes: int = 4) -> None:
#         super().__init__(depth, eval_function, maximizing_player)
#         self.num_processes = num_processes

#     def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
#         print(self.maximizing_player)
#         maximising_to_pass = self.maximizing_player
#         _, move = self.parallel_minimax(game, self.depth, float('-inf'), float('inf'), maximising_to_pass)
#         return move

#     def parallel_minimax(self, state: Game, depth: int, alpha: float, beta: float, maximizing_player: bool):
#         print(maximizing_player)
#         available_moves = self.compute_available_moves(state)
#         random.shuffle(available_moves)

#         with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
#             futures = [executor.submit(self.minimax, move, state, depth, alpha, beta, maximizing_player) for move in available_moves]

#         results = [future.result() for future in futures]

#         if maximizing_player:
#             max_eval, best_move = max(results, key=lambda x: x[0])
#             return max_eval, best_move
#         else:
#             min_eval, best_move = min(results, key=lambda x: x[0])
#             return min_eval, best_move

#     def minimax(self, move, state, depth, alpha, beta, maximizing_player):
#         child_state = MyGame(state)
#         from_pos, slide = move
#         _ = child_state._Game__move(from_pos, slide, child_state.current_player_idx)

#         eval, _ = super().minimax(child_state, depth - 1, alpha, beta, not maximizing_player)

#         return eval, move

# def minimax_function(move, state, depth, alpha, beta, maximizing_player, player):
#     child_state = MyGame(state)
#     from_pos, slide = move
#     _ = child_state._Game__move(from_pos, slide, child_state.current_player_idx)

#     eval, _ = player.minimax(child_state, depth - 1, alpha, beta, not maximizing_player)

#     return eval, move

# class ParallelMinimaxPlayer(MinimaxPlayer):
#     def __init__(self, depth: int = 3, eval_function: int = 4, maximizing_player: bool = True, num_processes: int = 4) -> None:
#         super().__init__(depth, eval_function, maximizing_player)
#         self.num_processes = num_processes

#     def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
#         print(self.maximizing_player)
#         maximising_to_pass = self.maximizing_player
#         _, move = self.parallel_minimax(game, self.depth, float('-inf'), float('inf'), maximising_to_pass)
#         return move

#     def parallel_minimax(self, state: Game, depth: int, alpha: float, beta: float, maximizing_player: bool):
#         print(maximizing_player)
#         available_moves = self.compute_available_moves(state)
#         random.shuffle(available_moves)

#         with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
#             futures = [executor.submit(minimax_function, move, state, depth, alpha, beta, maximizing_player, self) for move in available_moves]

#         results = [future.result() for future in futures]

#         if maximizing_player:
#             max_eval, best_move = max(results, key=lambda x: x[0])
#             return max_eval, best_move
#         else:
#             min_eval, best_move = min(results, key=lambda x: x[0])
#             return min_eval, best_move

#     def minimax(self, move, state, depth, alpha, beta, maximizing_player):
#         child_state = MyGame(state)
#         from_pos, slide = move
#         _ = child_state._Game__move(from_pos, slide, child_state.current_player_idx)

#         eval, _ = super().minimax(child_state, depth - 1, alpha, beta, not maximizing_player)

#         return eval, move
                

# class ParallelMinimaxPlayer(MinimaxPlayer):
#     def __init__(self, depth: int = 3, eval_function: int = 4, maximizing_player: bool = True, num_processes: int = 4) -> None:
#         super().__init__(depth, eval_function, maximizing_player)
#         self.num_processes = num_processes

#     def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
#         _, move = self.parallel_minimax(game, self.depth, float('-inf'), float('inf'), self.maximizing_player)
#         return move

#     def parallel_minimax(self, state: Game, depth: int, alpha: float, beta: float, maximizing_player: bool):
#         available_moves = self.compute_available_moves(state)
#         random.shuffle(available_moves)

#         with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
#             futures = [executor.submit(self.minimax, move, state, depth, alpha, beta, maximizing_player) for move in available_moves]

#         results = [future.result() for future in futures]

#         if maximizing_player:
#             max_eval, best_move = max(results, key=lambda x: x[0])
#             return max_eval, best_move
#         else:
#             min_eval, best_move = min(results, key=lambda x: x[0])
#             return min_eval, best_move

#     def minimax(self, move, state, depth, alpha, beta, maximizing_player):
#         child_state = MyGame(state)
#         from_pos, slide = move
#         _ = child_state._Game__move(from_pos, slide, child_state.current_player_idx)

#         eval, _ = super().minimax(child_state, depth - 1, alpha, beta, not maximizing_player)

#         return eval, move