from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import numpy as np

# Rules on PDF

class Move(Enum):
    '''
    Selects where you want to place the taken piece. The rest of the pieces are shifted
    '''
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3


class Player(ABC):
    def __init__(self) -> None:
        '''You can change this for your player if you need to handle state/have memory'''
        pass

    @abstractmethod
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        '''
        The game accepts coordinates of the type (X, Y). X goes from left to right, while Y goes from top to bottom, as in 2D graphics.
        Thus, the coordinates that this method returns shall be in the (X, Y) format.

        game: the Quixo game. You can use it to override the current game with yours, but everything is evaluated by the main game
        return values: this method shall return a tuple of X,Y positions and a move among TOP, BOTTOM, LEFT and RIGHT
        '''
        pass

class HumanPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        # print("Available moves:")
        # for i, move in enumerate(available_moves):
        #     print(f"{i}: {move}")

        # from_pos = tuple(map(int, input("From position: ").split()))
        # move = Move(input("Move: "))
        # from_pos, move = available_moves[int(input("Move: "))]
        x = int(input("Move: "))
        print("x: ", x)
        y = int(input("Move: "))
        print("y: ", y)
        move = Move(int(input("Move: ")))
        print("move: ", move)
        from_pos = (x, y)
        return from_pos, move


class Game(object):
    def __init__(self) -> None:
        self._board = np.ones((5, 5), dtype=np.uint8) * -1
        self.current_player_idx = 1
        self.trajectory_player_1 = list()
        self.trajectory_player_2 = list()
        self.winner = -1
        self.available_moves = list()

    def get_board(self) -> np.ndarray:
        '''
        Returns the board
        '''
        return deepcopy(self._board)

    def get_current_player(self) -> int:
        '''
        Returns the current player
        '''
        return deepcopy(self.current_player_idx)

    def print(self):
        '''Prints the board. -1 are neutral pieces, 0 are pieces of player 0, 1 pieces of player 1'''
        # print(self._board)
        for i in range(5):
            for j in range(5):
                if self._board[i][j] == -1:
                    print('⬜️', end=' ')
                elif self._board[i][j] == 0:
                    print('❌', end=' ')
                else:
                    print('⭕️', end=' ')            
            print()
        print()
        # print(self._board)

    def check_winner(self) -> int:
        '''Check the winner. Returns the player ID of the winner if any, otherwise returns -1'''
        # for each row
        player = self.get_current_player()
        winner = -1
        for x in range(self._board.shape[0]):
            # if a player has completed an entire row
            if self._board[x, 0] != -1 and all(self._board[x, :] == self._board[x, 0]):
                # return winner is this guy
                winner = self._board[x, 0]
        if winner > -1 and winner != self.get_current_player():
            return winner
        # for each column
        for y in range(self._board.shape[1]):
            # if a player has completed an entire column
            if self._board[0, y] != -1 and all(self._board[:, y] == self._board[0, y]):
                # return the relative id
                winner = self._board[0, y]
        if winner > -1 and winner != self.get_current_player():
            return winner
        # if a player has completed the principal diagonal
        if self._board[0, 0] != -1 and all(
            [self._board[x, x]
                for x in range(self._board.shape[0])] == self._board[0, 0]
        ):
            # return the relative id
            winner = self._board[0, 0]
        if winner > -1 and winner != self.get_current_player():
            return winner
        # if a player has completed the secondary diagonal
        if self._board[0, -1] != -1 and all(
            [self._board[x, -(x + 1)]
             for x in range(self._board.shape[0])] == self._board[0, -1]
        ):
            # return the relative id
            winner = self._board[0, -1]
        return winner

    def play(self, player1: Player, player2: Player) -> int:
        '''Play the game. Returns the winning player'''
        # print("1")
        human = False
        players = [player1, player2]
        
        if isinstance(player1, HumanPlayer) or isinstance(player2, HumanPlayer):
            human = True
            
        # print("2")
        winner = -1
        moves = 0
        # print("3")
        while winner < 0 and moves < 100:
            # print("4")
            self.current_player_idx += 1
            # print("5")
            self.current_player_idx %= len(players)
            # print("Player: ", self.current_player_idx)
            # print("6")
            ok = False
            # print("7")
            while not ok:
                # print("8")
                if human:
                    self.print()

                from_pos, slide = players[self.current_player_idx].make_move(self)
                # print("index: ", self.current_player_idx)
                # print("from_pos: ", from_pos)
                # print("slide: ", slide)
                # print("9")
                ok = self.__move(from_pos, slide, self.current_player_idx)
            
            print("Player: ", self.current_player_idx)
            print("from_pos: ", from_pos)
            print("slide: ", slide)
            # print("10")
            # hashable_state = tuple(map(tuple, self._board))
            # hashable_state = tuple(self._board.flatten())
            hashable_state = np.array2string(self._board.flatten(), separator='')
            # print("11")
            # self.trajectory.append(hashable_state)

            if self.current_player_idx == 0:
                self.trajectory_player_1.append(hashable_state)
            else:
                self.trajectory_player_2.append(hashable_state)
            # print("12")

            # print di debug
            # print("Player: ", self.current_player_idx)
            # print("Available moves:")
            # av_moves = self.get_available_moves()
            # print(av_moves)
            # print("move: ", from_pos, " ", slide)
            # self.print()
            winner = self.check_winner()
            # print("13")
            self.winner = winner
            moves += 1
            # print("14")
            # raise Exception
        return winner

    def __move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''
        if player_id > 2:
            return False
        # Oh God, Numpy arrays
        prev_value = deepcopy(self._board[(from_pos[1], from_pos[0])])
        acceptable = self.__take((from_pos[1], from_pos[0]), player_id)
        if acceptable:
            acceptable = self.__slide((from_pos[1], from_pos[0]), slide)
            if not acceptable:
                self._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable
    
    def make_single_move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        return self.__move(from_pos, slide, player_id)

    def __take(self, from_pos: tuple[int, int], player_id: int, make_move: bool = True) -> bool:
        '''Take piece'''
        # print("Taking ", from_pos, " ", player_id, " ", make_move)
        # acceptable only if in border
        acceptable: bool = (
            # check if it is in the first row
            (from_pos[0] == 0 and from_pos[1] < 5)
            # check if it is in the last row
            or (from_pos[0] == 4 and from_pos[1] < 5)
            # check if it is in the first column
            or (from_pos[1] == 0 and from_pos[0] < 5)
            # check if it is in the last column
            or (from_pos[1] == 4 and from_pos[0] < 5)
            # and check if the piece can be moved by the current player
        ) and (self._board[from_pos] < 0 or self._board[from_pos] == player_id)
        if acceptable and make_move:
            self._board[from_pos] = player_id
        # print("Acceptable: ", acceptable)
        return acceptable

    # def __slide(self, from_pos: tuple[int, int], slide: Move, make_move: bool = True) -> bool:
    #     '''Slide the other pieces'''
    #     # define the corners
    #     SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
    #     # if the piece position is not in a corner
    #     if from_pos not in SIDES:
    #         # if it is at the TOP, it can be moved down, left or right
    #         acceptable_top: bool = from_pos[0] == 0 and (
    #             slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
    #         )
    #         # if it is at the BOTTOM, it can be moved up, left or right
    #         acceptable_bottom: bool = from_pos[0] == 4 and (
    #             slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
    #         )
    #         # if it is on the LEFT, it can be moved up, down or right
    #         acceptable_left: bool = from_pos[1] == 0 and (
    #             slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
    #         )
    #         # if it is on the RIGHT, it can be moved up, down or left
    #         acceptable_right: bool = from_pos[1] == 4 and (
    #             slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
    #         )
    #     # if the piece position is in a corner
    #     else:
    #         # if it is in the upper left corner, it can be moved to the right and down
    #         acceptable_top: bool = from_pos == (0, 0) and (
    #             slide == Move.BOTTOM or slide == Move.RIGHT)
    #         # if it is in the lower left corner, it can be moved to the right and up
    #         acceptable_left: bool = from_pos == (4, 0) and (
    #             slide == Move.TOP or slide == Move.RIGHT)
    #         # if it is in the upper right corner, it can be moved to the left and down
    #         acceptable_right: bool = from_pos == (0, 4) and (
    #             slide == Move.BOTTOM or slide == Move.LEFT)
    #         # if it is in the lower right corner, it can be moved to the left and up
    #         acceptable_bottom: bool = from_pos == (4, 4) and (
    #             slide == Move.TOP or slide == Move.LEFT)
    #     # check if the move is acceptable
    #     acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
    #     # if it is
    #     if acceptable and make_move:
    #         # take the piece
    #         piece = self._board[from_pos]
    #         # if the player wants to slide it to the left
    #         if slide == Move.LEFT:
    #             # for each column starting from the column of the piece and moving to the left
    #             for i in range(from_pos[1], 0, -1):
    #                 # copy the value contained in the same row and the previous column
    #                 self._board[(from_pos[0], i)] = self._board[(
    #                     from_pos[0], i - 1)]
    #             # move the piece to the left
    #             self._board[(from_pos[0], 0)] = piece
    #         # if the player wants to slide it to the right
    #         elif slide == Move.RIGHT:
    #             # for each column starting from the column of the piece and moving to the right
    #             for i in range(from_pos[1], self._board.shape[1] - 1, 1):
    #                 # copy the value contained in the same row and the following column
    #                 self._board[(from_pos[0], i)] = self._board[(
    #                     from_pos[0], i + 1)]
    #             # move the piece to the right
    #             self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
    #         # if the player wants to slide it upward
    #         elif slide == Move.TOP:
    #             # for each row starting from the row of the piece and going upward
    #             for i in range(from_pos[0], 0, -1):
    #                 # copy the value contained in the same column and the previous row
    #                 self._board[(i, from_pos[1])] = self._board[(
    #                     i - 1, from_pos[1])]
    #             # move the piece up
    #             self._board[(0, from_pos[1])] = piece
    #         # if the player wants to slide it downward
    #         elif slide == Move.BOTTOM:
    #             # for each row starting from the row of the piece and going downward
    #             for i in range(from_pos[0], self._board.shape[0] - 1, 1):
    #                 # copy the value contained in the same column and the following row
    #                 self._board[(i, from_pos[1])] = self._board[(
    #                     i + 1, from_pos[1])]
    #             # move the piece down
    #             self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
    #     return acceptable
    
    # def get_available_moves(self):
    #     '''Returns a list of available moves for the player'''
    #     available_moves = []
        
    #     # TODO: migliorare efficienza senza chiamare ogni volta tutte le funzioni di take e slide, basta accertarsi numericamente che siano gli elementi del bordo esterno
    #     # for each row
    #     for row in range(5):
    #         # for each column
    #         for col in range(5):
    #             # for each possible move
    #             for move in Move:
    #                 from_pos = (row, col)
    #                 # check if the piece can be taken
    #                 # print("Checking ", from_pos, " ", move)
    #                 # acceptable = self.__take((from_pos[1], from_pos[0]), player_id, False)
    #                 acceptable = self.__take((from_pos[1], from_pos[0]), self.current_player_idx, False)
    #                 if acceptable:
    #                     # check if the pieces can be moved after the piece is taken
    #                     acceptable = self.__slide(from_pos, move, False)
    #                     if acceptable:
    #                         # print("move accepted")
    #                         available_moves.append((from_pos, move))
    #                     # self._board[from_pos] = -1
                            
    #     return available_moves

    # def get_available_moves_old(self):
    #     '''Returns a list of available moves for the player'''
    #     available_moves = []
        
    #     # TODO: migliorare efficienza senza chiamare ogni volta tutte le funzioni di take e slide, basta accertarsi numericamente che siano gli elementi del bordo esterno
    #     # for each row
    #     for row in range(5):
    #         # for each column
    #         for col in range(5):
    #             # for each possible move
    #             for move in Move:
    #                 from_pos = (row, col)
    #                 new_state = deepcopy(self)
    #                 if new_state.move(from_pos, move, new_state.current_player_idx):
    #                     available_moves.append((from_pos, move))
                            
    #     return available_moves

    def __available_moves(self):
        ''' Compute all possible moves in this state '''
        self.available_moves = []
        old_board = deepcopy(self._board)

        for row in range(5):
            for col in range(5):
                if row == 0 or row == 4 or col == 0 or col == 4:
                    for move in Move:
                        possible_move = self.__move((row, col), move, self.current_player_idx)
                        self._board = deepcopy(old_board)
                        if possible_move:
                            self.available_moves.append(((row, col), move))

    def get_available_moves(self):
        '''Returns a list of available moves for the player'''
        self.__available_moves()
        return self.available_moves
    
    # @staticmethod
    # def __acceptable_slides(from_position: tuple[int, int]):
    #     """When taking a piece from {from_position} returns the possible moves (slides)"""
    #     acceptable_slides = [Move.BOTTOM, Move.TOP, Move.LEFT, Move.RIGHT]
    #     axis_0 = from_position[0]    # axis_0 = 0 means uppermost row
    #     axis_1 = from_position[1]    # axis_1 = 0 means leftmost column

    #     if axis_0 == 0:  # can't move upwards if in the top row...
    #         acceptable_slides.remove(Move.TOP)
    #     elif axis_0 == 4:
    #         acceptable_slides.remove(Move.BOTTOM)

    #     if axis_1 == 0:
    #         acceptable_slides.remove(Move.LEFT)
    #     elif axis_1 == 4:
    #         acceptable_slides.remove(Move.RIGHT)
    #     return acceptable_slides

    # def __slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
    #     '''Slide the other pieces'''
    #     if slide not in self.__acceptable_slides(from_pos):
    #         return False  # consider raise ValueError('Invalid argument value')
    #     axis_0, axis_1 = from_pos
    #     # np.roll performs a rotation of the element of a 1D ndarray
    #     if slide == Move.RIGHT:
    #         self._board[axis_0] = np.roll(self._board[axis_0], -1)
    #     elif slide == Move.LEFT:
    #         self._board[axis_0] = np.roll(self._board[axis_0], 1)
    #     elif slide == Move.BOTTOM:
    #         self._board[:, axis_1] = np.roll(self._board[:, axis_1], -1)
    #     elif slide == Move.TOP:
    #         self._board[:, axis_1] = np.roll(self._board[:, axis_1], 1)
    #     return True

    def __slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        '''Slide the other pieces'''
        # define the corners
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom: bool = from_pos[0] == 4 and (
                slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right: bool = from_pos[1] == 4 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top: bool = from_pos == (0, 0) and (
                slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left: bool = from_pos == (4, 0) and (
                slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right: bool = from_pos == (0, 4) and (
                slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom: bool = from_pos == (4, 4) and (
                slide == Move.TOP or slide == Move.LEFT)
        # check if the move is acceptable
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        # if it is
        if acceptable:
            # take the piece
            piece = self._board[from_pos]
            # if the player wants to slide it to the left
            if slide == Move.LEFT:
                # for each column starting from the column of the piece and moving to the left
                for i in range(from_pos[1], 0, -1):
                    # copy the value contained in the same row and the previous column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i - 1)]
                # move the piece to the left
                self._board[(from_pos[0], 0)] = piece
            # if the player wants to slide it to the right
            elif slide == Move.RIGHT:
                # for each column starting from the column of the piece and moving to the right
                for i in range(from_pos[1], self._board.shape[1] - 1, 1):
                    # copy the value contained in the same row and the following column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i + 1)]
                # move the piece to the right
                self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
            # if the player wants to slide it upward
            elif slide == Move.TOP:
                # for each row starting from the row of the piece and going upward
                for i in range(from_pos[0], 0, -1):
                    # copy the value contained in the same column and the previous row
                    self._board[(i, from_pos[1])] = self._board[(
                        i - 1, from_pos[1])]
                # move the piece up
                self._board[(0, from_pos[1])] = piece
            # if the player wants to slide it downward
            elif slide == Move.BOTTOM:
                # for each row starting from the row of the piece and going downward
                for i in range(from_pos[0], self._board.shape[0] - 1, 1):
                    # copy the value contained in the same column and the following row
                    self._board[(i, from_pos[1])] = self._board[(
                        i + 1, from_pos[1])]
                # move the piece down
                self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
        return acceptable
            



''''
X  X  X  X  X  -->  row 0, col 0-4
X           X  -->  row 1, col 0,4
X           X  -->  row 2, col 0,4
X           X  -->  row 3, col 0,4
X  X  X  X  X  -->  row 4, col 0-4
'''
