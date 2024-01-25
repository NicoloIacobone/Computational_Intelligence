# this class extends the Game class
from game import Game, Move

class MyGame(Game):
    def __init__(self, game: Game = None):
        super().__init__() # self._board, self.current_player_idx

        # this is used to create a copy of the game when we need to test a move
        if game is not None:
            self._board = game._board.copy()
            self.current_player_idx = game.current_player_idx
    
    def nice_print(self):
        '''Prints the board. -1 are neutral pieces, 0 are pieces of player 0, 1 pieces of player 1'''
        board_str = ""

        for row in self._board:
            for cell in row:
                if cell == -1:
                    board_str += '⬜️ '
                elif cell == 0:
                    board_str += '❌ '
                else:
                    board_str += '⭕️ '

            board_str += '\n'

        print(board_str)

    def compute_available_moves(self):
        ''' Compute all possible moves in this state '''
        available_moves = [] # list of available moves in a given state
        old_board = self._board.copy() # save the old board to restore it after testing a move

        for row in range(5): # for each row
            for col in range(5): # for each column
                if row == 0 or row == 4 or col == 0 or col == 4: # if the cell is on the border
                    for move in Move: # for each possible move
                        possible_move = self._Game__move((row, col), move, self.current_player_idx) # test the move
                        self._board = old_board.copy() # restore the old board
                        if possible_move: # if the move is possible
                            available_moves.append(((row, col), move)) # add the move to the list of available moves

        return available_moves # return the list of available moves
    
    def make_move(self, row, col, move):
        ''' Make a move in the game '''
        return self._Game__move((row, col), move, self.current_player_idx) # make the move
        