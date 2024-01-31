import numpy as np

def my_hash(board):
    # Transform the original values of the board to their binary representation
    mapping = {-1: '00', 0: '01', 1: '10'}
    # Then, for each row, for each column, each value is appended to create a single string
    binary_string = ''.join(mapping[val] for row in board for val in row)

    # convert the binary string to an integer
    return int(binary_string, 2)

def my_undo_hash(compact_hash, shape=(5, 5)):
    # converts the hash into a binary string
    binary_string = format(compact_hash, f'0{shape[0] * shape[1] * 2}b')

    # Transform the binary string to the original values of the board
    mapping = {'00': -1, '01': 0, '10': 1}
    # Iterates over binary_string, taking 2 bits at a time and converting them to their original value
    values = [mapping[binary_string[i:i+2]] for i in range(0, len(binary_string), 2)]

    # Reshape the values to the original shape of the board
    return np.array(values).reshape(shape)