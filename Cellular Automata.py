import numpy as np


def create_map(rows, cols):
    """
    Return np array of shape (rows, cols) with alive (1) and dead (0) cells
    in organized clusters.

    Idea:
        - start with initial random configuration
        - kill cells where few cells alive (sparsity -> more sparsity)
        - spawn cells where many cells alive (density -> more density)

    Implementation:
        - Measure the local (for each cell) sparsity/density by counting
          the cell's alive neighbors (other cells in a radius of +-1)
        - Change the current cell's state when this count exceeds some
          threshold values (e.g. spawn a cell if count > birth_limit).
    """

    # hyperparameters
    generations = 8    # run the algorithm for n. generations
    init_chance = 0.4  # set alive if init_chance > U(0, 1)
    death_limit = 3    # set dead  if n. alive neighbors < death_limit
    birth_limit = 4    # set alive if n. alive neighbors > birth_limit

    # map initialization
    # We create an expanded array of size (rows+2, cols+2) which will act as
    # outer frame (made of ones) for the true inner map of size (rows, cols).
    # This comes handy for counting the neighbors of edge cells which, in this
    # way, will always be considered alive (1)
    map = np.ones((rows+2, cols+2))
    # The inner map we actually care to return and update has size (rows, cols)
    # and is randomly initialized with zeros and ones
    inner = np.random.choice(a=[1, 0],
                             size=(rows, cols),
                             p=[init_chance, 1-init_chance])
    # The inner map starts at 1 and ends at axis+1 (axis+2-1 = axis+1)
    map[1:rows+1, 1:cols+1] = inner

    # map creation
    for _ in range(generations):
        # needed for simultaneous update
        tmp_map = map.copy()
        # for each cell in inner
        for row in range(1, rows+1):
            for col in range(1, cols+1):
                is_alive = map[row][col]
                # make sure indexes in bounds
                top    = row-1 if (row-1) >= 0 else 0
                left   = col-1 if (col-1) >= 0 else 0
                bottom = row+1 if (row+1) <= (rows+1) else (rows+1)
                right  = col+1 if (col+1) <= (cols+1) else (cols+1)
                # count alive (1) neighbors (do not count current cell)
                n = tmp_map[top:bottom+1, left:right+1].sum() - is_alive
                # change current cell's state if condition met
                if is_alive:
                    if n < death_limit:
                        map[row][col] = 0
                else:
                    if n > birth_limit:
                        map[row][col] = 1

    # return inner
    return map[1:rows+1, 1:cols+1]

def encode_map(map, symbols):
    symbol_true, symbol_false = symbols
    return np.where(map, symbol_true, symbol_false)

def print_arr(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            print(arr[i][j], end='')
        print()


def main():
    # example
    np.random.seed(1)
    map = create_map(20, 40)
    map = encode_map(map, ['#', ' '])
    print_arr(map)


if __name__ == '__main__':
    main()
