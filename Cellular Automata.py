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
        - For each cell, measure the local sparsity/density by counting
          the cell's alive neighbors (other cells in unit radius)
        - Change the current cell's state when this count exceeds some
          threshold values (e.g. spawn a cell if count > birth_limit).
    """

    # hyperparameters
    generations = 8    # run the algorithm for n. generations
    init_chance = 0.4  # set alive if init_chance > U(0, 1)
    death_limit = 3    # set dead  if n. alive neighbors < death_limit
    birth_limit = 4    # set alive if n. alive neighbors > birth_limit

    # map initialization
    map = np.ones((rows+2, cols+2))  # out-of-bounds considered alive (1)
    inner = np.random.choice(a=[1, 0],
                             size=(rows, cols),
                             p=[init_chance, 1-init_chance])
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
    
    return map[1:rows+1, 1:cols+1]


def main():
    # example
    np.random.seed(1)
    map = create_map(20, 40)
    map = np.where(map, '#', ' ')
    print(map)


if __name__ == '__main__':
    main()
