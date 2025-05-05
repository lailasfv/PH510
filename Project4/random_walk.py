#!/bin/python3

import numpy as np
from monte_carlo import MonteCarlo

# just implementing random walk stuff at first
# i will make proper implementation when this makes sense

# montecarlo takes in N random walkers spread across nodes
# each walker started from i,j and walks to boundary x_n, y_n
# probability of walking starting from i,j arriving at x_n, y_n

# ISSUES WITH OUR MONTECARLO:
# 1.   Assessment 3 considered the boundaries in its evaluation
#      with (b-a)*(d-c)*...which we don't want now I think
# 2.   We also don't need the random point generation anymore
#      which will be why he asked us to make a separate method
#      for that, but it is still coded into all the integration

def random_walk(i, j, grid):
    """
    Random walker starting at (i,j) in a grid
    This function should add 1 to its ending position
    """
    pos = np.array([i,j])  # Starting position
    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # Possible walks
    x = len(grid)  # x boundary
    y = len(grid[0])  # y boundary

    while pos[0]>0 and pos[0]<x and pos[1]>0 and pos[1]<y:  # While not at boundary
        move = directions(np.random.randint(0, len(directions))
        pos += move

    grid[pos[0], pos[1]] += 1  # Adds one instance of walker reaching position

    return grid

# We then have N grids summed in montecarlo which are each adding instances of reaching
# boundary positions, and divide by N for probability of each
