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


def random_walk(i, j, x, y):
    """
    Random walker starting at (i,j) in a grid from (0, 0) to (x, y)
    This function should return the final position when the walker
    reaches the grid edge
    """
    pos = np.array([i,j])  # Starting position
    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # Possible walks

    while pos[0]>0 and pos[0]<x and pos[1]>0 and pos[1]<y:  # While not at boundary
        move = directions(np.random.randint(0, len(directions))
        pos += move

    return pos  # returns final position after it reaches boundary


# Next step - finding probability that i,j ends up at x_n, y_n
#             for Green's function

# I really have no idea how to do this rn
# I think we might actually need to make a probability matrix? I'm not sure how yet
