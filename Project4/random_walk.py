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


def random_walk(i, j, x, y, xn, yn):
    """
    Random walker starting at (i,j) in a grid from (0, 0) to (x, y)
    This function should return whether or not the walker reaches the
    desired boundary point xn, yn.
    """
    pos = np.array([i,j])  # Starting position
    pos2 = np.array([xn, yn]) # The position we hope to get to
    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # Possible walks

    while pos[0]>0 and pos[0]<x and pos[1]>0 and pos[1]<y:  # While not at boundary
        move = directions(np.random.randint(0, len(directions))
        pos += move

    return pos == pos2  # Returns 0 if not true and 1 if true


# I don't know if I like this approach yet
# We'd basically have to have an array of all of the boundary positions (x, y)
# And iterate N times through all of them? Which seems...bleh

# NOTE FOR TOMORROW: I can send in a whole array for xn yn
# no more need to pass in x, y as it'll just be the max of [xn, yn]
# and then when evaluating whether pos == pos2, it can add 1 to the boundary point
# that it DID reach
# I just need to check how this is handled logically in the code
