#!/bin/python3

import numpy as np
from mpi4py import MPI
from monte_carlo import MonteCarlo

# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()

def random_walk(i, j, grid):
    """
    Random walker starting at (i,j) in a grid
    This function should add 1 to its ending position
    """
    newgrid = np.zeros_like(grid)
    pos = np.array([i,j])  # Starting position
    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # Possible walks
    x = len(grid) - 1    # x boundary
    y = len(grid[0]) - 1 # y boundary

    move = directions[np.random.randint(0, len(directions))]
    pos += move

    while pos[0]>0 and pos[0]<x and pos[1]>0 and pos[1]<y:  # While not at boundary
        newgrid[pos[0], pos[1]] += 1  # adding instance of being at site p,q
        move = directions[np.random.randint(0, len(directions))]
        #print(pos)
        pos += move

    newgrid[pos[0], pos[1]] = 1  # Adds one instance of walker reaching boundary site
    #print("BOUNDARY REACHED:", pos)
    #print(newgrid)

    return newgrid

def boundaries(n, top, bottom, left, right):
    emp_grid = np.zeros([n, n])
    top_row = np.repeat(top, n)
    bot_row = np.repeat(bottom, n)
    left_col = np.repeat(left, n+2)
    right_col = np.repeat(right, n+2)

    # NOTE: The random walker will never reach a corner, so it doesn't
    #       matter which order this is in

    # Adding top and bottom boundaries
    new_rows = np.vstack((top_row, emp_grid, bot_row))
    # Adding left and right boundaries
    boundary_grid = np.column_stack((left_col, new_rows, right_col))

    return boundary_grid


#-------------------------------------------
# GENERAL INITIALISATION

NUM_WALKERS = int(1000)  # This is split across cores
SEED = 27347  # Random seed passed in to class methods

#-------------------------------------------
# The following arrays are used for all of tasks 3-5

# The number of grid points n may be increased by factors of 10
# but please don't change anything else here

n = 100  # Number of points in the grid - can't be smaller than 100
h = 10e-2/n  # Step size, since grid is 10cm x 10cm

grid = np.zeros([n+2, n+2]) # Grid for evaluating Green's at i, j

# Walker starting points
i_val = np.array([int(n/2), int(n/4), 
                  int(n/100), int(n/100)])  # starting position in i
j_val = np.array([int(n/2), int(n/4), 
                  int(n/4), int(n/100)])    # starting position in j

vari = np.array([i_val, j_val, grid], dtype=object)

boundarray = np.ones_like(grid)
boundarray[1:-1, 1:-1] = 0

#-------------------------------------------
# SETTING UP TASKS 3-5

# BOUNDARIES
boundary_4a = boundaries(n, 1, 1, 1, 1)

boundary_4b = boundaries(n, 1, 1, -1, -1)

boundary_4c = boundaries(n, 2, 2, 0, -4)


# CHARGE GRIDS
# We use full size grid to make things easier
grid_4a = np.zeros_like(grid)
grid_4a[1:-1, 1:-1] = 10*h**2  # charge not at boundaries

grid_4b = np.zeros_like(grid)
#for i in range(1, len(grid_4b)-1):

grid_4c = np.zeros_like(grid)
#for i in range(1, len(grid_4c)-1):
#    for j in range(1, len(grid_4c[0])-1):


#-------------------------------------------
# FINAL RUNNING OF THE CODE FOR TASKS 2-5

# This loop iterates through all 4 starting positions stated in task 3
for i in range(0, len(i_val)):
    vari = np.array([i_val[i], j_val[i], grid], dtype=object)

    setup = MonteCarlo([0], [1], random_walk, variables = vari)
    solve = MonteCarlo.method(setup, NUM_WALKERS, seed=SEED, method=0)

    if rank == 0:
        print(f"For i = {i_val[i]} and j = {j_val[i]}, boundary probability grid is:")
        print(solve[0]*boundarray)
        print("Potential")
        # For boundaries ONLY (first part of task 4c)
        print(f"Task 4.1a: All boundaries 1V : \
                potential = {np.sum(solve[0]*boundary_4a):4g}")
        print(f"Task 4.1b: T:1V, B:1V, L:-1V, R:-1V : \
                potential = {np.sum(solve[0]*boundary_4b):4g}")
        print(f"Task 4.1c: T:2V, B:2V, L:0V, R:-4V : \
                potential = {np.sum(solve[0]*boundary_4c):4g}")
        # For boundaries AND grid charge
        # NEED TO ADD AND TEST ALL 9 COMBINATIONS
        print(f"Task 4.1a boundaries and uniform grid : \
                {np.sum(solve[0]*boundary_4a) + h**2*np.sum(solve[0]*grid_4a):4g}")
        print(f"Task 4.1b boundaries and uniform grid : \
                {np.sum(solve[0]*boundary_4b) + h**2*np.sum(solve[0]*grid_4a):4g}")
        print(f"Task 4.1c boundaries and uniform grid : \
                {np.sum(solve[0]*boundary_4c) + h**2*np.sum(solve[0]*grid_4a):4g}")
        print("\n")
