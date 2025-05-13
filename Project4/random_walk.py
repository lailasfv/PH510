#!/bin/python3

import numpy as np
#import matplotlib.pyplot as plt
from mpi4py import MPI
from monte_carlo import MonteCarlo

# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()

def random_walk(i, j, N, grid):
    """
    N random walkers starting at (i,j) in a grid
    """
    newgrid = np.zeros_like(grid)
    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # Possible walks
    x = len(grid) - 1    # x boundary
    y = len(grid[0]) - 1 # y boundary
    num_steps = 0

    for i in range (0, N):
        pos = np.array([i, j])  # Restarts position for each walker
        move = directions[np.random.randint(0, len(directions))]
        pos += move

        while pos[0]>0 and pos[0]<x and pos[1]>0 and pos[1]<y:  # While not at boundary
            newgrid[pos[0], pos[1]] += 1  # adding instance of being at site p,q
            move = directions[np.random.randint(0, len(directions))]
            #print(pos)
            pos += move
            num_steps += 1

        newgrid[pos[0], pos[1]] += 1  # Adds one instance of walker reaching boundary site
        #print("BOUNDARY REACHED:", pos)
        #print(newgrid)
    newgrid[1:-1, 1:-1] = newgrid[1:-1, 1:-1]/num_steps

    return newgrid/N


def boundaries(n, top, bottom, left, right):
    "Adds boundary values to an n by n array"
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

NUM_ITERATIONS = int(1000)  # This is split across cores
NUM_WALKERS = np.array([int(1000)])  # Walkers processed in each iteration
SEED = 27347  # Random seed passed in to class methods

#-------------------------------------------
# The following arrays are used for all of tasks 3-5

n = 101  # Number of points in the grid - odd number 101 or greater
h = 10e-1/n  # Step size, since grid is 10cm x 10cm

grid = np.zeros([n+2, n+2]) # Grid for evaluating Green's at i, j

# Walker starting points
i_val = np.array([int((n-1)/2), int((n+1)/4),
                  int((n+1)/100), int((n+1)/100)])
j_val = np.array([int((n-1)/2), int((n+1)/4),
                  int((n+1)/4), int((n+1)/100)])

vari = np.array([i_val, j_val, NUM_WALKERS, grid], dtype=object)

boundarray = np.ones_like(grid)
boundarray[1:-1, 1:-1] = 0

#-------------------------------------------
# SETTING UP TASKS 3-5

# BOUNDARIES
# n steps, top, bottom, left, right
boundary_4a = boundaries(n, 1, 1, 1, 1)

boundary_4b = boundaries(n, 1, 1, -1, -1)

boundary_4c = boundaries(n, 2, 0, 2, -4)


# CHARGE GRIDS
# We use full size grid to make things easier
grid_4a = np.zeros_like(grid)
grid_4a[1:-1, 1:-1] = 10/(n**2)  # charge not at boundaries

grid_4b = np.zeros_like(grid)
charge_gradient = np.linspace(1, 0, len(grid)-2) # creates the correct gradient scale over the grid
for i in range(1, len(grid_4b)-1):
    grid_4b[i, 1:-1] = charge_gradient[i-1] * h**2
    
# plt.figure(figsize=(6,6))
# plt.pcolormesh(grid_4b, cmap='gist_gray')
# plt.colorbar(label="Intensity")


grid_4c = np.zeros_like(grid)
centre = (len(grid)-1)/2 # works best for odd n

for x in range(1, len(grid_4c)-1):
    for y in range(1, len(grid_4c)-1):
        r = np.sqrt(((x - centre)*h)**2 + ((y - centre)*h)**2)
        grid_4c[x, y] = np.exp(-2000*r)
        
# plt.figure(figsize=(6,6))
# plt.pcolormesh(grid_4c, cmap='gist_gray')
# plt.colorbar(label="Intensity")


#-------------------------------------------
# FINAL RUNNING OF THE CODE FOR TASKS 2-5

# This loop iterates through all 4 starting positions stated in task 3
for i in range(0, len(i_val)):
    vari = np.array([i_val[i], j_val[i], grid], dtype=object)

    setup = MonteCarlo([0], [1], random_walk, variables = vari)
    solve = MonteCarlo.method(setup, NUM_ITERATIONS, seed=SEED, method=0)

    if rank == 0:
        print(f"For i = {(10*i_val[i])/(n-1)}cm and j = {(10*j_val[i])/(n-1)}cm:")
        print(solve[0]*boundarray)
        print("Potential")
        # For boundaries ONLY (first part of task 4)
        print(f"Task 4.1a: All boundaries 1V : \
                potential = {np.sum(solve[0]*boundary_4a):4f}")
        print(f"Task 4.1b: T:1V, B:1V, L:-1V, R:-1V : \
                potential = {np.sum(solve[0]*boundary_4b):4f}")
        print(f"Task 4.1c: T:2V, B:0V, L:2V, R:-4V : \
                potential = {np.sum(solve[0]*boundary_4c):4f}\n")

        # For boundaries AND grid charge
        print(f"Task 4.1a boundaries and uniform grid : \
                {np.sum(solve[0]*boundary_4a) + np.sum(h**2*solve[0]*grid_4a):4f}")
        print(f"Task 4.1b boundaries and uniform grid : \
                {np.sum(solve[0]*boundary_4b) + np.sum(h**2*solve[0]*grid_4a):4f}")
        print(f"Task 4.1c boundaries and uniform grid : \
                {np.sum(solve[0]*boundary_4c) + np.sum(h**2*solve[0]*grid_4a):4f}\n")

        print(f"Task 4.1a boundaries and uniform gradient : \
                {np.sum(solve[0]*boundary_4a) + np.sum(h**2*solve[0]*grid_4b):4f}")
        print(f"Task 4.1b boundaries and uniform gradient : \
                {np.sum(solve[0]*boundary_4b) + np.sum(h**2*solve[0]*grid_4b):4f}")
        print(f"Task 4.1c boundaries and uniform gradient : \
                {np.sum(solve[0]*boundary_4c) + np.sum(h**2*solve[0]*grid_4b):4f}\n")

        print(f"Task 4.1a boundaries and point charge at centre : \
                {np.sum(solve[0]*boundary_4a) + np.sum(h**2*solve[0]*grid_4c):4f}")
        print(f"Task 4.1b boundaries and point charge at centre : \
                {np.sum(solve[0]*boundary_4b) + np.sum(h**2*solve[0]*grid_4c):4f}")
        print(f"Task 4.1c boundaries and point charge at centre : \
                {np.sum(solve[0]*boundary_4c) + np.sum(h**2*solve[0]*grid_4c):4f}\n")
        print("\n")
