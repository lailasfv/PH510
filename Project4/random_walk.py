#!/bin/python3
"""
Module for solving the Poisson and Laplace equations on a square grid using the random walk method,
in conjunction with a previously created monte_carlo class. Boundary function is used for the
Laplace boundaries, and charge grids for the Poisson equation are created independently.

MIT License

Copyright (c) 2025 Tyler Chauvy, Adam John Rae, Laila Safavi

See LICENSE.txt for details
"""

import numpy as np
from mpi4py import MPI
from monte_carlo import MonteCarlo

# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()

def random_walk(i_index, j_index, grid):
    """
    Walkers starting at a given point i,j in a grid take random steps in one of four directions
    with equal probability. At every step, each walker adds one instance of stepping on that
    grid point. Each walker continues taking random steps until it reaches the grid boundary.
    Function returns a probability grid of any random walker reaching each of the grid points

    Parameters
    ----------
    i_index : INT
        Walker starting point in index 0 of the grid
    j_index : INT
        Walker starting point in index 1 of the grid
    grid : ARRAY
        Empty square grid that the walker begins in

    Returns
    -------
    final_grid : ARRAY
        A grid showing the probability that a walker encounters each grid point

    """
    sum_grid = np.zeros_like(grid)
    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # Possible walks
    x = len(grid) - 1    # x boundary
    y = len(grid[0]) - 1 # y boundary

    num_walkers = 100  # Hardcoded bc variable passing broke

    for walker in range (0, num_walkers):  # Iterates over each walker
        newgrid = np.zeros_like(grid)
        pos = np.array([i_index, j_index])  # Starting position
        move = directions[np.random.randint(0, len(directions))]
        pos += move
        num_steps = 1  # One small step for a walker

        while pos[0]>0 and pos[0]<x and pos[1]>0 and pos[1]<y:  # While not at boundary
            newgrid[pos[0], pos[1]] += 1  # adding instance of being at site p,q
            move = directions[np.random.randint(0, len(directions))]
            pos += move
            num_steps += 1

        newgrid[pos[0], pos[1]] = 1  # Adds one instance of walker reaching boundary site
        newgrid[1:-1, 1:-1] = newgrid[1:-1, 1:-1]/num_steps
        sum_grid += newgrid

    final_grid = sum_grid/num_walkers

    return final_grid


def boundaries(n, top, bottom, left, right):
    """
    A function that adds given boundaries to a grid.

    Parameters
    ----------
    n : INT
        The length of the grid
    top : FLOAT
        The desired boundary value at the top of the grid
    bottom : FLOAT
        The desired boundary value at the bottom of the grid
    left : FLOAT
        The desired boundary value at the left-side of the grid
    right : FLOAT
        The desired boundary value at the right-side of the grid

    Returns
    -------
    boundary_grid : ARRAY
        An empty grid with desired boundary values

    """
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

NUM_ITERATIONS = int(1600)  # This is split across cores
SEED = 27347  # Random seed passed in to class methods

#-------------------------------------------
# The following arrays are used for all of tasks 3-5

if rank == 0:
    print("TRYING FINER GRID SPACING")

GRID_POINTS = 499  # Number of points in the grid
H = 10e-1/GRID_POINTS  # Step size, since grid is 10cm x 10cm

greens_grid = np.zeros([GRID_POINTS+2, GRID_POINTS+2]) # Grid for evaluating Green's at i, j

# Walker starting points
i_val = np.array([int((GRID_POINTS+1)/2), int((GRID_POINTS+1)/4),
                  int((GRID_POINTS+1)/100), int((GRID_POINTS+1)/100)])
j_val = np.array([int((GRID_POINTS+1)/2), int((GRID_POINTS+1)/4),
                  int((GRID_POINTS+1)/4), int((GRID_POINTS+1)/100)])

vari = np.array([i_val, j_val, greens_grid], dtype=object)

boundarray = np.ones_like(greens_grid)
boundarray[1:-1, 1:-1] = 0

#-------------------------------------------
# SETTING UP TASKS 3-5

# BOUNDARIES
# GRID_POINTS steps, top, bottom, left, right
boundary_4a = boundaries(GRID_POINTS, 1, 1, 1, 1)

boundary_4b = boundaries(GRID_POINTS, 1, 1, -1, -1)

boundary_4c = boundaries(GRID_POINTS, 2, 0, 2, -4)


# CHARGE GRIDS
# We use full size grid to make things easier
grid_4a = np.zeros_like(greens_grid)
grid_4a[1:-1, 1:-1] = 10/(GRID_POINTS**2)  # charge not at boundaries

grid_4b = np.zeros_like(greens_grid)
charge_gradient = np.linspace(1, 0, GRID_POINTS) # creates the correct gradient scale over the grid
for i in range(1, len(grid_4b)-1):
    grid_4b[i, 1:-1] = charge_gradient[i-1] * H**2

grid_4c = np.zeros_like(greens_grid)
CENTRE = (len(greens_grid)-1)/2 # works best for odd GRID_POINTS

for k in range(1, len(grid_4c)-1):
    for l in range(1, len(grid_4c)-1):
        r = np.sqrt(((k - CENTRE)*H)**2 + ((l - CENTRE)*H)**2)
        grid_4c[k, l] = np.exp(-2000*r)

#-------------------------------------------
# FINAL RUNNING OF THE CODE FOR TASKS 2-5

# This loop iterates through all 4 starting positions stated in task 3
for i in range(0, len(i_val)):
    vari = np.array([i_val[i], j_val[i], greens_grid], dtype=object)

    setup = MonteCarlo([0], [1], random_walk, variables = vari)
    solve = MonteCarlo.method(setup, NUM_ITERATIONS, seed=SEED, method=0)

    if rank == 0:
        print(f"For i = {(10*i_val[i])/(GRID_POINTS+1)}cm and\
              j = {(10*j_val[i])/(GRID_POINTS+1)}cm:")
        print(solve[0]*boundarray)
        print(f"Average error: {np.mean(solve[2])} \n")
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
                {np.sum(solve[0]*boundary_4a) + np.sum(H**2*solve[0]*grid_4a):4f}")
        print(f"Task 4.1b boundaries and uniform grid : \
                {np.sum(solve[0]*boundary_4b) + np.sum(H**2*solve[0]*grid_4a):4f}")
        print(f"Task 4.1c boundaries and uniform grid : \
                {np.sum(solve[0]*boundary_4c) + np.sum(H**2*solve[0]*grid_4a):4f}\n")

        print(f"Task 4.1a boundaries and uniform gradient : \
                {np.sum(solve[0]*boundary_4a) + np.sum(H**2*solve[0]*grid_4b):4f}")
        print(f"Task 4.1b boundaries and uniform gradient : \
                {np.sum(solve[0]*boundary_4b) + np.sum(H**2*solve[0]*grid_4b):4f}")
        print(f"Task 4.1c boundaries and uniform gradient : \
                {np.sum(solve[0]*boundary_4c) + np.sum(H**2*solve[0]*grid_4b):4f}\n")

        print(f"Task 4.1a boundaries and point charge at centre : \
                {np.sum(solve[0]*boundary_4a) + np.sum(H**2*solve[0]*grid_4c):4f}")
        print(f"Task 4.1b boundaries and point charge at centre : \
                {np.sum(solve[0]*boundary_4b) + np.sum(H**2*solve[0]*grid_4c):4f}")
        print(f"Task 4.1c boundaries and point charge at centre : \
                {np.sum(solve[0]*boundary_4c) + np.sum(H**2*solve[0]*grid_4c):4f}\n")
        print("\n")
