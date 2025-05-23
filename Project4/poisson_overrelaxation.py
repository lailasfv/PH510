#!/bin/python3
"""
Module for solving the Poisson's equation on a square grid at specified
grid sites, potential edge boundary conditions, and charge boundary condition
inside the grid, giving the potential values at those points under those
conditions.

MIT License

Copyright (c) 2025 Tyler Chauvy, Adam John Rae, Laila Safavi

See LICENSE.txt for details
"""

import numpy as np


def over_relaxation_poisson(grid_length_n, h, f, conditions,
                            max_change_criteria, iteration_limit, w, points):
    """
    Over-relaxation method to calculate the solution of Poisson's
    equation at specific grid sites.

    Parameters
    ----------
    grid_length_n : Integer
        Number of grid point on one side to form a NxN square grid.
    h : Float
        Spacing between grid points, in meters.
    f : Array
        Array of arrays of the form [x,y,f(x,y)]. x is the line
        position and y is the column position both in meters. The
        (0,0)m origin point is at the top left corner of the grid.
        f(x,y) is the charge at point (x,y) in Coulombs.
        Only put elements [x,y,f(x,y)] in the array for when f(x,y)!=0.
        For Laplace equation where f(x,y)=0 everywhere set
        f=np.array([]) and pass this empty array to the function.
        Those are a boundary conditions.
    conditions : Array
        Array of arrays of the form [x,y,V(x,y)]. x is the line
        position and y is the column position both in meters. The
        (0,0)m origin point is at the top left corner of the grid.
        (x,y) positions used in this array must be edge points only.
        V(x,y) is equal to the potential in Volts that is the closest
        to the edge point of the grid (x,y).
        Those are edge boundary conditions.
    max_change_criteria : Float
        This is the convergence criteria. When we have
        abs(grid-grid')<max_change_criteria for all the points on the grid,
        where grid is the grid calculated at the previous iteration
        and grid' is the grid calculated at the iteraiton currently
        running, then the grid elements are deemed to have converged
        and the result grid' is given, stopping the calculations.
        Please note this is not a percentage but an absolute difference.
    iteration_limit : Integer
        Number of iterations the code will run through before "giving up",
        even if it has not reached the desired convergence.
        This is to ensure the code is not running for longer period of time
        than expected and either needs someone to shut it down manually
        or gets shut down automatically without giving the result it got
        to in that amout of itterations.
    w : Float
        Over-relaxation parameter. For w=1, this over-relaxation method
        becomes a Gauss-Siedel relaxation method.
        For this over-relaxation method to converge, we need 0<w<2.
        To have accelerated convergence, we need 1<w<2.
    points: Array
        Array of arrays of the form [x,y] which are the position of
        the grids points, in meters, where we want to know the potential V.

    Returns
    -------
    Array
        Values of the potential, in volts, at the points [x,y] that we passed
        to the function to evaluate the potential at.
        Each potential value corresponds to the point [x,y] at the same index
        in the array 'points'.

    """
    grid_length_n = int(grid_length_n)
    grid = np.zeros([grid_length_n, grid_length_n])  # Creating the base for
    # the NxN grid.
    # We then put the potential bondary conditions on the
    # the edge of the grid we just constructed
    # using the conditions array which described what value should go where
    for cond in conditions:
        # we turn the position (x,y)cm into grid index going from 0 to N-1
        axis1 = int(cond[0]/h)
        axis2 = int(cond[1]/h)
        grid[axis1, axis2] = cond[2]  # we assign the potential value
        # to its grid edge spot
    # then we create a grid_f to store where we have charges on our grid
    # to use in later calculations
    grid_f = np.zeros([grid_length_n, grid_length_n])
    for values in f:
        axis1 = int(values[0]/h)
        axis2 = int(values[1]/h)
        grid_f[axis1, axis2] = values[2]

    # we put random numbers to start calculations with in inner
    # parts of the grid
    k = 1  # we do not want to overwrite our edge condition so we start at
    # index 1 (included)
    while k < grid_length_n-1:
        # we do not want to overwrite our edge condition so we stop at
        # N-2 (included)
        m = 1
        while m < grid_length_n-1:
            # the outer while loop goes through the row and this inner one
            # through the columns for each row
            grid[k, m] = np.random.rand(1)[0]
            m = m+1
        k = k+1

    # We now start the actual calculations and thus do a loop to
    # calculate each iteration
    t = 0
    while t < iteration_limit:
        # This condition is so the code does not run forever or is stopped
        # by a time limit without giving results

        # We copy the previous iterations' results onto a "check"
        #  grid to use them later for a convergence check.
        # This is done element per element to avoid copying issues
        # that were not letting me perform this check properly
        check = np.zeros([grid_length_n, grid_length_n])
        x = 0
        while x < grid_length_n:
            y = 0
            while y < grid_length_n:
                check[x, y] = grid[x, y]
                y = y+1
            x = x+1

        # We then calculate the new grid's element's value for this
        # iteration going line by line and of each line going left to right
        # column wise. We skip the edges that should not change.
        # We use the over-relaxation method's formula and use
        # the latest calculated elements (which were either already
        # calculated by this iteration, or not yet and thus are the ones
        # from the previous iteration).
        k2 = 1
        while k2 < grid_length_n-1:
            m2 = 1
            while m2 < grid_length_n-1:
                grid[k2, m2] = (w/4*(grid_f[k2, m2]*h**2
                                     + (grid[k2+1, m2]
                                        + grid[k2-1, m2]
                                        + grid[k2, m2+1]
                                        + grid[k2, m2-1]))
                                + (1-w)*grid[k2, m2])
                m2 = m2+1
            k2 = k2+1

        # we then look for the biggest abs(phi-phi') value, where phi and
        # phi' are grid element values for the previous iteration and this
        # one
        index1 = 1  # we skip the edges that do not change
        max_change = 0
        while index1 < grid_length_n-1:
            # <N-1 because we skip the edges that do not change
            index2 = 1
            while index2 < grid_length_n-1:
                change = abs(check[index1, index2]-grid[index1,
                                                        index2])
                if change > max_change:
                    max_change = change
                index2 = index2+1
            index1 = index1+1

        # We check if we have reached convergence as described by the
        # criteria given to the method
        if max_change < max_change_criteria:
            # If the grid elements have converged we print the values
            print("We found the result in", t+1, "iterations:")
            # we have t+1 as we start at t=0
            potential_at_points = []  # prepping the return array
            for p in points:
                print("point (", p[0], ",", p[1], ")m: ",
                      str.format("{0:.3e}", grid[int(p[0]/h),
                                                 int(p[1]/h)]),
                      "V", sep="")
                potential_at_points.append(grid[int(p[0]/h), int(p[1]/h)])
                # we print the potential at each desired grid point
            t = iteration_limit  # to stop the while loop
        # If the elements have not converged yet, we check if
        # we are about to reach the iteration limit/if this is
        # the last iteration allowed
        elif t == (iteration_limit-1):
            # And if it is we let the user known we did not reach
            # convergence
            print("We ran out of time!")
            # but still print some results to inform
            potential_at_points = []  # prepping the return array
            for p in points:
                print("point (", p[0], ",", p[1], ")m: ",
                      str.format("{0:.3e}", grid[int(p[0]/h),
                                                 int(p[1]/h)]),
                      "V", sep="")
                potential_at_points.append(grid[int(p[0]/h), int(p[1]/h)])
                # we print the potential at each desired grid point
            # and print what was the biggest change between this last
            # interation and the previous one
            print("max change:", max_change)
            # so they can make an inform decision on how to treat
            # those results and possibly how to change the convergence
            # parameter or iteration limit
        t = t+1
    return np.array(potential_at_points)


def boundary_conditions(grid_length_n, h, top, bottom, left, right):
    """
    Function to quickly make the array of arrays storing the potential (in
    Volts) at the square grid's edge points used by the over-relaxation method
    to solve Poisson equations.

    Parameters
    ----------
    grid_length_n : Integer
        Number of point on each side of the NxN grid.
    h : Float
        Distance between each grid point in cm.
    top : Float
        Potential value at the top edge of the grid, in Volts.
    bottom : Float
        Potential value at the bottom edge of the grid, in Volts.
    left : Float
        Potential value at the left edge of the grid, in Volts.
    right : Float
        Potential value at the right edge of the grid, in Volts.

    Returns
    -------
    Array
        Array of arrays of the form [x,y,V(x,y)]. x is the line
        position and y is the column position both in meters, and (x,y)
        describing only edge points.
        V(x,y) is the potential value the closest to that (x,y) edge point,
        in volts.
    """
    conditions = []  # using a list so we can append elements to it
    index2 = 0
    while index2 < grid_length_n:
        # for each point (0,y) of the top row we append an [0,y,V(0,y)] array:
        conditions.append([0, index2*h, top])
        # for each point ((N-1)*h,y) of the bottom row we append an
        # [(N-1)*h,y,V((N-1)*h,y)] array:
        conditions.append([(grid_length_n-1)*h, index2*h, bottom])
        if 0 < index2 < (N2-1):
            # We want to use the same logic but for the furthest left
            # and furthest right columns, but without rewriting what
            # we have for the corners so we skip index2=0 and index2=N2-1
            # for each non-corner point (x,0) (0<x<N-1) on the
            # furthest left column we append an [x,0,V(x,0)] array:
            conditions.append([index2*h, 0, left])
            # for each non-corner point (x,(N-1)*h) (0<x<N-1) on the
            # furthest right column we append an [x,(N-1)*h,V(x,(N-1)*h)]
            # array:
            conditions.append([index2*h, (grid_length_n-1)*h, right])
        index2 = index2+1
    # we turn the list into an array when returning as the over-relaxation
    # method uses array that are easier to INDEX
    return np.array(conditions)


# TASK 5
LENGTH = 10 / 100  # in meters, as 10cm=10/100m=0.1m
N2 = 101  # number of grid points such that we get H2=0.001m=0.1cm
# this is needed so we can calculate values at points (0.1,2.5)cm and
# (0.1,0.1cm)
H2 = LENGTH/(N2-1)  # in meters, the spacing between the grid points
CONVERGENCE = 10**(-14)
RUN_LIMIT = 2000
w_optimal = 2/(1+np.sin(np.pi/N2))  # the optimal parametter for
# over-relaxation on an NxN square grid
points_desired = np.array([[5.0/100, 5.0/100], [2.5/100, 2.5/100],
                           [0.1/100, 2.5/100], [0.1/100, 0.1/100]])

print("NO CHARGE")
f2 = np.array([])  # empty array as there is no charge in the grid
print("Uniform +1V edges")
conds2_a = boundary_conditions(N2, H2, 1, 1, 1, 1)  # using function to
# create the potential boundaries arrays
task4_relaxation_a = over_relaxation_poisson(N2, H2, f2, conds2_a, CONVERGENCE,
                                             RUN_LIMIT, w_optimal,
                                             points_desired)
print("")
# then using over-relaxation method to get the potential at the desired points
print("Top and bottom edges at +1V, left and right at -1V")
conds2_b = boundary_conditions(N2, H2, 1, 1, -1, -1)
task4_relaxation_b = over_relaxation_poisson(N2, H2, f2, conds2_b, CONVERGENCE,
                                             RUN_LIMIT, w_optimal,
                                             points_desired)
print("")
print("Top and left edges at +2V, bottom at 0V, and right at -4V")
conds2_c = boundary_conditions(N2, H2, 2, 0, 2, -4)
task4_relaxation_c = over_relaxation_poisson(N2, H2, f2, conds2_c, CONVERGENCE,
                                             RUN_LIMIT, w_optimal,
                                             points_desired)
print("")
print("")

print("10C CHARGE SPREAD UNIFORMLY")
# We make the array of arrays describing the charge at each inner grid point
# for a charge of 10C uniformly spread over the inner-grid
INDEX3_1 = 1  # So we do not put it on edges, only the inner-grid
CHARGE_EACH_POINT = 10 / (N2-2)**2  # spreading the charge uniformly on the
# inner grid
f3 = []
while INDEX3_1 < N2-1:
    # going line by line
    # and making sure we do not put a charge on an edge point with <N2-1
    INDEX3_2 = 1
    while INDEX3_2 < N2-1:
        # getting each columns of the line
        f3.append([INDEX3_1*H2, INDEX3_2*H2, CHARGE_EACH_POINT])
        # appending the charge at each inner grid point in the form
        # [x,y,f(x,y)]
        INDEX3_2 = INDEX3_2+1
    INDEX3_1 = INDEX3_1+1
f3_array_a = np.array(f3)  # turning our list into an array for the method

print("Uniform +1V edges")
task4_relaxation_a_a = over_relaxation_poisson(N2, H2, f3_array_a, conds2_a,
                                               CONVERGENCE, RUN_LIMIT,
                                               w_optimal, points_desired)
print("")
print("Top and bottom edges at +1V, left and right at -1V")
task4_relaxation_b_a = over_relaxation_poisson(N2, H2, f3_array_a, conds2_b,
                                               CONVERGENCE, RUN_LIMIT,
                                               w_optimal, points_desired)
print("")
print("Top and left edges at +2V, bottom at 0V, and right at -4V")
task4_relaxation_c_a = over_relaxation_poisson(N2, H2, f3_array_a, conds2_c,
                                               CONVERGENCE, RUN_LIMIT,
                                               w_optimal, points_desired)
print("")
print("")

print("UNIFORM CHARGE GRADIENT")
charge_gradient = np.linspace(1, 0, N2-2)  # creates the correct gradient scale
# over the inner grid
f4 = []
INDEX4_1 = 1
while INDEX4_1 < (N2-1):
    INDEX4_2 = 1
    while INDEX4_2 < (N2-1):
        f4.append([INDEX4_1*H2, INDEX4_2*H2, charge_gradient[INDEX4_1-1]])
        # all elements of the same line gets the same charge
        # but we get an list [x,y,f(x)] for each y position/column on that
        # x/line
        # to skip the edge points we need that 0<x,y<N-1
        # but to get the correct element in charge_gradient we need
        # an INDEX going from 0 (included) to N-2 (included) hence the
        # "-1" in "charge_gradient[INDEX4_1-1]"
        INDEX4_2 = INDEX4_2+1
    INDEX4_1 = INDEX4_1+1
f4_array_b = np.array(f4)
print("Uniform +1V edges")
task4_relaxation_a_b = over_relaxation_poisson(N2, H2, f4_array_b, conds2_a,
                                               CONVERGENCE, RUN_LIMIT,
                                               w_optimal, points_desired)
print("")
print("Top and bottom edges at +1V, left and right at -1V")
task4_relaxation_b_b = over_relaxation_poisson(N2, H2, f4_array_b, conds2_b,
                                               CONVERGENCE, RUN_LIMIT,
                                               w_optimal, points_desired)
print("")
print("Top and left edges at +2V, bottom at 0V, and right at -4V")
task4_relaxation_c_b = over_relaxation_poisson(N2, H2, f4_array_b, conds2_c,
                                               CONVERGENCE, RUN_LIMIT,
                                               w_optimal, points_desired)

print("")
print("")

print("EXPONENTIALLY DECAYING CHARGE")
CENTRE = (N2-1)/2  # We get the index of the grid's centre
# works best for odd N
f5 = []
INDEX5_1 = 1
while INDEX5_1 < N2-1:
    INDEX5_2 = 1
    while INDEX5_2 < N2-1:
        r = np.sqrt(((INDEX5_1 - CENTRE)*H2)**2 + ((INDEX5_2 - CENTRE)*H2)**2)
        # We calculate the radius/distance of the grid element from the centre
        f5.append([INDEX5_1*H2, INDEX5_2*H2, np.exp(-2000*r)])
        # from that get the potential at that value and append [x,y,V(r)]
        # where r is the radius dependent on x,y calculated at the previous
        # line
        INDEX5_2 = INDEX5_2+1
    INDEX5_1 = INDEX5_1+1
f5_array_c = np.array(f5)

print("Uniform +1V edges")
task4_relaxation_a_c = over_relaxation_poisson(N2, H2, f5_array_c, conds2_a,
                                               CONVERGENCE, RUN_LIMIT,
                                               w_optimal, points_desired)
print("")
print("Top and bottom edges at +1V, left and right at -1V")
task4_relaxation_b_c = over_relaxation_poisson(N2, H2, f5_array_c, conds2_b,
                                               CONVERGENCE, RUN_LIMIT,
                                               w_optimal, points_desired)
print("")
print("Top and left edges at +2V, bottom at 0V, and right at -4V")
task4_relaxation_c_c = over_relaxation_poisson(N2, H2, f5_array_c, conds2_c,
                                               CONVERGENCE, RUN_LIMIT,
                                               w_optimal, points_desired)
