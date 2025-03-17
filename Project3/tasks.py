#!/bin/python3
"""
Module for performing integration of functions using MonteCarlo class
"""
import numpy as np
from mpi4py import MPI
from monte_carlo import MonteCarlo

# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()

def step(x, r):
    """
    Function for any round shape with radius R
    Returns 1 if the random point x is within the radius
    and 0 if not

    Parameters
    ----------
    x: Float (random point)
    r: Float (shape radius)

    Returns
    -------
    Integer 1 or 0

    """
    return round(np.sum(np.square(x)), 5) <= (r**2)

def gaussian_1d(x, x0, sig, factor):
    """
    One dimensional gaussian function evaluated at point x

    Parameters
    ----------
    x: Float (random point)
    x0: Float (mean)
    sig: Float (standard deviation)
    factor: Float (multiplication factor for indefinite integral)

    Returns
    -------
    val: Float

    """
    return np.linalg.norm(factor/(sig*(2*np.pi)**0.5) * \
           np.exp((-(x-x0)**2)/(2*sig**2)))

def gaussian_md(x, x0, sig, factor):
    """
    N dimensional gaussian function evaluated at point x

    Parameters
    ----------
    x: Float (random point)
    x0: Float (mean)
    sig: Float (standard deviation)
    factor: Float (multiplication factor for indefinite integral)

    Returns
    -------
    val: Float

    """
    power = len(x)/2
    val = np.linalg.norm(factor/(sig*2*(2*np.pi)**power) * \
    np.exp((-(x-x0)**2)/(2*sig**2)))
    return val


# NOTE: NEW WAY TO PASS IN METHODS - DELETE THESE COMMENTS LATER
# integral = MonteCarlo(start, end, num_points, function, variables, seed, method)
# Methods: 0 = definite integral, 1 = indefinite, 2 = importance sampling (requires func2)


#-------------------------------------------
# INITIALISATION FOR ALL INTEGRALS

NUM_POINTS = int(100000)  # This is split across nodes
SEED = 12345  # Random seed passed in to class methods

# ------------------------------------------
# SHAPES
# All evaluated at unit radius 1

radius = np.array([1])

a1 = np.repeat(-radius, 2)  # Circle Start
b1 = np.repeat(radius, 2)  # Circle End
real = np.pi*radius**2  # Real Circle Area
circle = MonteCarlo(a1, b1, NUM_POINTS, step, seed=SEED,
                    method=0, variables=radius)

a2 = np.repeat(-radius, 3)  # Sphere Start
b2 = np.repeat(radius, 3)  # Sphere End
real2 = 4/3*np.pi*radius**3  # Real 3D Sphere Volume
sphere = MonteCarlo(a2, b2, NUM_POINTS, step, seed=SEED,
                    method=0, variables=radius)

a3 = np.repeat(-radius, 4)  # 4D Hypersphere Start
b3 = np.repeat(radius, 4)  # 4D Hypersphere End
real3 = 1/2*np.pi**2*radius**4  # Real 4D Hypersphere Volume
hypersphere = MonteCarlo(a3, b3, NUM_POINTS, step, seed=SEED,
                         method=0, variables=radius)

a4 = np.repeat(-radius, 5)  # 5D Hypersphere Start
b4 = np.repeat(radius, 5)  # 5D Hypersphere End
real4 = 8/15*np.pi**2*radius**5  # Real 5D Hypersphere Volume
hypersphere2 = MonteCarlo(a4, b4, NUM_POINTS, step, seed=SEED,
                          method=0, variables=radius)


if rank == 0:
    print("TASK 1 - Shapes")
    print(f"2D Circle with radius {radius[0]}: {circle}")
    print(f"Real value: {real}")
    print(f"3D Sphere with radius {radius[0]}: {sphere}")
    print(f"Real value: {real2}")
    print(f"4D Hypersphere with radius {radius[0]}: {hypersphere}")
    print(f"Real value: {real3}")
    print(f"5D Hypersphere with radius {radius[0]}: {hypersphere2}")
    print(f"Real value: {real4}")
    print("\n")

# ------------------------------------------
# GAUSSIAN
# 1D evaluated with different x0 and sigma

MEAN = 4
SIGMA = 0.4
MEAN2 = 3
SIGMA2 = 0.2

vari = np.array([MEAN, SIGMA])
vari2 = np.array([MEAN2, SIGMA2])
gaussOutput = MonteCarlo([-1], [1], NUM_POINTS, gaussian_1d, seed=SEED, method=1,
                         variables=vari)
gaussOutput2 = MonteCarlo([-1], [1], NUM_POINTS, gaussian_1d, seed=SEED, method=1,
                          variables=vari2)

MEAN3 = np.array([2, 5, 6, 9, 4, 2])
SIGMA3 = np.array([0.2, 0.4, 0.1, 0.3, 0.2, 0.5])

vari3 = np.array([MEAN3, SIGMA3])

gaussOutput3 = MonteCarlo([-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1], NUM_POINTS,
                          gaussian_md, seed=SEED, method=1, variables=vari3)

if rank == 0:
    print("TASK 2 - Gaussian")
    print(f"1D Gaussian with mean {MEAN} and SD {SIGMA}: {gaussOutput}")
    print(f"1D Gaussian with mean {MEAN2} and SD {SIGMA2}: {gaussOutput2}")
    print(f"6D Gaussian with mean {MEAN3} and SD {SIGMA3}: {gaussOutput3}")
    print("Real value should always be 1.0 (normalised function)")
    print("\n")

# First Importance Sampling Test
def func_to_integrate_1(x):
    return 2**x

def sampling_func_1(x):
    a = 1/(1-np.exp(-4))
    return a*np.exp(x)

def inverse_sampling_1(x):
    a=1/(1-np.exp(-4))
    return np.log(x/a+np.exp(-4))

def func_with_importance_sampling_1(x):
    return func_to_integrate_1(x)/sampling_func_1(x)


integral_1_with_importance = MonteCarlo([-4], [0], NUM_POINTS, func_with_importance_sampling_1,
                                        seed=SEED, method=2, func2=inverse_sampling_1)

integral_1_without_importance = MonteCarlo([-4], [0], NUM_POINTS, func_to_integrate_1,
                                           seed=SEED, method=0)

if rank == 0:
    print("IMPORTANCE TEST 1: 2^x from -4 to 0")
    print("with importance sampling", integral_1_with_importance)
    print("without importance sampling", integral_1_without_importance)
    print("actual value", 15/(16*np.log(2)))
    print("")


# Second Importance Sampling Test
def func_to_integrate_2(x):
    return np.exp(-x**3)

def sampling_func_2(x):
    a = 1/(1-np.exp(-2))
    return a*(np.exp(-x))

def inverse_sampling_2(x):
    a = 1/(1-np.exp(-2))
    return -np.log(1-x/a)  # THIS SHOULD BE WRONG for lower limits

def func_with_importance_sampling_2(x):
    return func_to_integrate_2(x)/sampling_func_2(x)

integral_2_with_importance = MonteCarlo([0], [2], NUM_POINTS, func_with_importance_sampling_2,
                                        seed=SEED, method=2, func2=inverse_sampling_2)

integral_2_without_importance = MonteCarlo([0], [2], NUM_POINTS, func_to_integrate_2,
                                           seed=SEED, method=0)


if rank==0:
    print("IMPORTANCE TEST 2: exp(-x^3) from 0 to 2")
    print("with importance sampling", integral_2_with_importance)
    print("without importance sampling", integral_2_without_importance)
    print("actual value 0.8929535142938763")
    print("")


# Third Imporptance Sampling Test
def func_to_integrate_3(x):
    return np.exp(-x**2)

def sampling_func_3(x):
    a=1/(1-np.exp(-1))
    return a*np.exp(-x)

def inverse_sampling_3(x):
    a=1/(1-np.exp(-1))
    return -np.log(1-x/a)

def func_with_importance_sampling_3(x):
    return func_to_integrate_3(x)/sampling_func_3(x)

integral_3_with_importance = MonteCarlo([0], [1], NUM_POINTS, func_with_importance_sampling_3,
                                        seed=SEED, method=2, func2=inverse_sampling_3)

integral_3_without_importance = MonteCarlo([0], [1], NUM_POINTS, func_to_integrate_3,
                                           seed=SEED, method=0)
if rank==0:
    print("IMPORTANCE TEST 3: exp(-x^2) from 0 to 1")
    print("with importance sampling", integral_3_with_importance)
    print("without importance sampling", integral_3_without_importance)
    print("actual value 0.746824132812427")
    print("")

MPI.Finalize()
