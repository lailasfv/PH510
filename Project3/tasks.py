#!/bin/python3
"""
Module for the creation of an object oriented class for Monte Carlo integral methods
"""
import numpy as np
from mpi4py import MPI
from monte_carlo import MonteCarlo

# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()
        

def step(x, R):  # FUNCTION FOR ANY ROUND SHAPE
    return round(np.sum(np.square(x)), 5) <= (R**2)

def test(x, a, b):
    return a*x**2 + b

def gaussian1D(x, x0, sig, factor):
    return np.linalg.norm(factor/(sig*(2*np.pi)**0.5) * np.exp((-(x-x0)**2)/(2*sig**2)))

def gaussianMD(x, x0, sig, factor):
    power = len(x)/2
    val = np.linalg.norm(factor/(sig*2*(2*np.pi)**power) * np.exp((-(x-x0)**2)/(2*sig**2)))
    return val


# NOTE: NEW WAY TO PASS IN METHODS
# integral = MonteCarlo(start, end, num_points, function, variables, seed, method)
# Methods: 0 = definite integral, 1 = indefinite, 2 = importance sampling (requires func2)


num_points = int(1000000)
seed = 12345

radius = np.array([1])
radius2 = np.array([3])

dimensions = 5

a = np.repeat(-radius, dimensions)
b = np.repeat(radius, dimensions)

a2 = np.repeat(-radius2, dimensions)
b2 = np.repeat(radius2, dimensions)

sphere = MonteCarlo(a, b, num_points, step, seed=seed, method=0, variables=radius)
sphere2 = MonteCarlo(a2, b2, num_points, step, seed=seed, method=0, variables=radius2)

real = 8/15*np.pi**2*radius[0]**dimensions
real2 = 8/15*np.pi**2*radius2[0]**dimensions

if rank == 0:
    print(f"5D Sphere with radius {radius[0]}: {sphere}")
    print(f"Real value: {real}")
    print(f"5D Sphere with radius {radius2[0]}: {sphere2}")
    print(f"Real value: {real2}")
    

# TESTING GAUSSIAN 

mean = 4
sigma = 0.4

vari2 = np.array([mean, sigma])
gaussOutput = MonteCarlo([-1], [1], num_points, gaussian1D, seed=seed, method=1,
                         variables=vari2)

mean2 = np.array([2, 5, 6, 9, 4, 2])
sigma2 = np.array([0.2, 0.4, 0.1, 0.3, 0.2, 0.5])

vari2b = np.array([mean2, sigma2])

gaussOutput2 = MonteCarlo([-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1], num_points, 
                          gaussianMD, seed=seed, method=1, variables=vari2b)

if rank == 0:
    print(gaussOutput)
    print(gaussOutput2)
    

# First Importance Sampling Test
def func_to_integrate_1(x):
    return 2**x

def sampling_func_1(x):
    A = 1/(1-np.exp(-4))
    return(A*np.exp(x))

def inverse_sampling_1(x):
    A=1/(1-np.exp(-4))
    return(np.log(x/A+np.exp(-4)))

def func_with_importance_sampling_1(x):
    return (func_to_integrate_1(x)/sampling_func_1(x))


integral_1_with_importance = MonteCarlo([-4], [0], num_points, func_with_importance_sampling_1, 
                                        seed=seed, method=2, func2=inverse_sampling_1)

integral_1_without_importance = MonteCarlo([-4], [0], num_points, func_to_integrate_1, 
                                           seed=seed, method=0)

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
    A = 1/(1-np.exp(-2))
    return A*(np.exp(-x))

def inverse_sampling_2(x):
    A = 1/(1-np.exp(-2))
    return (-np.log(1-x/A))  # THIS SHOULD BE WRONG for lower limits

def func_with_importance_sampling_2(x):
    return (func_to_integrate_2(x)/sampling_func_2(x))

integral_2_with_importance = MonteCarlo([0], [2], num_points, func_with_importance_sampling_2, 
                                        seed=seed, method=2, func2=inverse_sampling_2)

integral_2_without_importance = MonteCarlo([0], [2], num_points, func_to_integrate_2, 
                                           seed=seed, method=0)


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
    A=1/(1-np.exp(-1))
    return A*np.exp(-x)

def inverse_sampling_3(x):
    A=1/(1-np.exp(-1))
    return -np.log(1-x/A)

def func_with_importance_sampling_3(x):
    return (func_to_integrate_3(x)/sampling_func_3(x))

integral_3_with_importance = MonteCarlo([0], [1], num_points, func_with_importance_sampling_3, 
                                        seed=seed, method=2, func2=inverse_sampling_3)

integral_3_without_importance = MonteCarlo([0], [1], num_points, func_to_integrate_3, 
                                           seed=seed, method=0)
if rank==0:
    print("IMPORTANCE TEST 3: exp(-x^2) from 0 to 1")
    print("with importance sampling", integral_3_with_importance)
    #print(f"5D Sphere with radius {radius[0]}: {sphere}")
    print("without importance sampling", integral_3_without_importance)
    print("actual value 0.746824132812427")
    print("")

MPI.Finalize()
