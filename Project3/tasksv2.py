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

def func_to_integrate_1(x):
    """
    Function f(x)=2**x to be integrated

    Parameters
    ----------
    x : Float
        Variable.

    Returns
    -------
    Float
        f(x)=2**x.

    """
    return 2**x

def sampling_func_1(x):
    """
    Sampling function f(x)=exp(x)/(1-exp(-4)) normalised for [-4,0] to
    be used when integrating a function over the interval [-4,0]
    using importance sampling

    Parameters
    ----------
    x : Float
        Variable.

    Returns
    -------
    Float
        f(x)=exp(x)/(1-exp(-4)).

    """
    a = 1/(1-np.exp(-4))
    return a*np.exp(x)

def inverse_sampling_1(x):
    """
    Inverse of the definite integral of the sampling function,
    f(x')=exp(x')/(1-exp(-4)), with lower limit -4 and higher limit x

    Parameters
    ----------
    x : Float
        Variable.

    Returns
    -------
    Float
        f(x)=ln(x/(1-exp(-4))+exp(-4))

    """
    a = 1/(1-np.exp(-4))
    return np.log(x/a+np.exp(-4))

def func_to_integrate_2(x):
    """
    Function f(x)=exp(-x**3) to be integrated

    Parameters
    ----------
    x : Float
        Variable.

    Returns
    -------
    Float
        f(x)=exp(-x**3).

    """
    return np.exp(-x**3)

def sampling_func_2(x):
    """
    Sampling function f(x)=exp(-x)/(1-exp(-2)) normalised for [0,2] to
    be used when integrating a function over the interval [0,2]
    using importance sampling

    Parameters
    ----------
    x : Float
        Variable.

    Returns
    -------
    Float
        f(x)=exp(-x)/(1-exp(-2)).

    """
    a = 1/(1-np.exp(-2))
    return a*(np.exp(-x))

def inverse_sampling_2(x):
    """
    Inverse of the definite integral of the sampling function,
    f(x')=exp(-x')/(1-exp(-2)), with lower limit 0 and higher limit x

    Parameters
    ----------
    x : Float
        Variable.

    Returns
    -------
    Float
        f(x)=-ln(1-x/(1-exp(-2)))

    """
    a = 1/(1-np.exp(-2))
    return -np.log(1-x/a)  # THIS SHOULD BE WRONG for lower limits

def func_to_integrate_3(x):
    """
    Function f(x)=exp(-x**2) to be integrated

    Parameters
    ----------
    x : Float
        Variable.

    Returns
    -------
    Float
        f(x)=exp(-x**2).

    """
    return np.exp(-x**2)

def sampling_func_3(x):
    """
    Sampling function f(x)=exp(-x)/(1-exp(-1)) normalised for [0,1] to
    be used when integrating a function over the interval [0,1]
    using importance sampling

    Parameters
    ----------
    x : Float
        Variable.

    Returns
    -------
    Float
        f(x)=exp(-x)/(1-exp(-1)).

    """
    a = 1/(1-np.exp(-1))
    return a*np.exp(-x)

def inverse_sampling_3(x):
    """
    Inverse of the definite integral of the sampling function,
    f(x')=exp(-x')/(1-exp(-1)), with lower limit 0 and higher limit x

    Parameters
    ----------
    x : Float
        Variable.

    Returns
    -------
    Float
        f(x)=-ln(1-x/(1-exp(-1)))

    """
    a = 1/(1-np.exp(-1))
    return -np.log(1-x/a)

#-------------------------------------------
# INITIALISATION FOR ALL INTEGRALS

NUM_POINTS = int(1000000)  # This is split across nodes
SEED = 12345  # Random seed passed in to class methods

# ------------------------------------------
# SHAPES
# All evaluated at unit radius 1

radius = np.array([1])

a1 = np.repeat(-radius, 2)  # Circle Start
b1 = np.repeat(radius, 2)  # Circle End
real = np.pi*radius**2  # Real Circle Area
circle_setup = MonteCarlo(a1, b1, step, variables=radius)
circle = MonteCarlo.method(circle_setup,
                           NUM_POINTS, seed=SEED, method=0)

a2 = np.repeat(-radius, 3)  # Sphere Start
b2 = np.repeat(radius, 3)  # Sphere End
real2 = 4/3*np.pi*radius**3  # Real 3D Sphere Volume
sphere_setup = MonteCarlo(a2, b2, step, variables=radius)
sphere = MonteCarlo.method(sphere_setup,
                           NUM_POINTS, seed=SEED, method=0)

a3 = np.repeat(-radius, 4)  # 4D Hypersphere Start
b3 = np.repeat(radius, 4)  # 4D Hypersphere End
real3 = 1/2*np.pi**2*radius**4  # Real 4D Hypersphere Volume
hypersphere_setup = MonteCarlo(a3, b3, step, variables=radius)
hypersphere = MonteCarlo.method(hypersphere_setup,
                                NUM_POINTS, seed=SEED, method=0)

a4 = np.repeat(-radius, 5)  # 5D Hypersphere Start
b4 = np.repeat(radius, 5)  # 5D Hypersphere End
real4 = 8/15*np.pi**2*radius**5  # Real 5D Hypersphere Volume
hypersphere_setup2 = MonteCarlo(a4, b4, step, variables=radius)
hypersphere2 = MonteCarlo.method(hypersphere_setup2,
                                 NUM_POINTS, seed=SEED, method=0)


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

gaussInput = MonteCarlo([-1], [1], gaussian_1d, variables=vari)
gaussOutput = MonteCarlo.method(gaussInput, NUM_POINTS, seed=SEED, method=1)

gaussInput2 = MonteCarlo([-1], [1], gaussian_1d, variables=vari2)
gaussOutput2 = MonteCarlo.method(gaussInput2, NUM_POINTS, seed=SEED, method=1)

# 6D evaluated with different x0 and sigma

MEAN3 = np.array([2, 5, 6, 9, 4, 2])
SIGMA3 = np.array([0.2, 0.4, 0.1, 0.3, 0.2, 0.5])

vari3 = np.array([MEAN3, SIGMA3])

gaussInput3 = MonteCarlo([-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1],
                          gaussian_md, variables=vari3)
gaussOutput3 = MonteCarlo.method(gaussInput3, NUM_POINTS, seed=SEED, method=1)

if rank == 0:
    print("TASK 2 - Gaussian")
    print(f"1D Gaussian with mean {MEAN} and SD {SIGMA}: {gaussOutput}")
    print(f"1D Gaussian with mean {MEAN2} and SD {SIGMA2}: {gaussOutput2}")
    print(f"6D Gaussian with mean {MEAN3} and SD {SIGMA3}: {gaussOutput3}")
    print("Real value should always be 1.0 (normalised function)")
    print("\n")

input_importance_3 = MonteCarlo([0], [1], func_to_integrate_3, variables=None)
output_importance_3 = MonteCarlo.method(input_importance_3, NUM_POINTS, seed=SEED, method=2, func2=sampling_func_3, func3=inverse_sampling_3)
real_value_importance_sampling_3 = 0.746824132812427

if rank == 0:
    print(f"With importance sampling: {output_importance_3}")
    print(f"Real value: {real_value_importance_sampling_3}")
    print("")

'''
# ------------------------------------------
# IMPORTANCE SAMPLING
# First Importance Sampling Test

importanceInput1 = MonteCarlo([-4], [0], func_to_integrate_1)
importanceOutput1 = MonteCarlo.method(importanceInput1, NUM_POINTS, seed=SEED, method=2,
                                      func2=sampling_func_1, func3=inverse_sampling_1)

unimportanceInput1 = MonteCarlo([-4], [0], func_to_integrate_1)
unimportanceOutput1 = MonteCarlo.method(unimportanceInput1, NUM_POINTS, seed=SEED, method=0)

real_value_importance_sampling_1 = 15/(16*np.log(2))

if rank == 0:
    print("IMPORTANCE SAMPLING TEST 1: 2^x from -4 to 0")
    print(f"With importance sampling: {importanceOutput1}")
    print(f"Without importance sampling: {unimportanceOutput1}")
    print(f"Real value: {real_value_importance_sampling_1}")
    print("")


# Second Importance Sampling Test
importanceInput1 = MonteCarlo([0], [2], func_to_integrate_2)
importanceOutput1 = MonteCarlo.method(importanceInput1, NUM_POINTS, seed=SEED, method=2,
                                      func2=sampling_func_2, func3=inverse_sampling_2)

unimportanceInput2 = MonteCarlo([0], [2], func_to_integrate_2)
unimportanceOutput2 = MonteCarlo.method(unimportanceInput2, NUM_POINTS, seed=SEED, method=0)

real_value_importance_sampling_2 = 0.8929535142938763

if rank == 0:
    print("IMPORTANCE SAMPLING TEST 2: exp(-x^3) from 0 to 2")
    print(f"With importance sampling: {importanceOutput1}")
    print(f"Without importance sampling: {unimportanceOutput2}")
    print(f"Real value: {real_value_importance_sampling_2}")
    print("")


# Third Importance Sampling Test
importanceInput3 = MonteCarlo([0], [1], func_to_integrate_3)
importanceOutput3 = MonteCarlo.method(importanceInput3, NUM_POINTS, seed=SEED, method=2,
                                      func2=sampling_func_3, func3=inverse_sampling_3)

unimportanceInput3 = MonteCarlo([0], [1], func_to_integrate_3)
unimportanceOutput3 = MonteCarlo.method(unimportanceInput3, NUM_POINTS, seed=SEED, method=0)

real_value_importance_sampling_3 = 0.746824132812427

if rank == 0:
    print("IMPORTANCE SAMPLING TEST 2: exp(-x^2) from 0 to 1")
    print(f"With importance sampling: {importanceOutput3}")
    print(f"Without importance sampling: {unimportanceOutput3}")
    print(f"Real value: {real_value_importance_sampling_3}")
    print("")
'''
MPI.Finalize()
