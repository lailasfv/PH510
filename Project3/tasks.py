# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 23:22:34 2025

@author: adamr
"""
import numpy as np
from mpi4py import MPI
from monte_carlo import Monte_Carlo

# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()

def funct2_try(x):
    return 2**x

def samp2(x):
    A=1/(1-np.exp(-4))
    return(A*np.exp(x))

def func_imp_2(x):
    return (funct2_try(x)/samp2(x))

def inverse_samp2(x):
    A=1/(1-np.exp(-4))
    #A=1
    return(np.log(x/A+np.exp(-4)))
"""
funct_test = Monte_Carlo([0],[2],num_points,func_w_imp_sampling)
#funct_test = Monte_Carlo([0],[1],num_points,funct_without_imp_sampling)
integral= Monte_Carlo.integral_importance_sampling(funct_test, inverse_samp)
print(integral)
"""
num_points = int(100000)
seed2 = 12345
funct_test2 = Monte_Carlo([-4],[0],num_points,func_imp_2)
integral2= Monte_Carlo.integral_importance_sampling(funct_test2, inverse_samp2,seed2)
print(integral2)
        

def step(x, R):  # FUNCTION FOR ANY ROUND SHAPE
    return round(np.sum(np.square(x)), 5) <= (R**2)

def test(x, a, b):
    return a*x**2 + b

def gaussian1D(x, x0, sig, factor):
    return np.linalg.norm(factor/(sig*(2*np.pi)**0.5) * np.exp((-(x-x0)**2)/(2*sig**2)))

def gaussianMD(x, x0, sig, factor):
    power = len(x)/2
    return np.linalg.norm(factor/(sig*2*(2*np.pi)**power) * np.exp((-(x-x0)**2)/(2*sig**2)))


num_points = int(100000)
seed = 12345

a = np.array([3])
b = np.array([6])

vari = np.array([1, 2])

test_x_square = Monte_Carlo(a, b, num_points, test, variables=vari)
integral = Monte_Carlo.integral(test_x_square, seed)

if rank == 0:
    print(f"Evaluating integral of x^2 between {a} and {b}: {test_x_square}")

radius = np.array([1])
radius2 = np.array([3])

dimensions = 5

a = np.repeat(-radius, dimensions)
b = np.repeat(radius, dimensions)

a2 = np.repeat(-radius2, dimensions)
b2 = np.repeat(radius2, dimensions)

sphere_in = Monte_Carlo(a, b, num_points, step, variables=radius)
sphere2_in = Monte_Carlo(a2, b2, num_points, step, variables=radius2)

sphere = Monte_Carlo.integral(sphere_in, seed)
sphere2 = Monte_Carlo.integral(sphere2_in, seed)

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
gaussTest = Monte_Carlo([-5], [5], num_points, gaussian1D, variables=vari2)
gaussOutput = Monte_Carlo.infinity(gaussTest, seed)

mean2 = np.array([2, 5, 6, 9, 4, 2])
sigma2 = np.array([0.2, 0.4, 0.1, 0.3, 0.2, 0.5])

vari2b = np.array([mean2, sigma2])

gaussTest2 = Monte_Carlo([-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1],
                         num_points, gaussianMD, variables=vari2b)
gaussOutput2 = Monte_Carlo.infinity(gaussTest2, seed)

if rank == 0:
    print(gaussOutput)
    print(gaussOutput2)
    
MPI.Finalize()