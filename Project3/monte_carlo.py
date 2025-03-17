#!/bin/python3
"""
Module for the creation of an object oriented class for Monte Carlo integration methods

MIT License

Copyright (c) 2025 Tyler Chauvy, Adam John Rae, Laila Safavi

See LICENSE.txt for details
"""

from numpy.random import SeedSequence, default_rng
import numpy as np
from mpi4py import MPI

# MPI.Init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nworkers = comm.Get_size()


class MonteCarlo:
    """
    Class for calculating integrals through the Monte Carlo method
    """
    def __init__(self, starts, ends, func, variables=None):
        self.starts = starts
        self.ends = ends
        self.f = func
        if variables is None:
            variables = [] # variables defaults to an empty array if none are supplied
        else:
            self.variables = variables

    def __str__(self):
        """
        Assumes floating point when printing
        """
        return (f"(Integral: {self.data[0]:.4f}, Var: {self.data[1]:.4f},"
                f" Err: {self.data[2]:.4f})")

    def random(self, seed):
        """
        Establishes the random numbers for each worker
        """
        ss = SeedSequence(seed)  # getting the random numbers
        child_seeds = ss.spawn(nworkers)  # random numbers for each worker
        streams = [default_rng(s) for s in child_seeds]
        # getting the random numbers in arrays we like
        return streams
    
    def method(self, num_counts, seed, method, func2=None):
        self.num_counts = num_counts
        if method == 0:
            self.data = self.integral(seed)
        elif method == 1:
            self.data = self.infinity(seed)
        elif method == 2:
            self.data = self.integral_importance_sampling(func2, seed)
            
    def reduce_sum(self, value):
        value_message = np.array(value, dtype=np.float64)
     
        summation = np.array(0, dtype=np.float64)

        comm.Allreduce(value_message, summation)
        return summation

    def integral(self, seed):
        """
        Monte Carlo integral calculator for definite integrals

        Parameters
        ----------
        seed : Integer

        Returns
        -------
        self.data : Array
            Integral value, variance, and error.

        """
        dim = len(self.starts)
        streams = MonteCarlo.random(self, seed)
        points = streams[rank].random((int(self.num_counts/nworkers), dim))

        # Sending messages in MPI comm requires array
        sum_f = 0
        expect_f_squared = 0

        for p in points:
            count = 0
            while count < dim:
                # making sure they are in the interval we need them to be
                p[count] = p[count] * \
                    (self.ends[count]-self.starts[count])+self.starts[count]
                count = count+1
            sum_f = sum_f + (self.f(p, *self.variables))  # we get sum(f) for each worker
            expect_f_squared = expect_f_squared + \
                (self.f(p, *self.variables))**2  # we get sum(f**2) for each worker

        final_sum_f = MonteCarlo.reduce_sum(self, sum_f)
        final_f_squared = MonteCarlo.reduce_sum(self, expect_f_squared)

        prefactor1 = 1  # this will be used to create the (b-a)(d-c)...
        prefactor2 = 1/(self.num_counts)
        d = 0
        while d < dim:
            # we get our (b-a)(c-d...)
            prefactor1 = prefactor1*(self.ends[d]-self.starts[d])
            d = d+1

        final_i = prefactor1*prefactor2*final_sum_f  # our integral
        final_var = prefactor2 * \
            (final_f_squared*prefactor2-(final_sum_f*prefactor2)**2)  # our variance
        final_error = prefactor1*np.sqrt(final_var)  # our error
        self.data = np.array([final_i, final_var, final_error])

        return self.data

    def infinity(self, seed):
        """
        Monte Carlo integral calculator for infinite/improper cases

        Parameters
        ----------
        seed : Integer

        Returns
        -------
        self.data : Array
            Integral value, variance, and error.

        """
        dim = len(self.starts)
        inf_starts = -1  # gross way of doing this tbh
        inf_ends = 1
        streams = MonteCarlo.random(self, seed)
        points = streams[rank].random((int(self.num_counts/nworkers), dim))

        # Sending messages in MPI comm requires array
        sum_f = 0
        expect_f_squared = 0

        for p in points:
            count = 0
            while count < dim:
                # making sure they are in the interval we need them to be
                p[count] = p[count] * (inf_ends-inf_starts)+inf_starts
                count = count+1

            x = p/(1-p**2)  # value to be passed in to f(x)
            factor = (1+p**2)/((1-p**2)**2)  # integral factor
            sum_f = sum_f + (self.f(x, *self.variables, factor))  # we get sum(f) for each worker
            expect_f_squared = expect_f_squared + \
                self.f(x, *self.variables, factor**2)  # we get sum(f**2) for each worker

        final_sum_f = MonteCarlo.reduce_sum(self, sum_f)
        final_f_squared = MonteCarlo.reduce_sum(self, expect_f_squared)

        prefactor1 = 1  # this will be used to create the (b-a)(d-c)...
        prefactor2 = 1/(self.num_counts)
        d = 0
        while d < dim:
            # we get our (b-a)(c-d...)
            prefactor1 = prefactor1*(inf_ends-inf_starts)
            d = d+1

        final_i = prefactor1*prefactor2*final_sum_f  # our integral
        final_var = prefactor2 * \
            (final_f_squared*prefactor2-(final_sum_f*prefactor2)**2)  # our variance
        final_error = prefactor1*np.sqrt(final_var)  # our error
        self.data = np.array([final_i, final_var, final_error])

        return self.data

    def integral_importance_sampling(self, inverse_samp, seed):
        """
        Monte Carlo integral calculator that implements an importance sampling method

        Parameters
        ----------
        inverse_samp : Function
        seed : Integer

        Returns
        -------
        self.data : Array
            Integral value, variance, and error.

        """
        dim = len(self.starts)
        streams = MonteCarlo.random(self, seed)
        points = streams[rank].random((int(self.num_counts/nworkers), dim))

        # Sending messages in MPI comm requires array
        sum_f = 0
        expect_f_squared = 0

        for p in points:
            count = 0
            while count < dim:
                p[count] = inverse_samp(p[count])
                count = count+1
            sum_f = sum_f + (self.f(p, *self.variables))  # we get sum(f) for each worker
            expect_f_squared = expect_f_squared + \
                (self.f(p, *self.variables))**2  # we get sum(f**2) for each worker

        final_sum_f = MonteCarlo.reduce_sum(self, sum_f)
        final_f_squared = MonteCarlo.reduce_sum(self, expect_f_squared)

        prefactor2 = 1/(self.num_counts)
        final_i = final_sum_f*prefactor2 # our integral
        final_var = prefactor2 * \
            (final_f_squared*prefactor2-(final_sum_f*prefactor2)**2)  # our variance
        final_error = np.sqrt(final_var)  # our error
        self.data = np.array([final_i, final_var, final_error])

        return self.data
