#!/bin/python3
"""
Module for the creation of an object oriented class for Monte Carlo integration methods

MIT License

Copyright (c) 2025 Tyler Chauvy, Adam John Rae, Laila Safavi

See LICENSE.txt for details
"""

import math
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
        self.data = 0
        if variables is None:
            self.variables = [] # variables defaults to an empty array if none are supplied
        else:
            self.variables = variables

    def __str__(self):
        """
        Assumes floating point when printing
        """
        return (f"(Integral: {self.data[0]:.4f}, Var: {self.data[1]:.4f},"
                f" Err: {self.data[2]:.4f})")  # CHANGE THIS

    def random(self, seed):
        """
        Establishes the random numbers for each worker
        """
        ss = SeedSequence(seed)  # getting the random numbers
        child_seeds = ss.spawn(nworkers)  # random numbers for each worker
        streams = [default_rng(s) for s in child_seeds]
        # getting the random numbers in arrays we like
        return streams

    def method(self, num_counts, seed, method, func2=None, func3=None):
        """
        

        Parameters
        ----------
        num_counts : Integer
            Total number of points to be evaluated in integral
        seed : Integer
            Seed for random number generation
        method : Integer
            Decides method to integrate with.
            0 = Definite integral
            1 = Indefinite integral
            2 = Definite integral with importance sampling
        func2 : Function
            Sampling function used to do importance sampling. The default is None.
        func3 : Function
            Inverse function used to do importance sampling. The default is None.

        Returns
        -------
        Print statement of integral, variance and error 

        """
        if method == 0:
            self.data = self.integral(seed, num_counts)
        elif method == 1:
            self.data = self.infinity(seed, num_counts)
        elif method == 2:
            self.data = self.integral_importance_sampling(seed, num_counts, func2, func3)
        return (f"(Integral: {self.data[0]:.4g}, Var: {self.data[1]:.4g},"
                f" Err: {self.data[2]:.4g})")

    def reduce_sum(self, value):
        """
        

        Parameters
        ----------
        value : float64
            Value from current rank

        Returns
        -------
        summation : float64
            Summation of total value across all ranks

        """
        value_message = np.array(value, dtype=np.float64)

        summation = np.array(0, dtype=np.float64)

        comm.Allreduce(value_message, summation)
        return summation

    def integral(self, seed, num_counts):
        """
        Monte Carlo integral calculator for definite integrals

        Parameters
        ----------
        seed : Integer
            Integer used to initialise random number generator and get a
            sequence of random numbers

        num_counts : Integer
            Total number of points to evaluate the function to perform
            integration

        Returns
        -------
        self.data : Array
            Integral value, variance, and error.

        """
        dim = len(self.starts)
        streams = MonteCarlo.random(self, seed)
        points = streams[rank].random((int(num_counts/nworkers), dim))

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
            # we get sum(f) and sum(f**2) for each worker
            sum_f = math.fsum([sum_f, self.f(p, *self.variables)])
            expect_f_squared = math.fsum([expect_f_squared, (self.f(p, *self.variables))**2])

        final_sum_f = MonteCarlo.reduce_sum(self, sum_f)
        final_f_squared = MonteCarlo.reduce_sum(self, expect_f_squared)

        prefactor1 = 1  # this will be used to create the (b-a)(d-c)...
        prefactor2 = 1/(num_counts)
        d = 0
        while d < dim:
            # we get our (b-a)(c-d...)
            prefactor1 = prefactor1*(self.ends[d]-self.starts[d])
            d = d+1

        final_i = prefactor1*prefactor2*final_sum_f  # our integral
        final_var = prefactor2 * \
            (final_f_squared*prefactor2-(final_sum_f*prefactor2)**2)  # our variance
        final_error = prefactor1*np.sqrt(final_var)  # our error

        return np.array([final_i, final_var, final_error])

    def infinity(self, seed, num_counts):
        """
        Monte Carlo integral calculator for infinite/improper cases

        Parameters
        ----------
        seed : Integer
            Integer use to initialise random number generator and get a
            sequence of random numbers

        num_counts : Integer
            Total number of points to evaluate the function to perform
            integration

        Returns
        -------
        self.data : Array
            Integral value, variance, and error.

        """
        dim = len(self.starts)
        inf_starts = -1  # gross way of doing this tbh
        inf_ends = 1
        streams = MonteCarlo.random(self, seed)
        points = streams[rank].random((int(num_counts/nworkers), dim))

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
            # we get sum(f) and sum(f**2) for each worker
            sum_f = math.fsum([sum_f, self.f(x, *self.variables, factor)])
            expect_f_squared = math.fsum([expect_f_squared, \
                self.f(x, *self.variables, factor**2)])

        final_sum_f = MonteCarlo.reduce_sum(self, sum_f)
        final_f_squared = MonteCarlo.reduce_sum(self, expect_f_squared)

        prefactor1 = 1  # this will be used to create the (b-a)(d-c)...
        prefactor2 = 1/(num_counts)
        d = 0
        while d < dim:
            # we get our (b-a)(c-d...)
            prefactor1 = prefactor1*(inf_ends-inf_starts)
            d = d+1

        final_i = prefactor1*prefactor2*final_sum_f  # our integral
        final_var = prefactor2 * \
            (final_f_squared*prefactor2-(final_sum_f*prefactor2)**2)  # our variance
        final_error = prefactor1*np.sqrt(final_var)  # our error

        return np.array([final_i, final_var, final_error])

    def integral_importance_sampling(self, seed, num_counts, samp, inverse_samp):
        """
        Monte Carlo integral calculator that implements an importance sampling method

        Parameters
        ----------
        seed : Integer
            Integer use to initialise random number generator and get a
            sequence of random numbers

        num_counts : Integer
            Total number of points to evaluate the function to perform
            integration
        samp: Function
            Sampling function used to do importance sampling - the function is
            ASSUMED TO HAVE BEEN NORMALISED over the range [starts,ends]
        inverse_samp : Function
            Inverse function of the definite integral of the sampling function
            over the range [starts,x]

        Returns
        -------
        self.data : Array
            Integral value, variance, and error.

        """
        dim = len(self.starts)
        streams = MonteCarlo.random(self, seed)
        points = streams[rank].random((int(num_counts/nworkers), dim))

        # Sending messages in MPI comm requires array
        sum_f = 0
        expect_f_squared = 0

        for p in points:
            count = 0
            while count < dim:
                p[count] = inverse_samp(p[count])
                count = count+1
            # we get sum(f) and sum(f**2) for each worker
            sum_f = math.fsum([sum_f, (self.f(p, *self.variables)/samp(p))])
            expect_f_squared = math.fsum([expect_f_squared, \
                (self.f(p)/samp(p, *self.variables))**2])

        final_sum_f = MonteCarlo.reduce_sum(self, sum_f)
        final_f_squared = MonteCarlo.reduce_sum(self, expect_f_squared)

        prefactor2 = 1/(num_counts)
        final_i = final_sum_f*prefactor2 # our integral
        final_var = prefactor2 * \
            (final_f_squared*prefactor2-(final_sum_f*prefactor2)**2)  # our variance
        final_error = np.sqrt(final_var)  # our error

        return np.array([final_i, final_var, final_error])
