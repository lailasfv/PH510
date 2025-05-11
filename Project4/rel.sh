#!/bin/bash

#======================================================
#
# Job script for Assignment 4 of PH510 - Poisson Relaxation
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=teaching
#
# Specify project account
#SBATCH --account=teaching
#
# No. of tasks required (ntasks=1 for a single-core job)
#SBATCH --ntasks=1
#SBATCH --distribution=block:block
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=01:00:00
#
# Job name
#SBATCH --job-name=PH510Assignment4
#
# Output file
#SBATCH --output=slurm-%j.out
#======================================================

module purge

#Example module load command. 
#Load any modules appropriate for your program's requirements

module load fftw/gcc-8.5.0/3.3.10

#======================================================
# Prologue script to record job details
# Do not change the line below
#======================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------

python3 ./poisson_relaxation.py

#======================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------
