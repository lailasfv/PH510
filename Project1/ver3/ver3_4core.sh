#!/bin/bash

#======================================================
#
# Job script for Task 1 of PH510
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=teaching-gpu
#
# Specify project account
#SBATCH --account=teaching
#
# No. of tasks required (ntasks=1 for a single-core job)
#SBATCH --ntasks=4
#SBATCH --distribution=block:block
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=0:04:00
#
# Job name
#SBATCH --job-name=Task1PH510
#
# Output file
#SBATCH --output=slurm-%j.out
#======================================================

module purge

#Example module load command. 
#Load any modules appropriate for your program's requirements

module load fftw/gcc-8.5.0/3.3.10
module load openmpi/gcc-8.5.0/4.1.1

#======================================================
# Prologue script to record job details
# Do not change the line below
#======================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------

mpirun -np 4 ./Task1ver3.py

#======================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------
