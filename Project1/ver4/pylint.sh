#!/bin/bash

#======================================================
#
# Job script to calculate pylint score
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
#SBATCH --ntasks=1
#SBATCH --distribution=block:block
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=0:04:00
#
# Job name
#SBATCH --job-name=PH510pylint
#
# Output file
#SBATCH --output=slurm-%j.out
#======================================================

module purge

#Example module load command. 
#Load any modules appropriate for your program's requirements

module add miniconda/3.12.8

#======================================================
# Prologue script to record job details
# Do not change the line below
#======================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------

pylint --extension-pkg-whitelist=mpi4py.MPI pi_final.py

#======================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------
