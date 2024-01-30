#!/bin/bash -l

qsub -A APSDataAnalysis -q debug -l select=$1 -l walltime=00:10:00 -l filesystems=home:eagle -l place=scatter A_mpi_job.sh