#!/bin/bash -l
qsub -l select=$1,filesystems=home:eagle,walltime=0:10:00 A_mpi_job.sh