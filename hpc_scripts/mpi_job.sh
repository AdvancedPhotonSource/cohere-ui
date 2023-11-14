#!/bin/bash
#PBS -l select=2
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -l filesystems=home
#PBS -A <project-name>
#PBS -o logs/
#PBS -e logs/

module load conda/2022-07-19
conda activate

python -m venv --system-site-packages /eagle/projects/APSDataAnalysis/bfrosik/bcdi/cohere_env
source /eagle/projects/APSDataAnalysis/bfrosik/bcdi/cohere_env/bin/activate

module load cudatoolkit-standalone/11.8.0

export PYTHONPATH=/eagle/projects/APSDataAnalysis/bfrosik/bcdi/cohere_env/bin/python
WD=/eagle/projects/APSDataAnalysis/bfrosik/bcdi/cohere-ui/hpcscripts
cd ${WD}

mpiexec -np NRANKS --ppn RANKS_PER_NODE ./set_affinity_gpu.sh ${WD}/SCRIPT CONF DATA_FILE
