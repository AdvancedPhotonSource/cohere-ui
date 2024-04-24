#!/bin/bash
#PBS -q debug
#PBS -A APSDataAnalysis
#PBS -o logs/
#PBS -e logs/

RD=ROOT_DIR
module load conda/2022-07-19
conda activate

python -m venv --system-site-packages ${RD}/cohere_env
source ${RD}/cohere_env/bin/activate

module load cudatoolkit-standalone/11.8.0

export PYTHONPATH=${RD}/cohere_env/bin/python
WD=${RD}/hpc_scripts
cd ${WD}

mpiexec -np NRANKS --ppn RANKS_PER_NODE ./set_affinity_gpu.sh ${WD}/SCRIPT CONF DATA_FILE

