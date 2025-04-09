#!/bin/bash
#PBS -l select=2
#PBS -l walltime=00:20:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A APSDataAnalysis
#PBS -o logs/
#PBS -e logs/

ROOT_DIR=/eagle/projects/APSDataAnalysis/bfrosik/bragg/
RD=/eagle/projects/APSDataAnalysis/bfrosik/bragg/
module use /soft/modulefiles
module load conda/2024-04-29
conda activate

python -m venv --system-site-packages ${RD}cohere_env
source ${RD}/cohere_env/bin/activate

module load cudatoolkit-standalone/11.8.0

export PYTHONPATH=${RD}/cohere_env/bin/python
WD=${RD}cohere-ui/hpc_scripts
#cd ${WD}

ED=/eagle/projects/APSDataAnalysis/bfrosik/bragg/cohere-ui/example_workspace/ev_227-272/

mpiexec -np 16 --ppn 8 ${WD}/set_affinity_gpu.sh ${WD}/te_rec.py ${ED}
