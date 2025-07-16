#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

from cohere_scripts.inner_scripts.te_rec import time_evolving_rec
import time
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="directory with datafiles")
    args = parser.parse_args()
    st = time.time()
    # running on hpc
    hpc = True
    exit_code = time_evolving_rec(args.exp_dir, hpc)
    en = time.time()
    print(f'reconstruction took {en - st} seconds.')
    exit(exit_code)

# mpiexec -n 16 python hpc_scripts/te_rec.py <experiment_dir>