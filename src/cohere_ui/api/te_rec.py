#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

#from cohere_ui.api.te_rec import time_evolving_rec
import os
import argparse
import cohere_core.utilities.utils as ut
from cohere_core.controller.phasing import TeRec
from mpi4py import MPI
import time
import ast


def time_evolving_rec(exp_dir, devices):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # print('in rec, rank {}'.format(rank))

    # The directories with phasing data are ordered by scan number. They have to be associated with ranks
    # in that order. This scheme of saving the ordered data files and reading the file in each rank and
    # getting the file indexed by rank is managing that order.
    with open(ut.join(exp_dir, 'phasing_dirs'), 'r') as f:
        dfiles = f.readline()
    dfiles = ast.literal_eval(dfiles)

    conf = ut.join(exp_dir, 'conf', 'config_rec')
    params = ut.read_config(conf)
    params['weight'] = 0.1

    scandir = dfiles[rank]
    datafile = ut.join(scandir, 'phasing_data', 'data.npy')
#    print('datafile', datafile)
    worker = TeRec(params, datafile, 'cp', comm)

    # sort devices, so the neighbours are in most cases on the same device
    devices.sort()
    ret_code = worker.init_dev(devices[rank])

    if ret_code < 0:
        print ('init_dev failed, check algorithm sequence and triggers in configuration', rank)
        return ret_code

    worker.exchange_data_info()
    print('rank, is full data', rank, worker.is_full_data)

    ret_code = worker.init_iter_loop()
    if ret_code < 0:
        print ('init_iter_loop failed, check algorithm sequence and triggers in configuration')
        return ret_code

    ret_code = worker.iterate()
    if ret_code < 0:
        print ('reconstruction failed during iterations')
        return ret_code

    if 'save_dir' in params:
        save_dir = params['save_dir']
    else:
        save_dir, filename = os.path.split(datafile)
        save_dir = save_dir.replace('phasing_data', 'results_phasing')
    worker.save_res(save_dir)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="directory with datafiles")
    parser.add_argument("devices", help="list of devices to be used")
    args = parser.parse_args()
    st = time.time()
    # running on hpc
    hpc = True
    exit_code = time_evolving_rec(args.exp_dir, ast.literal_eval(args.devices))
    en = time.time()
    print(f'reconstruction took {en - st} seconds.')
    exit(exit_code)

# mpiexec --oversubscribe -np 18 python cohere-ui/src/cohere_ui/api/te_rec.py ~/cohere-test/te_2825-2876/ '[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1]'