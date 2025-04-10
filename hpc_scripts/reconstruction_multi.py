#!/usr/bin/env python

import argparse
import cohere_core.utilities.utils as ut
from cohere_scripts.inner_scripts.reconstruction_populous import single_rec_process
from mpi4py import MPI


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['main',
           ]


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="experiment directory")
    args = parser.parse_args()

    exp_dir = args.exp_dir
    pars = ut.read_config(ut.join(exp_dir, 'conf', 'config_rec'))
    datafile = ut.join(exp_dir, 'phasing_data', 'data.tif')
    save_dir = ut.join(exp_dir, f'result_phasing_{rank}')

    single_rec_process('cp', pars, datafile, None, None, (None, save_dir), True)


if __name__ == "__main__":
    exit(main())

