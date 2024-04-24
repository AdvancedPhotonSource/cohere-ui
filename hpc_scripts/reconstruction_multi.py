#!/usr/bin/env python

import os
import argparse
import importlib
import cohere_core.utilities.utils as ut
import cohere_core.controller.phasing as calc
from mpi4py import MPI


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['set_lib',
           'reconstruction']


def set_lib(pkg):
    global devlib
    if pkg == 'cp':
        devlib = importlib.import_module('cohere_core.lib.cplib').cplib
    elif pkg == 'np':
        devlib = importlib.import_module('cohere_core.lib.nplib').nplib
    elif pkg == 'torch':
        devlib = importlib.import_module('cohere_core.lib.torchlib').torchlib
    calc.set_lib(devlib)


def reconstruction(conf_file, datafile):
    """
    Controls multiple reconstructions, the reconstructions run concurrently.

    This script is typically started with cohere_core-ui helper functions. The 'init_guess' parameter in the configuration file defines whether guesses are random, or start from some saved states. It will set the initial guesses accordingly and start phasing process, running each reconstruction in separate thread. The results will be saved in configured 'save_dir' parameter or in 'results_phasing' subdirectory if 'save_dir' is not defined.

    Parameters
    ----------
    conf_file : str
        configuration file name

    datafile : str
        data file name
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # the config_rec might be an alternate configuration with a postfix that will be included in save_dir
    filename = conf_file.split('/')[-1]
    conf_dir = conf_file[:-len(filename)]
    save_dir = conf_dir.replace('conf', 'results_phasing')
    if rank == 0:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

    comm.Barrier()

    pars = ut.read_config(conf_file)
    if 'init_guess' in pars and pars['init_guess'] == 'AI_guess':
        print('multiple reconstruction do not support AI_guess initial guess')
        return

    prev_dir = None
    if 'init_guess' in pars and pars['init_guess'] == 'continue':
        if not 'continue_dir' in pars:
            print('continue directory must be defined if initial guess is continue')
            return

        continue_dir = pars['continue_dir']
        if os.path.isdir(continue_dir):
            prev_dirs = os.listdir(continue_dir)
            if len(prev_dirs) > rank:
                prev_dir = ut.join(continue_dir, prev_dirs[rank])
                if not os.path.isfile(ut.join(prev_dir, 'image.npy')):
                    prev_dir = None

    set_lib('cp')

    worker = calc.Rec(pars, datafile)
    ret = worker.init_dev(-1)
    if ret < 0:
        return

    ret = worker.init(prev_dir)
    if ret < 0:
        return

    ret = worker.iterate()
    if ret < 0:
        return

    save_sub = ut.join(save_dir, str(rank))
    if not os.path.isdir(save_sub):
        os.mkdir(save_sub)
    worker.save_res(save_sub)

    print('done multi-reconstruction')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("conf_file", help="conf_file")
    parser.add_argument("datafile", help="datafile")
    args = parser.parse_args()

    reconstruction(args.conf_file, args.datafile)


if __name__ == "__main__":
    exit(main())

