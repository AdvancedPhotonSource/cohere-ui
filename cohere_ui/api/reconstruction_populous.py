# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import argparse
import cohere_core.utilities.dvc_utils as dvut
import cohere_core.controller.phasing as calc
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
import cohere_core.utilities.utils as ut
import threading


class Devices:
    def __init__(self, devices):
        self.devices = devices
        self.index = 0

    def assign_gpu(self):
        thr = threading.current_thread()
        thr.gpu = self.devices[self.index]
        self.index = self.index + 1


def single_rec_process(pkg, pars, datafile, gen, alpha_dir, rec_attrs, hpc=False):
    """
    Performs a single reconstruction.

    :param pkg: str
        defines package used as backend
    :param pars: dict
        reconstruction parameters
    :param datafile: str
        filename containing data
    :param gen: int
        current generation
    :param alpha_dir: str
        directory where result of alpha will be placed
    :param rec_attrs: dict
        objects (previous dir, save dir) used in this reconstruction
    :return: list of two elements
        metric for the result of reconstruction, save_dir of the results
    """
    dvut.set_lib_from_pkg(pkg)
    devlib = ut.get_lib(pkg)

    prev_dir, save_dir = rec_attrs
    worker = calc.Rec(pars, datafile, pkg)
    thr = threading.current_thread()
    if hpc:
        dev = -1
    else:
        dev = thr.gpu
    if worker.init_dev(dev) < 0:
        print(f'reconstruction failed, device not initialized to {thr.gpu}')
        metric = None
    else:
        ret_code = worker.init_iter_loop(prev_dir, gen)
        if ret_code < 0:
            print('reconstruction failed, check algorithm sequence and triggers in configuration')
            metric = None
        else:
            if gen is not None and gen > 0:
                threshold = pars['ga_sw_thresholds'][gen]
                sigma = pars['ga_sw_gauss_sigmas'][gen]
                breed_mode = pars['ga_breed_modes'][gen]
                alpha = devlib.load(ut.join(alpha_dir, 'image.npy'))
                worker.ds_image = dvut.breed(breed_mode, alpha, worker.ds_image)
                worker.support = dvut.shrink_wrap(worker.ds_image, threshold, sigma)

            ret_code = worker.iterate()
            if ret_code == 0:
                worker.save_res(save_dir)
                metric = worker.get_metric()
            else:    # bad reconstruction
                print('reconstruction failed during iterations')
                metric = None

    return [metric, save_dir]


def multi_rec(pkg, save_dir, devices, no_recs, pars, datafile, prev_dirs, gen=None, alpha_dir=None, q=None):
    """
    This function controls the multiple reconstructions.

    :param pkg: str
        library acronym to use for reconstruction. Supported:
        np - to use numpy
        cp - to use cupy
    :param save_dir: str
        a directory where the subdirectories will be created to save all the results for multiple reconstructions
    :param devices: list
        list of GPUs available for this reconstructions
    :param no_recs: int
        number of reconstructions
    :param pars: dict
        parameters for reconstruction
    :param datafile: str
        name of file containing data for reconstruction
    :param prev_dirs: list
        directories that hols results of previous reconstructions if it is continuation or None(s)
    :param gen: int
        current generation
    :param alpha_dir: str
        directory of alpha reconstruction
    :param q: queue
        if provided the results will be queued
    :return:
    """
    results = []

    def collect_result(result):
        results.append(result)

    #workers = [calc.Rec(pars, datafile, pkg) for _ in range(no_recs)]
    dev_obj = Devices(devices)
    iterable = []
    save_dirs = []

    for i in range(no_recs):
        save_sub = ut.join(save_dir, str(i))
        save_dirs.append(save_sub)
        iterable.append((prev_dirs[i], save_sub))
    func = partial(single_rec_process, pkg, pars, datafile, gen, alpha_dir)
    with Pool(processes=len(devices), initializer=dev_obj.assign_gpu, initargs=()) as pool:
        pool.map_async(func, iterable, callback=collect_result)
        pool.close()
        pool.join()
        pool.terminate()

    if q is not None:
        if len(results) == 0:
            q.put('failed')
        else:
            q.put(results[0])


def reconstruction(pkg, conf_file, datafile, dir, dev):
    """

    :param pkg: str
        library acronym to use for reconstruction. Supported:
        np - to use numpy
        cp - to use cupy
    :param conf_file: str
        name of configuration file with reconstruction parameters
    :param datafile: str
        data file name
    :param dir: str
        directory of the experiment
    :param dev: list
        list of devices available for reconstruction
    :return:
    """
    pars = ut.read_config(conf_file)
    no_rec = pars['reconstructions']
    prev_dirs = [None] * no_rec
    save_dir = ut.join(dir, 'results_phasing')
    multi_rec(pkg, save_dir, dev, no_rec, pars, datafile, prev_dirs)


def main():
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("lib", help="lib")
    parser.add_argument("conf_file", help="conf_file")
    parser.add_argument("datafile", help="datafile")
    parser.add_argument('dir', help='dir')
    parser.add_argument('dev', help='dev')

    args = parser.parse_args()
    dev = ast.literal_eval(args.dev)
    reconstruction(args.lib, args.conf_file, args.datafile, args.dir, dev)


if __name__ == "__main__":
    exit(main())