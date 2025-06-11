#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This file contains suite of scripts related to auto_data setting.
While preprocessing with auto option the scripts determine which scans are outliers, i.e. have
the greatest correlation error with relation to all other scans in the set.
"""

__author__ = "Paul Frosik"
__docformat__ = 'restructuredtext en'
__all__ = ['get_ref_correlation_err',
           'find_outliers_in_batch',
           'find_outlier_scans']

import os
import importlib
import cohere_core.utilities as ut
import cohere_core.utilities.dvc_utils as dvut
import shutil
from multiprocessing import Queue, Process, Pool
from functools import partial


def set_lib(pkg):
    # initialize the library to cupy if available, otherwise to numpy
    global devlib
    if pkg == 'cp':
        devlib = importlib.import_module('cohere_core.lib.cplib').cplib
    else:
        devlib = importlib.import_module('cohere_core.lib.nplib').nplib
    dvut.set_lib_from_pkg(pkg)


def get_ref_correlation_err(experiment_dir, scans, scan):
    """
    This function finds a mean of correlation errors calculated between given scan and all other scans.

    :param experiment_dir: str
        path to cohere experiment
    :param scans: list of int
        list of scans included in the batch
    :param scan: int
        scan number the mean correlation error is calculated for
    :return: float
        mean of all correlation errors
    """
    refarr = ut.read_tif(ut.join(experiment_dir, f'scan_{str(scan)}', 'preprocessed_data', 'prep_data.tif'))
    err = 0
    refarr = devlib.from_numpy(refarr)

    for s in scans:
       if s != scan:
            datafile = ut.join(experiment_dir, f'scan_{str(s)}', 'preprocessed_data', 'prep_data.tif')
            arr = devlib.from_numpy(ut.read_tif(datafile))
            e = dvut.correlation_err(refarr, arr)
            err += e
    return (err / (len(scans) - 1), scan)


def find_outliers_in_batch(experiment_dir, scans, q, no_processes):
    """
    Used by auto-data. This function is called after experiment data has been read for each scan that is part of batch, i.e. scans that are being added together to bear a data file.
    Each scan is aligned with other scans and correlation error is calculated for each pair. The errors are summed for each scan. Mertics such as average and standard deviation on the summed errors are used to find the outliers.
    The scans with summed errors exceeding standard deviation are considered outliers that are returned in a list. The outliers scans will be excluded from the data set.
    The outliers scans are added to the queue and will be consumed by calling process.

    :param experiment_dir: str
        path to the cohere experiment
    :param scans: list
        list of scans in the batch
    :param q: Queue
        a queue used to pass outliers scans calculated for this batch
    :param no_processes: int
        number processes allocated to this computing
    :return:
    """
    from statistics import mean, pstdev

    err_scan = []
    outlier_scans = []
    # if multiple processes can run concurrently use this code
    if no_processes > 1:
        func = partial(get_ref_correlation_err, experiment_dir, scans)
        with Pool(processes=no_processes) as pool:
            res = pool.map_async(func, scans)
            pool.close()
            pool.join()
        for r in res.get():
            err_scan.append(r)
    else:
        # otherwise run it sequentially
        for scan in scans:
            err_scan.append(get_ref_correlation_err(experiment_dir, scans, scan))

    err = [el[0].item() for el in err_scan]
    err_mean = mean(err)
    stdev = pstdev(err)
    # print('mean, std', mean, stdev)
    for (err_value, scan) in err_scan:
       # print(err_value, scan)
        if err_value > (err_mean + stdev):
            outlier_scans.append(scan)
    q.put(outlier_scans)


def find_outlier_scans(experiment_dir, scans_datainfo, separate_ranges):
    """
    This function finds batches of scans with number of scans greater than 3 and follows to find outliers in those batches.
    Scans data are read and saved in scan directories.
    The function finds available resources and calls concurrent processes on each batch to find outliers.
    The outliers scans are received through queue from each process.
    :param experiment_dir:
         path to the cohere experiment
    :param read_scan_func:
        function to read a scan data
    :return: list of int
        list of outliers scans
    """
    def remove_scan_dirs():
        # remove individual scan directories
        for scan_dir in os.listdir(experiment_dir):
            if scan_dir.startswith('scan'):
                shutil.rmtree(ut.join(experiment_dir, scan_dir))

    if separate_ranges:
        auto_batches = [batch for batch in scans_datainfo if len(batch) > 3]
        if len(auto_batches) == 0:
            remove_scan_dirs()
            return []
    else:
        auto_batches = [s_d for batch in scans_datainfo for s_d in batch]
        if len(auto_batches) <= 3:
            remove_scan_dirs()
            return []
        else:
            # make it a single sub-list
            auto_batches = [auto_batches]

    print('finding outliers')

    # find all (scan, directory) tuples in auto_batches
    single_scans_dinfo = [s_d for batch in auto_batches for s_d in batch]
    # process_separate_scans(read_scan_func, single_scans_dinfo, experiment_dir)
    #
    # this code determines which library to use and how many scans can be processed concurrently
    try:
        import cupy
        pkg = 'cp'
        data_size = ut.read_tif(ut.join(experiment_dir, f'scan_{str(single_scans_dinfo[0][0])}', 'preprocessed_data', 'prep_data.tif')).size
        job_size = data_size * 67 / 1000000. + 84 # empirically found constants
        # use the first GPU
        avail_devs_dict = ut.get_avail_gpu_runs(job_size, [0])
        avail_devs = []
        for k,v in avail_devs_dict.items():
            avail_devs.extend([k] * v)
        available_processes = len(avail_devs)
    except:
        pkg = 'np'
        available_processes = os.cpu_count() * 2
    set_lib(pkg)

    # the available processes will be distributed among processes for each batch, i.e. scan range
    no_concurrent = available_processes // len(auto_batches)
        # in case when number of batches is greater than available processes
        # the chunking will handle all batches

    # find outliers in each batch
    q = Queue()

    outliers = []
    chunk_size = available_processes
    while auto_batches:
        chunk, auto_batches = auto_batches[:chunk_size], auto_batches[chunk_size:]

        processes = []
        for batch in chunk:
            scans_in_batch = [s_d[0] for s_d in batch]
            p = Process(target=find_outliers_in_batch, args=(experiment_dir, scans_in_batch, q, no_concurrent))
            processes.append(p)
            p.start()
        i = len(processes)
        while i > 0:
            outliers.extend(q.get())
            i -= 1

        for p in processes:
            p.join()

    # remove individual scan directories
    remove_scan_dirs()

    outliers.sort()
    return outliers


# def auto_separate_scans(experiment_dir, prep_obj, no_auto_batches):
#     # this code determines which library to use and how many scans can be processed concurrently
#     try:
#         import cupy
#         lib = 'cp'
#         no_concurrent = 1
#     except:
#         lib = 'np'
#         # the available processes will be distributed among processes for each batch, i.e. scan range
#         no_concurrent = os.cpu_count() * 2 // no_auto_batches
#     set_lib(lib)
#
#     print('finding outliers')
#     dirs = []
#     scans = []
#     arrs = []
#     batches = prep_obj.get_batches()
#     for batch in batches:
#         dirs.extend(batch[0])
#         scans.extend(batch[1])
#     # for dir in dirs:
#     #     arr = devlib.from_numpy(prep_obj.read_scan(dir))
#     #     arrs.append(arr)
#     arr = devlib.from_numpy(prep_obj.read_scan(dirs[0]))
#     arrs.append((arr))
#     arr1 = devlib.from_numpy(prep_obj.read_scan(dirs[1]))
#     arrs.append((arr1))
#     # # save scans
#     # process_separate_scans(prep_obj, dirs, scans, experiment_dir)
#     print(scans)
#     errs = []
#     refarr = prep_obj.read_scan(dirs[0])
#     for dir in dirs[1:]:
#         arr = prep_obj.read_scan(dir)
#         errs.append(dvut.correlation_err(devlib.from_numpy(refarr), devlib.from_numpy(arr)))
#         refarr = arr
#     errs = [e.item() for e in errs]
#     for i in range(0,len(errs)):
#         print(scans[i+1], errs[i])
#     refarr = prep_obj.read_scan(dirs[3])
#     arr = prep_obj.read_scan(dirs[7])
#     print(errs)
#

