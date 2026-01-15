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
__all__ = ['get_correlation_errs_4scan',
           'find_outliers_in_batch',
           'find_outlier_scans']

import os
import importlib
import cohere_core.utilities as ut
import cohere_core.utilities.dvc_utils as dvut
import shutil
from multiprocessing import set_start_method
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from cohere_ui.api import balancer


def set_lib(pkg):
    # initialize the library to cupy if available, otherwise to numpy
    global devlib
    if pkg == 'cp':
        devlib = importlib.import_module('cohere_core.lib.cplib').cplib
    else:
        devlib = importlib.import_module('cohere_core.lib.nplib').nplib
    dvut.set_lib_from_pkg(pkg)


def save_corr_errs(save_dir, err_scan, scans):
    err_scan = devlib.to_numpy(err_scan)
    c_c_arr = np.array([e_s[1:] for e_s in err_scan])
    fig, ax = plt.subplots()
    plt.title(f'Cross-correlation of scans in {save_dir}')
    plt.imshow(c_c_arr, cmap='viridis_r')
    plt.colorbar()
    ax.set_xticks(range(len(scans)), labels=scans, rotation = 90, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(scans)), labels=scans)
    scans_range = f'{scans[0]}_{scans[-1]}'
    path = os.path.join(save_dir,f'cross_correlation_scans_{scans_range}.png')
    plt.savefig(path)


def get_correlation_errs_4scan(experiment_dir, scans, scan):
    """
    This function finds correlation errors calculated between given scan and all other scans, including itself.

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
    errs = [scan]
    refarr = devlib.from_numpy(refarr)

    for s in scans:
        datafile = ut.join(experiment_dir, f'scan_{str(s)}', 'preprocessed_data', 'prep_data.tif')
        arr = devlib.from_numpy(ut.read_tif(datafile))
        errs.append(dvut.correlation_err(refarr, arr))
    return errs


def find_corr_errs_in_batch(experiment_dir, scans, no_processes):
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
    corr_err4scans = [[] for scan in scans]
    # if multiple processes can run concurrently use this code
    if no_processes > 1:
        func = partial(get_correlation_errs_4scan, experiment_dir, scans)
        with ProcessPoolExecutor(max_workers=no_processes) as exe:
            # Maps the function with a iterable
            # result = exe.map(func, scans)

            futures = [exe.submit(get_correlation_errs_4scan, experiment_dir, scans, scan) for scan in scans]

            for future in as_completed(futures):
                result = future.result()
                print(f"Calculated product: {result}")

            corr_err4scans = [r for r in result]
    else:
        # otherwise run it sequentially
        for scan in scans:
            corr_err4scans.append(get_correlation_errs_4scan(experiment_dir, scan, scans))

    sorted(corr_err4scans)
    return corr_err4scans


def find_outliers_in_batch(err_4scans, scans):
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

    outlier_scans = []
    err = [mean(e_s[1:]) for e_s in err_4scans]
    err_mean = mean(err)
    stdev = pstdev(err)
    for (err_value, scan) in zip(err, scans):
        if err_value > (err_mean + stdev):
            outlier_scans.append(scan)
    return outlier_scans


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
    if separate_ranges:
        batches = [batch for batch in scans_datainfo]
    else:
        batches = [[s_d for batch in scans_datainfo for s_d in batch]]
    print('finding correlation errors')

    try:
        import cupy
        pkg = 'cp'
    except:
        pkg = 'np'

    set_lib(pkg)

    outliers = []
    for batch in batches:
        scans_in_batch = [s_d[0] for s_d in batch]
        corr_errs = find_corr_errs_in_batch(experiment_dir, scans_in_batch, len(batch))
        if len(batch) > 3:
            outliers.extend(find_outliers_in_batch(corr_errs, scans_in_batch))
        save_corr_errs(experiment_dir, corr_errs, scans_in_batch)

    outliers.sort()
    return outliers


def find_corr_errs(experiment_dir, scans_datainfo, separate_ranges):
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
    if separate_ranges:
        batches = [batch for batch in scans_datainfo]
    else:
        batches = [[s_d for batch in scans_datainfo for s_d in batch]]
    print('finding correlation errors')

    try:
        import cupy
        pkg = 'cp'
    except:
        pkg = 'np'
    set_lib(pkg)

    for batch in batches:
        scans_in_batch = [s_d[0] for s_d in batch]
        corr_errs = find_corr_errs_in_batch(experiment_dir, scans_in_batch, len(batch))
        save_corr_errs(experiment_dir, corr_errs, scans_in_batch)


def remove_scan_dirs(experiment_dir):
    # remove individual scan directories
    for scan_dir in os.listdir(experiment_dir):
        if scan_dir.startswith('scan'):
            shutil.rmtree(ut.join(experiment_dir, scan_dir))

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

