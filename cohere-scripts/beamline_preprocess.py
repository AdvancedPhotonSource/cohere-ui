#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script reads raw data, applies correction related to instrument, and saves prepared data.
This script is written for a specific APS beamline. It reads multiple raw data files in each scan directory, applies
darkfield and whitefield correction if applicable, creates 3D stack for each scan, then alignes and combines with
other scans.
"""

__author__ = "Barbara Frosik"
__docformat__ = 'restructuredtext en'
__all__ = ['handle_prep',
           'main']

import argparse
import os
import importlib
import time
import cohere_core.utilities as ut
import cohere_core.utilities.dvc_utils as dvut
import shutil
from multiprocessing import Queue, Process, Pool
from functools import partial
from multipeak import MultPeakPreparer
from prep_helper import SepPreparer, SinglePreparer, process_separate_scans
import common as com


def set_lib(lib):
    # initialize the library to cupy if available, otherwise to numpy
    global devlib
    if lib == 'cp':
        devlib = importlib.import_module('cohere_core.lib.cplib').cplib
    else:
        devlib = importlib.import_module('cohere_core.lib.nplib').nplib
    dvut.set_lib(devlib)


def prep_data(prep_obj, **kwargs):
    """
    Creates prep_data.tif file in <experiment_dir>/preprocessed_data directory or multiple prep_data.tif in <experiment_dir>/<scan_<scan_no>>/preprocessed_data directories.
    Parameters
    ----------
    none
    Returns
    -------
    nothingcreated mp
    """
    if prep_obj.multipeak:
        preparer = MultPeakPreparer(prep_obj)
    elif prep_obj.separate_scan_ranges or prep_obj.separate_scans:
        preparer = SepPreparer(prep_obj)
    else:
        preparer = SinglePreparer(prep_obj)

    batches = preparer.get_batches()
    if len(batches) == 0:
        return 'no scans to process'
    preparer.prepare(batches)

    return ''


def get_correlation_err(experiment_dir, scans, scan):
    """
    author: Paul Frosik
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
    Author: Paul Frosik

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
        func = partial(get_correlation_err, experiment_dir, scans)
        with Pool(processes=no_processes) as pool:
            res = pool.map_async(func, scans)
            pool.close()
            pool.join()
        for r in res.get():
            err_scan.append(r)
    else:
        # otherwise run it sequentially
        for scan in scans:
            err_scan.append(get_correlation_err(experiment_dir, scans, scan))

    err = [el[0].item() for el in err_scan]
    # print('err', err)
    err_mean = mean(err)
    stdev = pstdev(err)
    # print('mean, std', mean, stdev)
    for (err_value, scan) in err_scan:
       # print(err_value, scan)
        if err_value > (err_mean + stdev):
            outlier_scans.append(scan)
    # print('outliers scans', outlier_scans)
    q.put(outlier_scans)


def find_outlier_scans(experiment_dir, prep_obj):
    """
    This function finds batches of scans with number of scans greater than 3 and follows to find outliers in those batches.
    Scans data are read and saved in scan directories.
    The function finds available resources and calls concurrent processes on each batch to find outliers.
    The outliers scans are receives through queue from each process.
    :param experiment_dir:
         path to the cohere experiment
    :param prep_obj:
        object holding parameters for beamline preprocess
    :return: list of int
        list of outliers scans
    """
    print('finding outliers')
    auto_batches = []
    dirs = []
    scans = []
    batches = prep_obj.get_batches()
    for batch in batches:
        if len(batch[0]) > 3:
            dirs += batch[0]
            scans += batch[1]
            if prep_obj.separate_scan_ranges:
                auto_batches.append(batch)

    if len(auto_batches) == 0:
        return []

    if not prep_obj.separate_scan_ranges:
        auto_batches.append([dirs, scans])

    # save scans that are in auto_batches
    process_separate_scans(prep_obj, dirs, scans, experiment_dir)

    # this code determines which library to use and how many scans can be processed concurrently
    try:
        import cupy
        lib = 'cp'
        data_size = ut.read_tif(ut.join(experiment_dir, f'scan_{str(scans[0])}', 'preprocessed_data', 'prep_data.tif')).size
        job_size = data_size * 67 / 1000000. + 84 # empirically found constants
        # for now use the first GPU
        # it will be revised after cluster availability is merged
        avail_devs_dict = ut.get_avail_gpu_runs(job_size, [0])
        avail_devs = []
        for k,v in avail_devs_dict.items():
            avail_devs.extend([k] * v)
        available_processes = len(avail_devs)
    except:
        lib = 'np'
        available_processes = os.cpu_count() * 2
    set_lib(lib)

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
            p = Process(target=find_outliers_in_batch, args=(experiment_dir, batch[1], q, no_concurrent))
            processes.append(p)
            p.start()
        i = len(processes)
        while i > 0:
            outliers.extend(q.get())
            i -= 1

        for p in processes:
            p.join()

    # remove individual scan directories
    for scan_dir in os.listdir(experiment_dir):
        if scan_dir.startswith('scan'):
            shutil.rmtree(experiment_dir + '/' + scan_dir)
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

def handle_prep(experiment_dir, **kwargs):
    """
    Reads the configuration files and accrdingly creates prep_data.tif file in <experiment_dir>/prep directory or multiple
    prep_data.tif in <experiment_dir>/<scan_<scan_no>>/prep directories.
    Parameters
    ----------
    experimnent_dir : str
        directory with experiment files
    Returns
    -------
    experimnent_dir : str
        directory with experiment files
    """
    print('pre-processing data')

    debug = 'debug' in kwargs and kwargs['debug']
    conf_list = ['config_prep', 'config_instr', 'config_mp']
    err_msg, conf_maps = com.get_config_maps(experiment_dir, conf_list, debug)
    if len(err_msg) > 0:
        return err_msg

    main_conf_map = conf_maps['config']
    if 'beamline' in main_conf_map:
        beamline = main_conf_map['beamline']
        try:
            beam_prep = importlib.import_module(f'beamlines.{beamline}.prep')
        except Exception as e:
            print(e)
            print(f'cannot import beamlines.{beamline}.prep module.')
            return f'cannot import beamlines.{beamline}.prep module.'
    else:
        print('Beamline must be configured in main configuration file')
        return 'Beamline must be configured in main configuration file'

    prep_conf_map = conf_maps['config_prep']
    data_dir = prep_conf_map['data_dir'].replace(os.sep, '/')
    if not os.path.isdir(data_dir):
        print(f'data directory {data_dir} is not a valid directory')
        return f'data directory {data_dir} is not a valid directory'

    # create BeamPrepData object defined for the configured beamline
    instr_conf_map = conf_maps['config_instr']
    conf_map = main_conf_map
    conf_map.update(prep_conf_map)
    conf_map.update(instr_conf_map)

    if 'config_mp' in conf_maps:
        conf_map.update(conf_maps['config_mp'])

    prep_obj = beam_prep.BeamPrepData()
    msg = prep_obj.initialize(experiment_dir, conf_map)
    if len(msg) > 0:
        print(msg)
        return msg

    auto_data = 'auto_data' in conf_map and conf_map['auto_data']
    if auto_data:
        # auto_separate_scans(experiment_dir, prep_obj)
        # # clear any outliers that might be set in config_prep
        prep_obj.outliers_scans = []
        # set the auto found outliers in object
        srt = time.time()
        outliers_scans = find_outlier_scans(experiment_dir, prep_obj)
        spt = time.time()
       # print('time to find outliers', spt-srt)
        if len(outliers_scans) > 0:
            prep_obj.outliers_scans = outliers_scans
            # save configuration with the auto found outliers
            prep_conf_map['outliers_scans'] = outliers_scans
        ut.write_config(prep_conf_map, ut.join(experiment_dir, 'conf', 'config_prep'))

    msg = prep_data(prep_obj, experiment_dir=experiment_dir)
    if len(msg) > 0:
        print(msg)
        return msg
    print('finished beamline preprocessing')
    return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir",
                        help="directory where the configuration files are located")
    parser.add_argument("--debug", action="store_true",
                        help="if True the vrifier has no effect on processing")
    args = parser.parse_args()
    handle_prep(args.experiment_dir, debug=args.debug)


if __name__ == "__main__":
    exit(main())
