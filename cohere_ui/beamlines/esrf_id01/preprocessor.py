# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import os
import numpy as np
from multiprocessing import Pool, Process, Queue
import cohere_core.utilities as ut
import cohere_core.utilities.dvc_utils as dvut


def report_corr_err(ref_scan, scans_errs, save_dir):
    col_gap = 2
    scan_col_width = 10
    linesep = os.linesep
    report_table = f'correlation errors related to scan {ref_scan}{linesep}{linesep}'
    table_title = 'scan'
    table_title += table_title[0].ljust(scan_col_width + col_gap)
    table_title += f'correlation error{linesep}'
    report_table += table_title

    for scan, err in scans_errs:
        row = str(scan)
        row += ''.ljust(scan_col_width + col_gap)
        row += f'{err}{linesep}'
        report_table += row

    with open(ut.join(save_dir, f'corr_err_{ref_scan}.txt'), 'w+') as f:
        f.write(report_table)
        f.flush()


def read_align(scan_ar, refarr):
    """
    Aligns scan with reference array and enqueues the correlation error.
    Parameters
    ----------
    prep_obj : Object
        contains attributes for data preprocess
    refarr : ndarray
        reference array
    dir : str
        directory to the raw data
    q : Queue
        queue for passing the correlation error to the reporting process
    Returns
    -------
    aligned_array : array
        aligned array
    """
    # align
    [aligned, err] = dvut.align_arrays_pixel(refarr, scan_ar)
    return [np.absolute(aligned), err]


def combine_scans(scans_data_dict, experiment_dir, first_scan):
    refarr = scans_data_dict.pop(first_scan)

    # It is faster to run concurrently on cpu than on gpu which needs uploading
    # array on gpu memory. Setting library here before starting multiple processes
    dvut.set_lib_from_pkg('np')

    sumarr = np.zeros_like(refarr)
    sumarr = sumarr + refarr
    scans_errs = []
    for scan, scan_ar in scans_data_dict.items():
        ar, er = read_align(scan_ar, refarr)
        scans_errs.append((scan, er))
        sumarr = sumarr + ar

    report_corr_err(first_scan, scans_errs, experiment_dir)

    return sumarr


def process_batch(get_scan_func, scans_nodes, experiment_dir, separate_scan_ranges):
    if separate_scan_ranges:
        indx = str(scans_nodes[0][0])
        indx = f'{indx}-{str(scans_nodes[-1][0])}'
        save_dir = ut.join(experiment_dir, f'scan_{indx}', 'preprocessed_data')
    else:
        save_dir = ut.join(experiment_dir, 'preprocessed_data')
    save_file = ut.join(save_dir, 'prep_data.tif')

    all_scans_data_dict = get_scan_func([scan_node[0] for scan_node in scans_nodes])

    if len(scans_nodes) == 1:
        print('one scan', scans_nodes[0][0])
        arr = all_scans_data_dict[scans_nodes[0][0]]
    else:
        arr = combine_scans(all_scans_data_dict, experiment_dir, scans_nodes[0][0])
    # save the file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ut.save_tif(arr, save_file)


def process_separate_scans(read_scan_func, scans_datainfo, save_dir):
    for (scan, dinfo) in scans_datainfo:
        arr = read_scan_func(dinfo)
        scan_save_dir = ut.join(save_dir, f'scan_{scan}', 'preprocessed_data')
        if not os.path.exists(scan_save_dir):
            os.makedirs(scan_save_dir)
        ut.save_tif(arr, ut.join(scan_save_dir, 'prep_data.tif'))

