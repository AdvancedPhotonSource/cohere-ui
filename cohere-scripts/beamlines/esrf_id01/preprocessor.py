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
        row += row[0].ljust(scan_col_width + col_gap)
        row += f'{err}{linesep}'
        report_table += row

    with open(ut.join(save_dir, f'corr_err_{ref_scan}.txt'), 'w+') as f:
        f.write(report_table)
        f.flush()


def read_align(get_scan_func, refarr, scan_node):
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
    (scan, node) = scan_node
    # read
    arr = get_scan_func(node)
    # align
    aligned_err = dvut.align_arrays_pixel(refarr, arr)
    [aligned, err] = aligned_err
    return [np.absolute(aligned), err, scan]


def combine_scans(get_scan_func, scans_nodes, experiment_dir):
    (refscan, refnode) = scans_nodes.pop(0)
    refarr = get_scan_func(refnode)

    # It is faster to run concurrently on cpu than on gpu which needs uploading
    # array on gpu memory. Setting library here before starting multiple processes
    dvut.set_lib_from_pkg('np')

    sumarr = np.zeros_like(refarr)
    sumarr = sumarr + refarr
    scans_errs = []
    for scan_node in scans_nodes:
        ar, er, scan = read_align(get_scan_func, refarr, scan_node)
        scans_errs.append((scan, er))
        sumarr = sumarr + ar

    report_corr_err(refscan, scans_errs, experiment_dir)

    return sumarr


def process_batch(get_scan_func, scans_nodes, save_file, experiment_dir):
    if len(scans_nodes) == 1:
        arr = get_scan_func(scans_nodes[0][1])
    else:
        arr = combine_scans(get_scan_func, scans_nodes, experiment_dir)
    # save the file
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ut.save_tif(arr, save_file)

