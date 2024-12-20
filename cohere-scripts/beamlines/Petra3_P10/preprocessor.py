import os
import numpy as np
from multiprocessing import Pool, Process, Queue
import cohere_core.utilities as ut
import cohere_core.utilities.dvc_utils as dvut
from functools import partial


def report_corr_err(q, ref_scan, dir_no, save_dir):
    col_gap = 2
    scan_col_width = 10
    linesep = os.linesep
    report_table = f'correlation errors related to scan {ref_scan}{linesep}{linesep}'
    table_title = 'scan'
    table_title += table_title[0].ljust(scan_col_width + col_gap)
    table_title += f'correlation error{linesep}'
    report_table += table_title

    no = dir_no
    while no > 0:
        (scan, err) = q.get()
        row = str(scan)
        row += ''.ljust(scan_col_width + col_gap)
        row += f'{err}{linesep}'
        report_table += row
        no -= 1

    with open(ut.join(save_dir, f'corr_err_{ref_scan}.txt'), 'w+') as f:
        f.write(report_table)
        f.flush()


def read_align(get_scan_func, refarr, scan_dir):
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
    (scan, dir) = scan_dir
    # read
    arr = get_scan_func(dir)
    # align
    aligned_err = dvut.align_arrays_pixel(refarr, arr)
    [aligned, err] = aligned_err
    return [np.absolute(aligned), err, scan]


def combine_scans(get_scan_func, scans_dirs, experiment_dir):
    results = []

    def collect_result(result):
        results.append(result)

    (refscan, refdir) = scans_dirs.pop(0)
    refarr = get_scan_func(refdir)

    # It is faster to run concurrently on cpu than on gpu which needs uploading
    # array on gpu memory. Setting library here before starting multiple processes
    dvut.set_lib_from_pkg('np')

    # start reporting process. It will get correlation error for each scan with reference
    # to the refarray. It will receive the errors via queue.
    q = Queue()
    p = Process(target=report_corr_err, args=(q, refscan, len(scans_dirs), experiment_dir))
    p.start()

    nproc = min(len(scans_dirs), os.cpu_count() * 2)

    sumarr = np.zeros_like(refarr)
    sumarr = sumarr + refarr

    func = partial(read_align, get_scan_func, refarr)
    with Pool(processes=nproc) as pool:
        pool.map_async(func, scans_dirs, callback=collect_result)
        pool.close()
        pool.join()
        pool.terminate()

    if len(results) > 0:
        for res in results[0]:
            [ar, er, scan] = res
            sumarr = sumarr + ar
            q.put((scan, er))
    else:
        print(f'did not find any scans to align with {refscan}')

    return sumarr


def process_batch(get_scan_func, scans_dirs, save_file, experiment_dir):
    if len(scans_dirs) == 1:
        arr = get_scan_func(scans_dirs[0][1])
    else:
        arr = combine_scans(get_scan_func, scans_dirs, experiment_dir)
    # save the file
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # print(f"Saving array (max={int(arr.max())}) as {save_dir + '/' + filename}")
    ut.save_tif(arr, save_file)


