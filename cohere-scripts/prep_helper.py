import os
import re
import glob
import time

import numpy as np
from multiprocessing import Pool, Process, Queue
import cohere_core.utilities as ut
from functools import partial
import cohere_core.utilities.dvc_utils as dvut
import importlib


PREP_DATA_FILENAME = 'prep_data.tif'

def write_prep_arr(arr, save_dir, filename):
    """
    This function saves the prepared data in given directory. Creates the directory if
    it does not exist.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ut.save_tif(arr, ut.join(save_dir, filename))


def read_align(prep_obj, refarr, dir):
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
    # read
    arr = prep_obj.read_scan(dir)
    # assuming here the scan number is the last literal group in dir
    scan = int(next(re.finditer(r'\d+$', dir)).group(0))
    # align
    aligned_err = dvut.align_arrays_pixel(refarr, arr)
    [aligned, err] = aligned_err
    return [np.absolute(aligned), err, scan]


def read_scan_save(prep_obj, read_dir_write_dir):
    (read_dir, write_dir) = read_dir_write_dir
    # read scan
    arr = prep_obj.read_scan(read_dir)
    # clear seam
    arr = prep_obj.det_obj.clear_seam(arr)
    # write
    write_prep_arr(arr, write_dir, PREP_DATA_FILENAME)


def process_separate_scans(prep_obj, dirs, scans, dir):
    if len(scans) == 0:
        return
    nproc = min(len(dirs), os.cpu_count() * 2)
    poollist = [(dirs[i], ut.join(dir, f'scan_{str(scans[i])}', 'preprocessed_data')) for i in range(len(dirs))]
    func = partial(read_scan_save, prep_obj)
    with Pool(processes=nproc) as pool:
        pool.map_async(func, poollist)
        pool.close()
        pool.join()


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
        row += row[0].ljust(scan_col_width + col_gap)
        row += f'{err}{linesep}'
        report_table += row
        no -= 1

    with open(ut.join(save_dir, f'corr_err_{ref_scan}.txt'), 'w+') as f:
        f.write(report_table)
        f.flush()


def combine_scans(prep_obj, dirs, inds):
    results = []

    def collect_result(result):
        results.append(result)

    if len(dirs) == 1:
        return prep_obj.read_scan(dirs[0])
    scans_order = np.argsort(inds).tolist()
    refarr = None
    dir_no = len(dirs)
    ref_dir = ''
    while refarr is None and dir_no > 0:
        ref_dir = dirs.pop(scans_order[0])
        refarr = prep_obj.read_scan(ref_dir)
        dir_no -= 1
    if refarr is None:
        return None

    # assumming here the scan number is the last literal group in dir
    ref_array_scan = int(next(re.finditer(r'\d+$', ref_dir)).group(0))

    # start reporting process. It will get correlation error for each scan with reference
    # to the refarray. It will receive the errors via queue.
    q = Queue()
    p = Process(target=report_corr_err, args=(q, ref_array_scan, dir_no, prep_obj.experiment_dir))
    p.start()

    # It is faster to run concurrently on cpu than on gpu which needs uploading
    # array on gpu memory. Setting library here before starting multiple processes
    devlib = importlib.import_module('cohere_core.lib.nplib').nplib
    dvut.set_lib(devlib)

    nproc = min(len(dirs), os.cpu_count() * 2)

    sumarr = np.zeros_like(refarr)
    sumarr = sumarr + refarr

    func = partial(read_align, prep_obj, refarr)
    with Pool(processes=nproc) as pool:
        pool.map_async(func, dirs, callback=collect_result)
        pool.close()
        pool.join()
        pool.terminate()

    if len(results) > 0:
        for res in results[0]:
            [ar, er, scan] = res
            sumarr = sumarr + ar
            q.put((scan, er))
    else:
        print(f'did not find any scans to align with {ref_array_scan}')

    sumarr = prep_obj.det_obj.clear_seam(sumarr)
    return sumarr


class Preparer():
    def __init__(self, prep_obj):
        """
        Creates PrepData instance for beamline aps_34idc. Sets fields to configuration parameters.
        Parameters
        ----------
        experiment_dir : str
            directory where the files for the experiment processing are created
        Returns
        -------
        PrepData object
        """
        self.prep_obj = prep_obj
        self.no_scan_ranges = len(self.prep_obj.scan_ranges)
        self.unit_dirs_scan_indexes = {}


    def add_scan(self, scan_no, subdir):
        i = 0
        while scan_no > self.prep_obj.scan_ranges[i][1]:
            i += 1
            if i == self.no_scan_ranges:
                return
        if scan_no >= self.prep_obj.scan_ranges[i][0]:
            # add the scan
            if i not in self.unit_dirs_scan_indexes.keys():
                self.unit_dirs_scan_indexes[i] = [[], []]
            self.unit_dirs_scan_indexes[i][0].append(subdir)
            self.unit_dirs_scan_indexes[i][1].append(scan_no)


    def get_batches(self):
        data_dir = self.prep_obj.data_dir
        for scan_dir in os.listdir(data_dir):
            subdir = ut.join(data_dir, scan_dir)
            if os.path.isdir(subdir):
                # exclude directories with fewer tif files than min_files
                if len(glob.glob1(subdir, "*.tif")) < self.prep_obj.min_files and len(
                        glob.glob1(subdir, "*.tiff")) < self.prep_obj.min_files:
                    continue
                last_digits = re.search(r'\d+$', scan_dir)
                if last_digits is not None:
                    scan = int(last_digits.group())
                    if scan in self.prep_obj.exclude_scans:
                        continue
                    if self.prep_obj.auto_data and not self.prep_obj.separate_scans and scan in self.prep_obj.outliers_scans:
                        continue
                self.add_scan(scan, subdir)
        return list(self.unit_dirs_scan_indexes.values())


    def process_batch(self, dirs, scans, save_dir, filename):
        if len(dirs) == 1:
            arr = self.prep_obj.read_scan(dirs[0])
        else:
            arr = combine_scans(self.prep_obj, dirs, scans)
            arr = self.prep_obj.det_obj.clear_seam(arr)
        write_prep_arr(arr, save_dir, filename)


class SinglePreparer(Preparer):
    def __init__(self, prep_obj):
        """
        Creates PrepData instance for beamline aps_34idc. Sets fields to configuration parameters.
        Parameters
        ----------
        experiment_dir : str
            directory where the files for the experiment processing are created
        Returns
        -------
        PrepData object
        """
        super().__init__(prep_obj)

    def prepare(self, batches):
        all_dirs = []
        all_scans = []
        for batch in batches:
            all_dirs.extend(batch[0])
            all_scans.extend(batch[1])
        self.process_batch(all_dirs, all_scans, ut.join(self.prep_obj.experiment_dir, 'preprocessed_data'), PREP_DATA_FILENAME)


class SepPreparer(Preparer):
    def __init__(self, prep_obj):
        """
        Creates PrepData instance for beamline aps_34idc. Sets fields to configuration parameters.
        Parameters
        ----------
        experiment_dir : str
            directory where the files for the experiment processing are created
        Returns
        -------
        PrepData object
        """
        super().__init__(prep_obj)

    def prepare(self, batches):
        processes = []
        if self.prep_obj.separate_scans:
            dirs = []
            scans = []
            for batch in batches:
                dirs.extend(batch[0])
                scans.extend(batch[1])
            save_dir = self.prep_obj.experiment_dir
            process_separate_scans(self.prep_obj, dirs, scans, save_dir)
        else:
            for batch in batches:
                dirs = batch[0]
                scans = batch[1]
                indx = str(scans[0])
                if len(scans) > 1:
                    indx = f'{indx}-{str(scans[-1])}'
                save_dir = ut.join(self.prep_obj.experiment_dir, f'scan_{indx}', 'preprocessed_data')
                p = Process(target=self.process_batch,
                            args=(dirs, scans, save_dir, PREP_DATA_FILENAME))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
