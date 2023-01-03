import os
import sys
import re
import glob
import numpy as np
from multiprocessing import Pool, Process, cpu_count
import util.util as ut


def get_dirs(prep_obj, **kwargs):
    """
    Finds directories with data files.
    The names of the directories end with the scan number. Only the directories with a scan range and the ones covered by configuration are included.
    Parameters
    ----------
    prep_map : config object
        a configuration object containing experiment prep configuration parameters
    Returns
    -------
    dirs : list
        list of directories with raw data that will be included in prepared data
    scan_inxs : list
        list of scan numbers corresponding to the directories in the dirs list
    """
    no_scan_ranges = len(prep_obj.scan_ranges)
    unit_dirs_scan_indexes = {i: ([], []) for i in range(no_scan_ranges)}

    def add_scan(scan_no, subdir):
        i = 0
        while scan_no > prep_obj.scan_ranges[i][1]:
            i += 1
            if i == no_scan_ranges:
                return
        if scan_no >= prep_obj.scan_ranges[i][0]:
            # add the scan
            unit_dirs_scan_indexes[i][0].append(subdir)
            unit_dirs_scan_indexes[i][1].append(scan_no)

    try:
        data_dir = kwargs['data_dir'].replace(os.sep, '/')
    except:
        print('please provide data_dir in configuration file')
        return None, None

    def order_lists(dirs, inds):
        # The directory with the smallest index is placed as first, so all data files will
        # be alligned to the data file in this directory
        scans_order = np.argsort(inds).tolist()
        first_index = inds.pop(scans_order[0])
        first_dir = dirs.pop(scans_order[0])
        inds.insert(0, first_index)
        dirs.insert(0, first_dir)
        return dirs, inds

    for name in os.listdir(data_dir):
        subdir = data_dir + '/' + name
        if os.path.isdir(subdir):
            # exclude directories with fewer tif files than min_files
            if len(glob.glob1(subdir, "*.tif")) < prep_obj.min_files and len(
                    glob.glob1(subdir, "*.tiff")) < prep_obj.min_files:
                continue
            last_digits = re.search(r'\d+$', name)
            if last_digits is not None:
                scan = int(last_digits.group())
                if not scan in prep_obj.exclude_scans:
                    add_scan(scan, subdir)

    if prep_obj.separate_scan_ranges:
        for i in range(no_scan_ranges):
            if len(unit_dirs_scan_indexes[i]) > 1:
                unit_dirs_scan_indexes[i] = order_lists(unit_dirs_scan_indexes[i][0], unit_dirs_scan_indexes[i][1])
    else:
        # combine all scans
        dirs = [unit_dirs_scan_indexes[i][0] for i in range(no_scan_ranges)]
        inds = [unit_dirs_scan_indexes[i][1] for i in range(no_scan_ranges)]
        unit_dirs_scan_indexes = (sum(dirs, []), sum(inds, []))
        unit_dirs_scan_indexes = order_lists(unit_dirs_scan_indexes[0], unit_dirs_scan_indexes[1])
    return unit_dirs_scan_indexes


def prep_data(prep_obj, dirs_indexes, **kwargs):
    """
    Creates prep_data.tif file in <experiment_dir>/preprocessed_data directory or multiple prep_data.tif in <experiment_dir>/<scan_<scan_no>>/preprocessed_data directories.
    Parameters
    ----------
    none
    Returns
    -------
    nothing
    """

    def combine_scans(refarr, dirs, nproc, scan=''):
        sumarr = np.zeros_like(refarr)
        sumarr = sumarr + refarr
        prep_obj.fft_refarr = np.fft.fftn(refarr)

        # https://www.edureka.co/community/1245/splitting-a-list-into-chunks-in-python
        # Need to further chunck becauase the result queue needs to hold N arrays.
        # if there are a lot of them and they are big, it runs out of ram.
        # since process takes 10-12 arrays, divide nscans/15 (to be safe) and make that many
        # chunks to pool.  Also ask between pools how much ram is avaiable and modify nproc.
        while (len(dirs) > 0):
            chunklist = dirs[0:min(len(dirs), nproc)]
            poollist = [dirs.pop(0) for i in range(len(chunklist))]
            with Pool(processes=nproc) as pool:
                res = pool.map_async(prep_obj.read_align, poollist)
                pool.close()
                pool.join()
            for arr in res.get():
                sumarr = sumarr + arr
        prep_obj.write_prep_arr(sumarr, scan)

    if prep_obj.separate_scan_ranges:
        single = []
        pops = []
        for key in dirs_indexes:
            (dirs, inds) = dirs_indexes[key]
            if len(dirs) == 0:
                pops.append(key)
            elif len(dirs) == 1:
                single.append((dirs_indexes[key][0][0], str(dirs_indexes[key][1][0])))
                pops.append(key)
        for key in pops:
            dirs_indexes.pop(key)
        # process first the single scans
        if len(single) > 0:
            with Pool(processes=min(len(single), cpu_count())) as pool:
                pool.starmap_async(prep_obj.read_write, single)
                pool.close()
                pool.join()
        # then process scan ranges
        if len(dirs_indexes) == 0:
            return
        pr = []
        for dir_ind in dirs_indexes.values():
            (dirs, inds) = dir_ind
            first_dir = dirs.pop(0)
            refarr = prep_obj.read_scan(first_dir)
            if refarr is None:
                continue
            # estimate number of available cpus for each process
            arr_size = sys.getsizeof(refarr)
            nproc = int(ut.estimate_no_proc(arr_size, 15) / len(dirs_indexes))
            p = Process(target=combine_scans,
                        args=(refarr, dirs, max(1, nproc), str(inds[0]) + '-' + str(inds[-1])))
            p.start()
            pr.append(p)
        for p in pr:
            p.join()
    else:
        dirs, indexes = dirs_indexes[0], dirs_indexes[1]
        # dir_indexes consists of list of directories and corresponding list of indexes
        if len(dirs) == 1:
            arr = prep_obj.read_scan(dirs[0])
            if arr is not None:
                prep_obj.write_prep_arr(arr)
            return

        if prep_obj.separate_scans:
            iterable = list(zip(dirs, [str(ix) for ix in indexes]))
            with Pool(processes=min(len(dirs_indexes), cpu_count())) as pool:
                pool.starmap_async(prep_obj.read_write, iterable)
                pool.close()
                pool.join()
        else:
            first_dir = dirs.pop(0)
            refarr = prep_obj.read_scan(first_dir)
            if refarr is None:
                return
            arr_size = sys.getsizeof(refarr)
            nproc = ut.estimate_no_proc(arr_size, 15)
            combine_scans(refarr, dirs, nproc)
