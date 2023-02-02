import os
import sys
import re
import glob
import numpy as np
from multiprocessing import Pool, Process, cpu_count
import util.util as ut
import beamlines.mp_prep as mp
from functools import partial


def write_prep_arr(arr, save_dir, filename):
    """
    This function saves the prepared data in given directory. Creates the directory if
    it does not exist.
    """
    print('data array dimensions', arr.shape)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ut.save_tif(arr, save_dir + '/' + filename)


def read_align(prep_obj, refarr, dir):
    """
    Aligns scan with reference array.  Referrence array is field of this class.
    Parameters
    ----------
    dir : str
        directory to the raw data
    Returns
    -------
    aligned_array : array
        aligned array
    """
    # read
    arr = prep_obj.read_scan(dir)
    fft_refarr = np.fft.fftn(refarr)
    # align
    return np.abs(ut.shift_to_ref_array(fft_refarr, arr))


def combine_scans(prep_obj, dirs, inds):
    if len(dirs) == 1:
        return prep_obj.read_scan(dirs[0])
    scans_order = np.argsort(inds).tolist()
    ref_dir = dirs.pop(scans_order[0])
    refarr = prep_obj.read_scan(ref_dir)
    if refarr is None:
        return None
    arr_size = sys.getsizeof(refarr)
    nproc = min(len(dirs), ut.estimate_no_proc(arr_size, 15))

    sumarr = np.zeros_like(refarr)
    sumarr = sumarr + refarr

    # https://www.edureka.co/community/1245/splitting-a-list-into-chunks-in-python
    # Need to further chunck becauase the result queue needs to hold N arrays.
    # if there are a lot of them and they are big, it runs out of ram.
    # since process takes 10-12 arrays, divide nscans/15 (to be safe) and make that many
    # chunks to pool.  Also ask between pools how much ram is avaiable and modify nproc.
    while (len(dirs) > 0):
        chunklist = dirs[0:nproc]
        poollist = [dirs.pop(0) for i in range(len(chunklist))]
        func = partial(read_align, prep_obj, refarr)
        with Pool(processes=nproc) as pool:
            res = pool.map_async(func, poollist)
            pool.close()
            pool.join()
        for arr in res.get():
            sumarr = sumarr + arr
    sumarr = prep_obj.detector.clear_seam(sumarr)
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
        self.unit_dirs_scan_indexes = {i: [[], []] for i in range(self.no_scan_ranges)}

    def add_scan(self, scan_no, subdir):
        i = 0
        while scan_no > self.prep_obj.scan_ranges[i][1]:
            i += 1
            if i == self.no_scan_ranges:
                return
        if scan_no >= self.prep_obj.scan_ranges[i][0]:
            # add the scan
            self.unit_dirs_scan_indexes[i][0].append(subdir)
            self.unit_dirs_scan_indexes[i][1].append(scan_no)

    def get_batches(self):
        data_dir = self.prep_obj.data_dir
        for name in os.listdir(data_dir):
            subdir = data_dir + '/' + name
            if os.path.isdir(subdir):
                # exclude directories with fewer tif files than min_files
                if len(glob.glob1(subdir, "*.tif")) < self.prep_obj.min_files and len(
                        glob.glob1(subdir, "*.tiff")) < self.prep_obj.min_files:
                    continue
                last_digits = re.search(r'\d+$', name)
                if last_digits is not None:
                    scan = int(last_digits.group())
                    if not scan in self.prep_obj.exclude_scans:
                        self.add_scan(scan, subdir)
        return list(self.unit_dirs_scan_indexes.values())

    def process_batch(self, dirs, scans, save_dir, filename):
        batch_arr = combine_scans(self.prep_obj, dirs, scans)
        batch_arr = self.prep_obj.detector.clear_seam(batch_arr)
        write_prep_arr(batch_arr, save_dir, filename)


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
        self.process_batch(all_dirs, all_scans, self.prep_obj.experiment_dir + '/preprocessed_data', 'prep_data.tif')


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
            for batch in batches:
                dirs.extend(batch[0])
            for i in range(len(dirs)):
                save_dir = self.prep_obj.experiment_dir + '/scan_' + str(i) + '/preprocessed_data'
                self.process_batch([dirs[i]], None, save_dir, 'prep_data.tif')
        else:
            for i in range(len(batches)):
                dirs = batches[i][0]
                scans = batches[i][1]
                save_dir = self.prep_obj.experiment_dir + '/scan_' + str(i) + '/preprocessed_data'
                p = Process(target=self.process_batch,
                            args=(dirs, scans, save_dir, 'prep_data.tif'))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()


class MultPeakPreparer(Preparer):
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
        try:
            self.o_twin = mp.twin_matrix(prep_obj.hkl_in, prep_obj.hkl_out, prep_obj.twin_plane,
                                 prep_obj.sample_axis)
        except KeyError:
            self.o_twin = np.identity(3)


    def get_batches(self):
        batches = super().get_batches()
        for batch in batches:
            # figure order of the batches relative to params stored in prep_obj
            index = batch[1][0]  # first index of scan in batch
            i = 0
            while index > self.prep_obj.scan_ranges[i][-1]:
                i += 1
            batch.append(i)
        return batches


    def prepare(self, batches):
        processes = []
        for i in range(len(batches)):
            dirs = batches[i][0]
            scans = batches[i][1]
            order = batches[i][2]
            conf_scans = str(self.prep_obj.scan_ranges[order][0]) + '-' + str(self.prep_obj.scan_ranges[order][1])
            orientation = self.prep_obj.orientations[order]
            orientation = str(orientation[0]) + str(orientation[1]) + str(orientation[2])
            save_dir = self.prep_obj.experiment_dir + '/mp_' + conf_scans + '_' + orientation + '/preprocessed_data'
            p = Process(target=self.process_batch,
                        args=(dirs, scans, save_dir, 'prep_data.tif'))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


    def process_batch(self, dirs, scans, save_dir, filename):
        batch_arr = combine_scans(self.prep_obj, dirs, scans)
        batch_arr = self.prep_obj.detector.clear_seam(batch_arr)
        data = mp.rotate_peaks(self.prep_obj, batch_arr, scans, self.o_twin)
        write_prep_arr(data, save_dir, filename)


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
    if hasattr(prep_obj, 'multipeak') and hasattr(prep_obj, 'orientations'):
        preparer = MultPeakPreparer(prep_obj)
    elif prep_obj.separate_scan_ranges or prep_obj.separate_scans:
        preparer = SepPreparer(prep_obj)
    else:
        preparer = SinglePreparer(prep_obj)

    batches = preparer.get_batches()
    preparer.prepare(batches)
