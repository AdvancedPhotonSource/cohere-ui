#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script reads raw data, applies correction related to instrument, and saves prepared data.

This script is written for a specific APS beamline. It reads multiple raw data files in each scan directory, applies darkfield and whitefield correction if applicable, creates 3D stack for each scan, then alignes and combines with other scans.
"""

__author__ = "Ross Harder"
__docformat__ = 'restructuredtext en'
__all__ = ['get_dirs'
           'fast_shift',
           'shift_to_ref_array',
           'estimate_no_proc',
           'set_prep'
           'main',
           'PrepData.__init__',
           'PrepData.read_scan',
           'PrepData.write_prep_arr',
           'PrepData.read_align',
           'PrepData.read_write',
           'PrepData.prep_data']

import argparse
import pylibconfig2 as cfg
import numpy as np
import copy
import os
import sys
import glob
import tifffile as tif
import cohere.src_py.beamlines.spec as spec
import beamlines.aps_34id.detectors as det
import cohere.src_py.utilities.utils as ut
from multiprocessing import Pool
from multiprocessing import cpu_count
import re
import psutil


def get_dirs(scans, main_map, prep_map):
    """
    Finds directories with data files.
    
    The names of the directories end with the scan number. Only the directories with a scan range and the ones covered by configuration are included.

    Parameters
    ----------
    scans : list
        list of int, the first element indication first scan, the second, last scan. allowed one element in the list.
    main_map : config object
        a configuration object containing experiment main configuration parameters
    prep_map : config object
        a configuration object containing experiment prep configuration parameters
    
    Returns
    -------
    dirs : list
        list of directories with raw data that will be included in prepared data
    scan_inxs : list
        list of scan numbers corresponding to the directories in the dirs list
    """
    try:
        min_files = prep_map.min_files
    except:
        min_files = 0
    try:
        exclude_scans = prep_map.exclude_scans
    except:
        exclude_scans = []
    try:
        data_dir = prep_map.data_dir.strip()
    except:
        print('please provide data_dir')
        return

    dirs = []
    scan_inxs = []
    for name in os.listdir(data_dir):
        subdir = os.path.join(data_dir, name)
        if os.path.isdir(subdir):
            # exclude directories with fewer tif files than min_files
            if len(glob.glob1(subdir, "*.tif")) < min_files and len(glob.glob1(subdir, "*.tiff")) < min_files:
                continue
            last_digits = re.search(r'\d+$', name)
            if last_digits is not None:
                scan = int(last_digits.group())
                if scan >= scans[0] and scan <= scans[1] and not scan in exclude_scans:
                    dirs.append(subdir)
                    scan_inxs.append(scan)
    scans_order = np.argsort(scan_inxs).tolist()
    first_index = scan_inxs.pop(scans_order[0])
    first_dir = dirs.pop(scans_order[0])
    scan_inxs.insert(0, first_index)
    dirs.insert(0, first_dir)
    return dirs, scan_inxs


# supposedly this is faster than np.roll or scipy interpolation shift.
# https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def fast_shift(arr, shifty, fill_val=0):
    """
    Shifts array by given numbers for shift in each dimension.

    Parameters
    ----------
    arr : array
        array to shift
    shifty : list
        a list of integer to shift the array in each dimension
    fill_val : float
        values to fill emptied space
    Returns
    -------
    result : array
        shifted array
    """
    dims = arr.shape
    result = np.ones_like(arr)
    result *= fill_val
    result_slices = []
    arr_slices = []
    for n in range(len(dims)):
        if shifty[n] > 0:
            result_slices.append(slice(shifty[n], dims[n]))
            arr_slices.append(slice(0, -shifty[n]))
        elif shifty[n] < 0:
            result_slices.append(slice(0, shifty[n]))
            arr_slices.append(slice(-shifty[n], dims[n]))
        else:
            result_slices.append(slice(0, dims[n]))
            arr_slices.append(slice(0, dims[n]))
    result_slices = tuple(result_slices)
    arr_slices = tuple(arr_slices)
    result[result_slices] = arr[arr_slices]
    return result


def shift_to_ref_array(fft_ref, array):
    """
    Returns an array shifted to align with ref, only single pixel resolution pass fft of ref array to save doing that a lot.

    Parameters
    ----------
    fft_ref : array
        Fourier transform of reference array
    array : array
        array to align with reference array

    Returns
    -------
    shifted_array : array
        array shifted to be aligned with reference array
    """
    # get cross correlation and pixel shift
    fft_array = np.fft.fftn(array)
    cross_correlation = np.fft.ifftn(fft_ref * np.conj(fft_array))
    corelated = np.array(cross_correlation.shape)
    amp = np.abs(cross_correlation)
    intshift = np.unravel_index(amp.argmax(), corelated)
    shifted = np.array(intshift)
    pixelshift = np.where(shifted >= corelated / 2, shifted - corelated, shifted)
    shifted_arr = fast_shift(array, pixelshift)
    del cross_correlation
    del fft_array
    return shifted_arr


def estimate_no_proc(arr_size, factor):
    """
    Estimates number of processes the prep can be run on. Determined by number of available cpus and size of array

    Parameters
    ----------
    arr_size : int
        size of array
    factor : int
        an estimate of how much memory is required to process comparing to array size

    Returns
    -------
    number of processes
    """
    ncpu = cpu_count()
    freemem = psutil.virtual_memory().available
    nmem = freemem / (factor * arr_size)
    # decide what limits, ncpu or nmem
    if nmem > ncpu:
        return ncpu
    else:
        return int(nmem)


class PrepData:
    """
    This class contains fields needed for the data preparation, parsed from configuration file. The class uses helper functions to prepare the data.

    """
    def __init__(self, experiment_dir, *args):
        """
        Creates PrepData instance. Sets fields to configuration parameters.

        Parameters
        ----------
        experiment_dir : str
            directory where the files for the experiment processing are created

        Returns
        -------
        PrepData object
        """
        # move specfile to main config since many things need it.
        # think maybe have each program load main config and it's specific one.
        try:
            main_conf_file = os.path.join(experiment_dir, *("conf", "config"))
            with open(main_conf_file, 'r') as f:
                main_conf_map = cfg.Config(f.read())
        except Exception as e:
            print('Please check the configuration file ' + main_conf_file + '. Cannot parse ' + str(e))
            return
        try:
            prep_conf_file = os.path.join(experiment_dir, *("conf", "config_prep"))
            with open(prep_conf_file, 'r') as f:
                prep_conf_map = cfg.Config(f.read())
        except Exception as e:
            print('Please check the configuration file ' + prep_conf_file + '. Cannot parse ' + str(e))
            return
        self.experiment_dir = experiment_dir

        try:
            scans = [int(s) for s in main_conf_map.scan.split('-')]
        except:
            print("scans not defined in main config")

        # use last scan in series to get info. This still works if only one scan in list.
        scan_end = scans[-1]
        det_name = None
        self.roi = None
        try:
            specfile = main_conf_map.specfile.strip()
            # parse det name and saved roi from spec
            # get_det_from_spec is already a try block.  So maybe this is not needed?
            det_name, self.roi = spec.get_det_from_spec(specfile, scan_end)
            if det_name is not None and det_name.endswith(':'):
                det_name = det_name[:-1]
        except AttributeError:
            print("specfile not configured")
        except:
            print("Detector information not in spec file")

        # default detector get_frame method just reads tif files and doesn't do anything to them.
        try:
            det_name = prep_conf_map.detector
        except:
            if det_name is None:
                print('Detector name is not available, using default detector class')
                det_name = "default"

        # The detector attributes for background/whitefield/etc need to be set to read frames
        self.detector = det.create_detector(det_name)
        if self.detector is None:
            print ('no detector class ' + det_name + ' defined')
            return
        else:
            # if anything in config file has the same name as a required detector attribute, copy it to
            # the detector
            # this will capture things like whitefield_filename, etc.
            for attr in prep_conf_map.keys():
                if hasattr(self.detector, attr):
                    setattr(self.detector, attr, prep_conf_map.get(attr))

        # if roi is set in config file use it, just in case spec had it wrong or it's not there.
        try:
            self.roi = prep_conf_map.roi
        except:
            pass

        try:
            self.separate_scans = prep_conf_map.separate_scans
        except:
            self.separate_scans = False

        try:
            self.Imult = prep_conf_map.Imult
        except:
            self.Imult = None

            # build sub-directories map
        if len(scans) == 1:
            scans.append(scans[0])

        self.dirs, self.scans = get_dirs(scans, main_conf_map, prep_conf_map)

        if len(self.dirs) == 0:
            print('no data directories found')
        else:
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)


    def read_scan(self, dir):
        """
        Reads raw data files from scan directory, applies correction, and returns 3D corrected data for a single scan directory.
        
        The correction is detector dependent. It can be darkfield and/ot whitefield correction.

        Parameters
        ----------
        dir : str
            directory to read the raw files from

        Returns
        -------
        arr : ndarray
            3D array containing corrected data for one scan.
        """
        files = []
        files_dir = {}
        for file in os.listdir(dir):
            if file.endswith('tif'):
                fnbase = file[:-4]
            elif file.endswith('tiff'):
                fnbase = file[:-4]
            else:
                continue
            last_digits = re.search(r'\d+$', fnbase)
            if last_digits is not None:
                key = int(last_digits.group())
                files_dir[key] = file

        ordered_keys = sorted(list(files_dir.keys()))

        for key in ordered_keys:
            file = files_dir[key]
            files.append(os.path.join(dir, file))

        # look at slice0 to find out shape
        n = 0
        slice0 = self.detector.get_frame(files[n], self.roi, self.Imult)
        shape = (slice0.shape[0], slice0.shape[1], len(files))
        arr = np.zeros(shape, dtype=slice0.dtype)
        arr[:, :, 0] = slice0

        for file in files[1:]:
            n = n + 1
            slice = self.detector.get_frame(file, self.roi, self.Imult)
            arr[:, :, n] = slice
        return arr


    def write_prep_arr(self, arr, index=None):
        """
        This clear the seam dependable on detector from the prepared array and saves the prepared data in <experiment_dir>/prep directory of
        experiment or <experiment_dir>/<scan_dir>/prep if writing for separate scans.
        """
        if index is None:
            prep_data_dir = os.path.join(self.experiment_dir, 'prep')
        else:
            prep_data_dir = os.path.join(self.experiment_dir, *('scan_' + str(index), 'prep'))
        data_file = os.path.join(prep_data_dir, 'prep_data.tif')
        if not os.path.exists(prep_data_dir):
            os.makedirs(prep_data_dir)
        arr = self.detector.clear_seam(arr, self.roi)
        ut.save_tif(arr, data_file)


    def read_align(self, dir):
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
        arr = self.read_scan(dir, self.detector, self.roi, self.Imult)
        # align
        return np.abs(shift_to_ref_array(self.fft_refarr, arr))


    def read_write(self, index):
        arr = self.read_scan(self.dirs[index])
        self.write_prep_arr(arr, self.scans[index])


    # Pooling the read and align since that takes a while for each array
    def prep_data(self):
        """
        Creates prep_data.tif file in <experiment_dir>/prep directory or multiple prep_data.tif in <experiment_dir>/<scan_<scan_no>>/prep directories.

        Parameters
        ----------
        none

        Returns
        -------
        nothing
        """
        if len(self.dirs) == 1:
            arr = self.read_scan(self.dirs[0])
            self.write_prep_arr(arr)
        elif self.separate_scans:
            with Pool(processes=min(len(self.dirs), cpu_count())) as pool:
                pool.map_async(self.read_write, self.scans)
                pool.close()
                pool.join()
        else:
            first_dir = self.dirs.pop(0)
            refarr = self.read_scan(first_dir)
            sumarr = np.zeros_like(refarr)
            sumarr = sumarr + refarr
            self.fft_refarr = np.fft.fftn(refarr)
            arr_size = sys.getsizeof(refarr)
            
            # https://www.edureka.co/community/1245/splitting-a-list-into-chunks-in-python
            # Need to further chunck becauase the result queue needs to hold N arrays.
            # if there are a lot of them and they are big, it runs out of ram.
            # since process takes 10-12 arrays, divide nscans/15 (to be safe) and make that many
            # chunks to pool.  Also ask between pools how much ram is avaiable and modify nproc.
            
            while (len(self.dirs) > 0):
                nproc = estimate_no_proc(arr_size, 15)
                chunklist = self.dirs[0:min(len(self.dirs), nproc)]
                poollist = [self.dirs.pop(0) for i in range (len(chunklist))]
                with Pool(processes=nproc) as pool:
                    res = pool.map_async(self.read_align, poollist)
                    pool.close()
                    pool.join()
                for arr in res.get():
                    sumarr = sumarr + arr
            self.write_prep_arr(sumarr)

        print ('done with prep')



def set_prep(experiment_dir):
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
    prep_conf = os.path.join(experiment_dir, 'conf/config_prep')
    if os.path.isfile(prep_conf):
        p = PrepData(experiment_dir)
        if len(p.dirs) == 0:
            print('no data found')
        elif p.detector is None:
            print ('detector with configured name is not defined')
        else:
            p.prep_data()
    else:
        print('missing ' + prep_conf + ' file')
    return experiment_dir


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="directory where the configuration files are located")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    set_prep(experiment_dir)


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
