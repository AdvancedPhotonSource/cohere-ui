from cohere.src_py.beamlines.preparer import PrepData
import cohere.src_py.beamlines.spec as spec
import cohere.src_py.utilities.utils as ut
import os
import re
import glob
import numpy as np


class BeamPrepData(PrepData):
    """
    This class contains fields needed for the data preparation, parsed from spec or configuration file.
    The class uses helper functions to prepare the data.
    """

    def __init__(self, experiment_dir, main_conf_map, prep_conf_map, **args):
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
        self.experiment_dir = experiment_dir

        self.det_name = None
        self.roi = None
        try:
            self.scan_range = [int(s) for s in main_conf_map.scan.split('-')]
            # single scan or multiple scans will be given as range
            if len(self.scan_range) == 1:
                self.scan_range.append(self.scan_range[0])
            scan_end = self.scan_range[-1]
        except:
            print("scans not defined in main config")
            self.scan_range = None

        if self.scan_range is not None:
            try:
                specfile = main_conf_map.specfile.strip()
                # parse det name and saved roi from spec
                self.det_name, self.roi = spec.get_det_from_spec(specfile, scan_end)
                if self.det_name is not None and self.det_name.endswith(':'):
                    self.det_name = self.det_name[:-1]
            except AttributeError:
                print("specfile not configured")
            except:
                print("exception parsing spec file")

        # detector name from configuration will override the one paesed from spec file
        try:
            self.det_name = prep_conf_map.detector
        except:
            if self.det_name is None:
                # default detector get_frame method just reads tif files and doesn't do anything to them.
                print('Detector name is not available, using default detector class')
                self.det_name = "default"

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

        try:
            self.min_files = self.prep_map.min_files
        except:
            self.min_files = 0
        try:
            self.exclude_scans = self.prep_map.exclude_scans
        except:
            self.exclude_scans = []


    def get_dirs(self, **args):
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
        try:
            data_dir = args['data_dir']
        except:
            print('please provide data_dir in configuration file')
            return None, None

        dirs = []
        scan_inxs = []
        for name in os.listdir(data_dir):
            subdir = os.path.join(data_dir, name)
            if os.path.isdir(subdir):
                # exclude directories with fewer tif files than min_files
                if len(glob.glob1(subdir, "*.tif")) < self.min_files and len(glob.glob1(subdir, "*.tiff")) < self.min_files:
                    continue
                last_digits = re.search(r'\d+$', name)
                if last_digits is not None:
                    scan = int(last_digits.group())
                    if scan >= self.scan_range[0] and scan <= self.scan_range[1] and not scan in self.exclude_scans:
                        dirs.append(subdir)
                        scan_inxs.append(scan)
        # The directory with the smallest index is placed as first, so all data files will
        # be alligned to the data file in this directory
        scans_order = np.argsort(scan_inxs).tolist()
        first_index = scan_inxs.pop(scans_order[0])
        first_dir = dirs.pop(scans_order[0])
        scan_inxs.insert(0, first_index)
        dirs.insert(0, first_dir)
        return dirs, scan_inxs


    def read_scan(self, dir, **args):
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
        try:
            slice0 = self.detector.get_frame(files[n], self.roi, self.Imult)
        except Exception as e:
            print(e)
            return None
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


    def get_detector_name(self):
        return self.det_name


    def set_detector(self, det_obj, prep_conf_map):
        # The detector attributes for background/whitefield/etc need to be set to read frames
        self.detector = det_obj

        # if anything in config file has the same name as a required detector attribute, copy it to
        # the detector
        # this will capture things like whitefield_filename, etc.
        for attr in prep_conf_map.keys():
            if hasattr(self.detector, attr):
                setattr(self.detector, attr, prep_conf_map.get(attr))
