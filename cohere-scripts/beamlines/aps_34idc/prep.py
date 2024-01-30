import os
import re
import numpy as np
import glob
import beamlines.aps_34idc.detectors as det
from xrayutilities.io import spec as spec


def parse_spec(specfile, scan):
    """
    Returns detector name and detector area parsed from spec file for given scan.

    Parameters
    ----------
    specfile : str
        spec file name

    scan : int
        scan number to use to recover the saved measurements

    Returns
    -------
    dict
        dictionary of parameters; name : value
    """
    params_values = {}
    # Scan numbers start at one but the list is 0 indexed
    try:
        ss = spec.SPECFile(specfile)[scan - 1]
    except Exception as ex:
        print(str(ex))
        print('Could not parse ' + specfile)
        return params_values

    try:
        params_values['detector'] = str(ss.getheader_element('UIMDET'))
        if params_values['detector'].endswith(':'):
            params_values['detector'] = params_values['detector'][:-1]
    except Exception as ex:
        print(str(ex))

    try:
        params_values['roi'] = [int(n) for n in ss.getheader_element('UIMR5').split()]
    except Exception as ex:
        print (str(ex))

    return params_values


class BeamPrepData():
    """
    This class contains fields needed for the data preparation, parsed from spec or configuration file.
    The class uses helper functions to prepare the data.
    """

    def initialize(self, experiment_dir, conf_map):
        """
        Creates PrepData instance for beamline aps_34idc. Sets fields to configuration parameters.
        Parameters
        ----------
        conf_map : dict
            dictionary containing parameters
        Returns
        -------
        PrepData object
        """
        # set defaults
        self.detector = None
        self.roi = None
        self.auto_data = False
        self.separate_scans = False
        self.separate_scan_ranges = False
        self.multipeak = False
        self.Imult = None
        self.min_files = 0
        self.exclude_scans = []
        self.outliers_scans = []
        self.experiment_dir = experiment_dir
        self.scan_ranges = []

        last_scan = None
        if 'scan' in conf_map:
            scan_units = [u for u in conf_map['scan'].replace(' ','').split(',')]
            for u in scan_units:
                if '-' in u:
                    r = u.split('-')
                    self.scan_ranges.append([int(r[0]), int(r[1])])
                else:
                    self.scan_ranges.append([int(u), int(u)])
            last_scan = self.scan_ranges[-1][-1]
        else:
            print ('scan not defined in configuration')
            return ('scan not defined in configuration')

        if last_scan is not None and 'specfile' in conf_map:
            # parse det name and saved roi from spec
            spec_values = parse_spec(conf_map['specfile'], last_scan)
            for attr in spec_values.keys():
                setattr(self, attr, spec_values[attr])
        else:
            print ("specfile not configured")
            return("specfile not configured")

        # set members to values from configuration map
        for key, val in conf_map.items():
            setattr(self, key, val)

        conf_map['roi'] = self.roi

        self.det_obj = det.create_detector(self.detector)
        if self.det_obj is None:
            return f'cannot create detector {self.detector}'
        self.det_obj.set_detector(conf_map)
        self.data_dir = self.data_dir.replace(os.sep, '/')

        return ''


    def read_scan(self, dir, **kwargs):
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
            files.append(dir + '/' + file)

        # look at slice0 to find out shape
        n = 0
        try:
            slice0 = self.det_obj.get_frame(files[n], self.roi, self.Imult)
        except Exception as e:
            print(e)
            return None
        shape = (slice0.shape[0], slice0.shape[1], len(files))
        arr = np.zeros(shape, dtype=slice0.dtype)
        arr[:, :, 0] = slice0

        for file in files[1:]:
            n = n + 1
            slice = self.det_obj.get_frame(file, self.roi, self.Imult)
            arr[:, :, n] = slice
        return arr


    def add_scan(self, scan_no, subdir, unit_dirs_scan_indexes):
        i = 0
        while scan_no > self.scan_ranges[i][1]:
            i += 1
            if i == len(self.scan_ranges):
                return

        if scan_no >= self.scan_ranges[i][0]:
            # add the scan
            if i not in unit_dirs_scan_indexes.keys():
                unit_dirs_scan_indexes[i] = [[], []]
            unit_dirs_scan_indexes[i][0].append(subdir)
            unit_dirs_scan_indexes[i][1].append(scan_no)


    def get_batches(self):
        data_dir = self.data_dir
        unit_dirs_scan_indexes = {}
        for scan_dir in os.listdir(data_dir):
            subdir = data_dir + '/' + scan_dir
            if os.path.isdir(subdir):
                # exclude directories with fewer tif files than min_files
                if len(glob.glob1(subdir, "*.tif")) < self.min_files and len(
                        glob.glob1(subdir, "*.tiff")) < self.min_files:
                    continue
                last_digits = re.search(r'\d+$', scan_dir)
                if last_digits is not None:
                    scan = int(last_digits.group())
                    # skip scans outside ranges
                    if scan < self.scan_ranges[0][0]:
                        continue
                    if scan > self.scan_ranges[-1][-1]:
                        continue
                    if scan in self.exclude_scans:
                        continue
                    if self.auto_data and not self.separate_scans and scan in self.outliers_scans:
                        continue

                    self.add_scan(scan, subdir, unit_dirs_scan_indexes)

        return list(unit_dirs_scan_indexes.values())

