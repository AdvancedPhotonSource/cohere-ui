# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import numpy as np
import os
import re
import cohere_core.utilities as ut
from abc import ABC, abstractmethod

class Detector(ABC):
    """
    Abstract class representing detector.
    """

    def __init__(self, name):
        self.name = name


    def dirs4scans(self, scans):
        """
        Finds directories with data that correspond to given scans or scan ranges.

        :param scans : list
            list of sub-lists defining scan ranges, ordered. For single scan a range has the same scan as beginning and end.
            one scan example:
            scans : [[2834, 2834]]
            returns : [[(2834, f'{path}/data_S2834)]]

            separate ranges example:
            scans: [[2825, 2831], [2834, 2834], [2840, 2846]]
            returns: [[(2825, f'{path}/data_S2825'), (2828, f'{path}/data_S2828'), (2831, f'{path}/data_S2831')],
             [(2834, f'{path}/data_S2834)],
             [(2840, f'{path}/data_S2840'), (2843, f'{path}/data_S2843'), (2846, f'{path}/data_S2846')]]

        :return:
        list of sub-lists, each sublist containing tuples with the input scans and corresponding data directories
         within scan ranges.
        """
        # create empty results list that allocates a sub-list for each scan range
        scans_dirs_ranges = [[] for _ in range(len(scans))]
        sr_idx = 0
        scan_range = scans[sr_idx]
        scans_dirs = scans_dirs_ranges[sr_idx]

        # check for directories
        for scandir in sorted(os.listdir(self.data_dir)):
            scandir_full = ut.join(self.data_dir, scandir)
            if os.path.isdir(scandir_full):
                last_digits = re.search(r'\d+$', scandir)
                if last_digits is not None:
                    scan = int(last_digits.group())
                else:
                    continue
                if scan < scan_range[0]:
                    continue
                elif scan <= scan_range[-1]:
                    # scan within range
                    # before adding scan check if there is enough data files
                    if len(os.listdir(scandir_full)) >= self.min_frames:
                        scans_dirs.append((scan, scandir_full))
                    if scan == scan_range[-1]:
                        sr_idx += 1
                        if sr_idx > len(scans) - 1:
                            break
                        scan_range = scans[sr_idx]
                        scans_dirs = scans_dirs_ranges[sr_idx]
 
                elif scan > scan_range[-1]:
                    sr_idx += 1
                    if sr_idx > len(scans) - 1:
                        break
                    scan_range = scans[sr_idx]
                    scans_dirs = scans_dirs_ranges[sr_idx]

        # remove empty sub-lists
        scans_dirs_ranges = [e for e in scans_dirs_ranges if len(e) > 0]
        return scans_dirs_ranges


    def get_scan_array(self, scan_info):
        """
        Reads/loads raw data file and applies correction.

        Reads raw data from a directory. The directory name is scan_info. The raw data is in form of 2D
        frames. The frames are read, corrected and stocked into 3D data

        :param scan_info: directory where the detector to retrieve data for a scan
        :return: corrected data array
        """
        slices_files = {}
        for file_name in os.listdir(scan_info):
            if file_name.endswith('tif'):
                fnbase = file_name[:-4]
            else:
                continue
            # for aps_34idc the file names end with the slice number, followed by 'tif' extension
            last_digits = re.search(r'\d+$', fnbase)
            if last_digits is not None:
                key = int(last_digits.group())
                slices_files[key] = ut.join(scan_info, file_name)

        ordered_keys = sorted(list(slices_files.keys()))
        ordered_slices = [self.correct_frame(slices_files[k]) for k in ordered_keys]

        return np.stack(ordered_slices, axis=-1)


    @abstractmethod
    def check_mandatory_params(self, params):
        """
        checks if all mandatory parameters are in params.

        :params: parameters needed to create detector
        :return: message indicating problem or empty message if all is ok
        """


    @abstractmethod
    def correct_frame(self, frame):
        """
        Applies the correction for detector.

        :param frame: 2D raw data file representing a frame
        :return: corrected frame
        """


class Detector_34idcTIM1(Detector):
    """
    Subclass of Detector. Encapsulates "34idcTIM1" detector.
    """
    name = "34idcTIM1"
    dims = (256, 256)
    roi = (0, 256, 0, 256)
    pixel = (55.0e-6, 55e-6)
    pixelorientation = ('x+', 'y-')  # in xrayutilities notation
    darkfield = None
    data_dir = None
    min_frames = None  # defines minimum frame scans in scan directory
    Imult = 1.0

    def __init__(self, params):
        super(Detector_34idcTIM1, self).__init__(self.name)
        # The detector attributes for background/whitefield/etc need to be set to read frames
        # this will capture things like data directory, darkfield_filename, etc.
        self.data_dir = params.get('data_dir') # mandatory
        if 'roi' in params:
            self.roi = params.get('roi')
        if 'darkfield_filename' in params:
            self.darkfield = ut.read_tif(params.get('darkfield_filename'))
        self.min_frames = params.get('min_frames', 0)


    # TIM1 only needs bad pixels deleted.  Even that is optional.
    def correct_frame(self, filename):
        """
        Reads raw frame from a file, and applies correction for 34idcTIM1 detector, i.e. darkfield.
        Parameters
        ----------
        filename : str
            slice data file name
        Returns
        -------
        frame : ndarray
            frame after correction
        """
        roislice1 = slice(self.roi[0], self.roi[0] + self.roi[1])
        roislice2 = slice(self.roi[2], self.roi[2] + self.roi[3])

        frame = ut.read_tif(filename)

        if self.darkfield is not None:
            frame = np.where(self.darkfield[roislice1, roislice2] > 1, 0.0, frame)

        return frame


    @staticmethod
    def check_mandatory_params(params):
        """
        For the 34idcTIM1 detector the data directory is mandatory. The darkfield file is optional.

        :return: message indicating problem or empty message if all is ok
        """
        if  'data_dir' not in params:
            msg = 'data_dir parameter not configured, mandatory for 34idcTIM1 detector.'
            raise ValueError(msg)
        data_dir = params['data_dir']
        if not os.path.isdir(data_dir):
            msg = f'data_dir directory{data_dir} does not exist.'
            raise ValueError(msg)


class Detector_34idcTIM2(Detector):
    """
    Subclass of Detector. Encapsulates "34idcTIM2" detector.
    """
    name = "34idcTIM2"
    dims = (512, 512)
    roi = (0, 512, 0, 512)
    pixel = (55.0e-6, 55e-6)
    pixelorientation = ('x+', 'y-')  # in xrayutilities notation
    whitefield = None
    darkfield = None
    raw_frame = None
    min_frames = None  # defines minimum frame scans in scan directory
    Imult = None

    def __init__(self, params):
        super(Detector_34idcTIM2, self).__init__(self.name)
        # The detector attributes for background/whitefield/etc need to be set to read frames
        # this will capture things like data directory, whitefield_filename, etc.
        # keep parameters that are relevant to the detector
        self.data_dir = params.get('data_dir')
        if 'roi' in params:
            self.roi = params.get('roi')
        if 'whitefield_filename' in params:
            self.whitefield = ut.read_tif(params.get('whitefield_filename'))
            # the code below is specific to TIM2 detector, excluding the correction of the weird pixels
            self.whitefield[255:257, 0:255] = 0  # weird pixels on edge of seam (TL/TR). Kill in WF kills in returned frame as well.
            self.wfavg = np.average(self.whitefield)
            self.wfstd = np.std(self.whitefield)
            self.whitefield = np.where(self.whitefield < self.wfavg - 3 * self.wfstd, 0, self.whitefield)
            self.Imult = params.get('Imult', self.wfavg)
        if 'darkfield_filename' in params:
            self.darkfield = ut.read_tif(params.get('darkfield_filename'))
            if self.whitefield is not None:
                self.whitefield = np.where(self.darkfield > 1, 0, self.whitefield)  # kill known bad pixel

        self.min_frames = params.get('min_frames', 0)

    def correct_frame(self, filename):
        """
        Reads raw frame from a file, and applies correction for 34idcTIM2 detector, i.e. darkfield, whitefield,
        and seam.

        Parameters
        ----------
        filename : str
            data file name
        Returns
        -------
        frame : ndarray
            frame after correction
        """
        # roi is start,size,start,size
        # will be in imageJ coords, so might need to transpose,or just switch x-y
        # divide whitefield
        # blank out pixels identified in darkfield
        # insert 4 cols 5 rows if roi crosses asic boundary
        roislice1 = slice(self.roi[0], self.roi[0] + self.roi[1])
        roislice2 = slice(self.roi[2], self.roi[2] + self.roi[3])

        frame = ut.read_tif(filename)
        if self.whitefield is not None:
            frame = frame / self.whitefield[roislice1, roislice2] * self.Imult
        if self.darkfield is not None:
            frame = np.where(self.darkfield[roislice1, roislice2] > 1, 0.0, frame)

        frame = np.where(np.isfinite(frame), frame, 0)
        frame, seam_added = self.insert_seam(frame)
        frame = np.where(np.isnan(frame), 0, frame)

        if seam_added:
            frame = self.clear_seam(frame)
        return frame

    # frame here can also be a 3D array.
    def insert_seam(self, arr):
        """
        Inserts rows/columns correction in a frame for 34idcTIM2 detector.
        Parameters
        ----------
        arr : ndarray
            raw frame
        Returns
        -------
        frame : ndarray
            frame after insering rows/columns
        """
        # Need to break this out.  When aligning multi scans the insert will mess up the aligns
        # or maybe we just need to re-blank the seams after the aligns?
        # I can't decide if the seams are a detriment to the alignment.  might need to try some.
        s1range = range(self.roi[0], self.roi[0] + self.roi[1])
        s2range = range(self.roi[2], self.roi[2] + self.roi[3])
        dims = arr.shape
        seam_added = False

        # get the col that start at det col 256 in the roi
        try:
            i1 = s1range.index(256)  # if not in range try will except
            if i1 != 0:
                frame = np.insert(arr, i1, np.zeros((4, dims[0])), axis=0)
                seam_added = True
            # frame=np.insert(normframe, i1, np.zeros((5,dims[0])),axis=0)
            else:
                frame = arr
        except:
            frame = arr  # if there's no insert on dim1 need to copy to frame

        try:
            i2 = s2range.index(256)
            if i2 != 0:
                frame = np.insert(frame, i2, np.zeros((5, dims[0] + 4)), axis=1)
                seam_added = True
        except:
            # if there's no insert on dim2 thre's nothing to do
            pass

        return frame, seam_added

    # This is needed if the seam has already been inserted and shifts have moved intensity
    # into the seam.  Found that alignment of data sets was best done with the seam inserted.
    def clear_seam(self, arr):
        """
        Removes rows/columns correction from a frame for 34idcTIM2 detector.
        Parameters
        ----------
        arr : ndarray
            frame to remove seam
        roi : list
            detector area used to take image. If None the entire detector area will be used.
        Returns
        -------
        arr : ndarray
            frame after removing rows/columns
        """
        # modify the slices if 256 is in roi
        s1range = range(self.roi[0], self.roi[0] + self.roi[1])
        s2range = range(self.roi[2], self.roi[2] + self.roi[3])
        try:
            i1 = s1range.index(256)  # if not in range try will except
            if i1 != 0:
                s1range[0] = slice(i1, i1 + 4)
                arr[tuple(s1range)] = 0
        except:
            pass
        try:
            i2 = s2range.index(256)
            if i2 != 0:
                s2range[1] = slice(i2, i2 + 5)
                arr[tuple(s2range)] = 0
        except:
            pass

        return arr

    @staticmethod
    def check_mandatory_params(params):
        """
        For the 34idcTIM2 detector the data directory, whitefiled_filename, darkfield_ilename
        are mandatory parameters.

        :params: parameters needed to create detector
        :return: message indicating problem or empty message if all is ok
        """
        if  'data_dir' not in params:
            msg = 'data_dir parameter not configured, mandatory for 34idcTIM2 detector.'
            raise ValueError(msg)
        data_dir = params['data_dir']
        if not os.path.isdir(data_dir):
            msg = f'data_dir directory{data_dir} does not exist.'
            raise ValueError(msg)

        if 'whitefield_filename' not in params:
            msg = 'whitefield_filename parameter not configured, mandatory for 34idcTIM2 detector.'
            raise ValueError(msg)
        whitefield = params['whitefield_filename']
        if not os.path.isfile(whitefield):
            msg = f'whitefield_filename file {whitefield} does not exist.'
            raise ValueError(msg)

        if 'darkfield_filename' not in params:
            msg = 'darkfield_filename parameter not configured, mandatory for 34idcTIM2 detector.'
            raise ValueError(msg)
        darkfield = params['darkfield_filename']
        if not os.path.isfile(darkfield):
            msg = f'darkfield_filename file {darkfield} does not exist.'
            raise ValueError(msg)


def create_detector(det_name, params):
    for detector in Detector.__subclasses__():
        if detector.name == det_name:
            return  detector(params)
    msg = f'detector {det_name} not defined'
    raise ValueError(msg)


dets = {'34idcTIM1' : Detector_34idcTIM1, '34idcTIM2' : Detector_34idcTIM2}

def get_pixel(det_name):
    return dets[det_name].pixel


def get_pixel_orientation(det_name):
    return dets[det_name].pixelorientation


def check_mandatory_params(det_name, params):
    for detector in Detector.__subclasses__():
        if detector.name == det_name:
            return dets[det_name].check_mandatory_params(params)
    msg = f'detector {det_name} not defined'
    raise ValueError(msg)
