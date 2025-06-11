import os
import numpy as np
import cohere_core.utilities as ut
from abc import ABC, abstractmethod
import re


class Detector(ABC):
    """
    Class representing detector.

    Some functions are common for all detectors and are implemented in the base class.
    """

    def __init__(self, name):
        self.name = name

    def datainfo4scans(self, scans):
        """
        Finds info allowing to read data that correspond to given scans or scan ranges.
        The info can be directories where the data related to scans is stored or nodes in hd5 file
        that contain the data, or other info specific to a beamline.

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
        list of sub-lists, each sublist containing tuples with the input scans and corresponding data info
         within scan ranges.
        """
        # It is assumed scans is a single scan for simple beamline
        # The self.data_dir is a scan directory

        scan = scans[0][0]
        return [[scan, self.data_dir]]


    def get_scan_array(self, scan_info):
        """
        Reads/loads raw data file and applies correction. The correction is detector dependent.

        Reads raw data from a directory. The directory name is scan_info. The raw data is in form of 2D
        frames. The frames are read, corrected and stocked into 3D data
        This implementation is based on aps_34idc beamline.

        :param scan_info: info allowing detector to retrieve data for a scan
        :return: corrected data array
        """
        slices_files = {}
        for file_name in os.listdir(scan_info):
            if file_name.endswith('tif'):
                fnbase = file_name[:-4]   # drop the ".tif"
            elif file_name.endswith('tiff'):
                    fnbase = file_name[:-5]  # drop the ".tiff"
            else:
                continue
            # assuming the file names end with the slice number, followed by 'tif' or 'tiff' extension
            last_digits = re.search(r'\d+$', fnbase)
            if last_digits is not None:
                key = int(last_digits.group())
                slices_files[key] = ut.join(scan_info, file_name)

        ordered_keys = sorted(list(slices_files.keys()))
        ordered_slices = [self.correct_frame(slices_files[k]) for k in ordered_keys]

        return np.stack(ordered_slices, axis=-1)


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
    min_files = None  # defines minimum frame scans in scan directory
    Imult = 1.0

    def __init__(self, conf_params):
        super(Detector_34idcTIM1, self).__init__(self.name)
        # The detector attributes for background/whitefield/etc need to be set to read frames
        # this will capture things like data directory, darkfield_filename, etc.
        self.data_dir = conf_params.get('data_dir') # mandatory
        if 'roi' in conf_params:
            self.roi = conf_params.get('roi')
        if 'darkfield_filename' in conf_params:
            self.darkfield = ut.read_tif(conf_params.get('darkfield_filename'))


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


class Detector_34idcTIM2(Detector):
    """
    Subclass of Detector. Encapsulates any detector. Values are based on "34idcTIM2" detector.
    """
    name = "34idcTIM2"
    dims = (512, 512)
    roi = (0, 512, 0, 512)
    pixel = (55.0e-6, 55e-6)
    pixelorientation = ('x+', 'y-')  # in xrayutilities notation
    whitefield = None
    darkfield = None

    def __init__(self, conf_params):
        super(Detector_34idcTIM2, self).__init__(self.name)
        # The detector attributes specific for the detector.
        # Can include data directory, whitefield_filename, roi, etc.

        # keep parameters that are relevant to the detector
        if 'roi' in conf_params:
            self.roi = conf_params.get('roi')
        if 'data_dir' in conf_params:
            self.data_dir = conf_params.get('data_dir')
        if 'whitefield_filename' in conf_params:
            self.whitefield = ut.read_tif(conf_params.get('whitefield_filename'))
            # the code below is specific to TIM2 detector, excluding the correction of the weird pixels
            # self.whitefield[255:257, 0:255] = 0  # wierd pixels on edge of seam (TL/TR). Kill in WF kills in returned frame as well.
            self.wfavg = np.average(self.whitefield)
            self.wfstd = np.std(self.whitefield)
            self.whitefield = np.where(self.whitefield < self.wfavg - 3 * self.wfstd, 0, self.whitefield)
            self.Imult = conf_params.get('Imult', self.wfavg)
        if 'darkfield_filename' in conf_params:
            self.darkfield = ut.read_tif(conf_params.get('darkfield_filename'))
            if self.whitefield is not None:
                self.whitefield = np.where(self.darkfield > 1, 0, self.whitefield)  # kill known bad pixel


    def correct_frame(self, frame_filename):
        """
        Applies correction for the detector.

        This example is based on aps_34idc beamline, TIM2 detector and applies darkfield, whitefield.

        :param frame: 2D raw data file representing a frame
        :return: corrected frame
        """
        roislice1 = slice(self.roi[0], self.roi[0] + self.roi[1])
        roislice2 = slice(self.roi[2], self.roi[2] + self.roi[3])

        frame = ut.read_tif(frame_filename)
        if self.whitefield is not None:
            frame = frame / self.whitefield[roislice1, roislice2] * self.Imult
        else:
            print('whitefield_filename not given, not correcting')
        if self.darkfield is not None:
            frame = np.where(self.darkfield[roislice1, roislice2] > 1, 0.0, frame)
        else:
            print('darkfield_filename not given, not correcting')

        frame = np.where(np.isfinite(frame), frame, 0)

        return frame


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

