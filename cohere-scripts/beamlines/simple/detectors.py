import numpy as np
import cohere_core.utilities as ut
from abc import ABC, abstractmethod


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

        :param scans :
            list of lists defining scan ranges, ordered. For single scan a range is formed with the same scan
            (in example scan 2834).
            one scan example:
            scans : [[2834, 2834]]
            returns : [[(2834, f'{path}/data_S2834)]]

            ex1: [[2825, 2831], [2834, 2834], [2840, 2876]]

        :return:
        list of the input scans, or scans ranges with the corresponding info
        In the example below the info is a directory.
        ex1: [[(2825, f'{path}/data_S2825'), (2828, f'{path}/data_S2828'), (2831, f'{path}/data_S2831')],
             [(2834, f'{path}/data_S2834)],
             [(2840, f'{path}/data_S2840'), (2843, f'{path}/data_S2843'), (2846, f'{path}/data_S2846'), (2849, f'{path}/data_S2849'),
              (2852, f'{path}/data_S2852'), (2855, f'{path}/data_S2855'), (2858, f'{path}/data_S2858'), (2861, f'{path}/data_S2861'),
              (2864, f'{path}/data_S2864'), (2867, f'{path}/data_S2867'), (2870, f'{path}/data_S2870'), (2873, f'{path}/data_S2873'),
              (2876, f'{path}/data_S2876')]]
        """
        # Below is an example from esrf_id01 beamline
        # scans_nodes_ranges = []
        # for (start, stop) in scans:
        #     scans_nodes_ranges.append([(i, f"{i}.1/measurement/{self.name}") for i in range(start, stop+1)])
        #
        # return scans_nodes_ranges

        # for test only. Returns full filename of prep_data.tif processed before for the scan
        scan = scans[0][0]
        return [[(scan, f'example_workspace/scan_{scan}/preprocessed_data/prep_data.tif')]]


    def get_scan_array(self, scan_info):
        """
        Reads/loads raw data file and applies correction. The correction is detector dependent.

        :param scan_info: info allowing detector to retrieve data for a scan
        :return: corrected data array
        """
        # This is an example of implementation. The functions get_scan_data and correct have to be implemented
        # in the detector. In some cases the correction would be done within the getting scan data (for aps_34idc).
        # The roi should be accounted for during the process.
        # scan_data = get_scan_data(scan_info)
        # scan_data = correct(scan_data)
        # return scan_data

        # for test only. Reading a file assuming name scan_info.
        return ut.read_tif(scan_info)


    def get_pixel(self):
        """
        Returns detector pixel size.  Concrete function in subclass returns value applicable to the detector.

        :return: tuple, size of pixel
        """
        return self.pixel


class Default(Detector):
    """
    Subclass of Detector. Encapsulates any detector. Values are based on "34idcTIM2" detector.
    """
    name = "default"
    dims = (512, 512)
    roi = (0, 512, 0, 512)
    pixel = (55.0e-6, 55e-6)
    pixelorientation = ('x+', 'y-')  # in xrayutilities notation
    whitefield_filename = None
    darkfield_filename = None
    whitefield = None
    darkfield = None
    raw_frame = None
    min_files = None  # defines minimum frame scans in scan directory
    Imult = None

    def __init__(self, **kwargs):
        super(Default, self).__init__(self.name)
        # The detector attributes specific for the detector.
        # Can include data directory, whitefield_filename, roi, etc.
        for key, val in kwargs.items():
            setattr(self, key, val)

    # Below place functions that will deliver the functionality to the get_scan_array function.

def create_detector(det_name, **kwargs):
    if det_name == 'default':
        return Default(**kwargs)
    else:
        print(f'detector {det_name} not defined.')
        return None


