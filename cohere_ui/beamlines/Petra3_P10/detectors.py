# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import numpy as np
import os
from abc import ABC, abstractmethod
import h5py
import cohere_ui.beamlines.Petra3_P10.p10_scan_reader as p10sr
import cohere_core.utilities as ut


class Detector(ABC):
    """
    Abstract class representing detector.
    """

    def __init__(self, name="default"):
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
        scans_dirs_ranges = [[] for _ in range(len(scans))]

        for sr_idx in range(len(scans)):
            scan_range = scans[sr_idx]
            scans_dirs = scans_dirs_ranges[sr_idx]
            for scan in range(scan_range[0], scan_range[1] + 1):
                scandir = ut.join(ut.join(self.data_dir, self.sample + '_{:05d}'.format(scan)))
                if not os.path.isdir(scandir):
                    print(f'scan directory {scandir} does not exist.')
                    continue
                if self.min_frames is not None:
                    # exclude directories with fewer files than min_frames
                    scanmeta = p10sr.P10Scan(self.data_dir, self.sample, scan, pathsave='', creat_save_folder=False)
                    nframes = int(scanmeta.get_command_infor()['motor1_step_num'])
                    if nframes < self.min_frames:
                        print(f'directory for scan {scan} contains fewer than {self.min_frames} files.')
                        continue
                scans_dirs.append((scan, scandir))
        # remove empty sub-lists
        scans_dirs_ranges = [e for e in scans_dirs_ranges if len(e) > 0]
        return scans_dirs_ranges


    def get_scan_array(self, scan_dir):
        """
        Reads/loads raw data file and applies correction.

        Reads raw data from a directory. The directory name is scan_info. The raw data is in form of 2D
        frames. The frames are read, corrected and stocked into 3D data

        :param scan_dir: directory where the detector to retrieve data for a scan
        :return: corrected data array
        """
        data_dir = os.path.join(scan_dir, self.name)
        for hfile in os.listdir(data_dir):
            if '_data_' in hfile:
                with h5py.File(ut.join(data_dir, hfile), "r") as f:
                    data = np.array(f['entry/data/data'][self.slice], dtype=float)
                break
        data = self.correct(data)
        return data.transpose()


    @abstractmethod
    def correct(self, data):
        """
        Applies the correction for detector.

        :param data: 2D raw data file representing a frame
        :return: corrected frame
        """


class Detector_e4m(Detector):
    """
    Subclass of Detector. Encapsulates "e4m" detector.
    """
    name = "e4m"
    dims = (2167, 2070)
    pixel = (75.0e-6, 75e-6)
    pixelorientation = ('x+', 'y-')  # in xrayutilities notation
    darkfield_filename = None
    darkfield = None
    data_dir = None
    min_frames= None  # defines minimum frame scans in scan directory
    Imult = 1.0
    max_crop=None
    
    ROIS={ 0:(0,0,2070,2160), 1:(0,0,1030,514), 2:(0,550,1030,1065), 3:(0,1100,1030,1616), 
      4:(0,1650,1030,2160), 5:(1040,0,2070,514), 6:(1040,550,2070,1065),
      7:(1040,1100,2070,1616), 8:(1040,1650,2070,2160)}

    module_corners= ( (0,0), (0,554), (0,1105), (0,1656), (1043,0), (1043,554), (1043,1105), (1043,1656) )
    #corner + 254,255,256,257 in X dir need to be blocked.  Only one step in Y
    #corner + 511,512,513,514
    #corner + 767,768,769,770
    #corner + 1024,1025,1026,1027
    module_corners= ( (0,0), (0,554), (0,1105), (0,1656), (1043,0), (1043,554), (1043,1105), (1043,1656) )
    module_x = (0,1043)
    module_y = (0,554,1105,1656)
    asic_x = (256, 515, 773)
    
    def __init__(self, params):
        super(Detector_e4m, self).__init__(self.name)
        # The detector attributes for background/whitefield/etc need to be set to read frames
        # this will capture things like data directory, whitefield_filename, etc.
        # keep parameters that are relevant to the detector
        self.data_dir = params.get('data_dir')
        self.sample = params.get('sample')
        mask = np.load(params['darkfield_filename'])
        mask[mask > 0] = np.nan
        mask[~np.isnan(mask)] = 1
        self.darkfield = mask
        if params.get('clear_asicbounds', True):
            for c in self.module_x:
                for x in self.asic_x:
                    self.darkfield[:, np.s_[c + x - 2:c + x + 2]] = np.nan
            for c in self.module_y:
                self.darkfield[np.s_[c + 256 - 2:c + 256 + 2], :] = np.nan

        self.min_frames = params.get('min_frames', None)
        r=self.ROIS[params.get('detector_module')]
        self.slice=np.s_[:,r[1]:r[3],r[0]:r[2]]
        self.max_crop = params.get('max_crop', None)


    def correct(self, data):
        cor = self.darkfield[self.slice[1:]]
        cor.shape = (1,) + cor.shape
        data = data * cor
        data = np.nan_to_num(data)
        if self.max_crop is not None:
            maxpos = np.unravel_index(data.argmax(), data.shape)
            maxslice = np.s_[::,
                       maxpos[1] - int(self.max_crop[0] / 2):maxpos[1] + int(self.max_crop[0] / 2),
                       maxpos[2] - int(self.max_crop[1] / 2):maxpos[2] + int(self.max_crop[1] / 2)]
            data = data[maxslice]
        return data


    @staticmethod
    def check_mandatory_params(params):
        """
        For the e4m detector the data_dir, sample, darkfield_ilename, detector_module
        are mandatory parameters.

        :params: parameters needed to create detector
        :return: message indicating problem or empty message if all is ok
        """
        if  'data_dir' not in params:
            msg = 'data_dir parameter not configured, mandatory for e4m detector.'
            raise ValueError(msg)
        data_dir = params['data_dir']
        if not os.path.isdir(data_dir):
            msg = f'data_dir directory {data_dir} does not exist.'
            raise ValueError(msg)

        if 'sample' not in params:
            msg = 'sample parameter not configured, mandatory for e4m detector.'
            raise ValueError(msg)

        if 'darkfield_filename' not in params:
            msg = 'darkfield_filename parameter not configured, mandatory for e4m detector.'
            raise ValueError(msg)
        darkfield = params['darkfield_filename']
        if not os.path.isfile(darkfield):
            msg = f'darkfield_filename file {darkfield} does not exist.'
            raise ValueError(msg)

        if 'detector_module' not in params:
            msg = 'detector_module parameter not configured, mandatory for e4m detector.'
            raise ValueError(msg)


def create_detector(det_name, params):
    for detector in Detector.__subclasses__():
        if detector.name == det_name:
            return  detector(params)
    msg = f'detector {det_name} not defined'
    raise ValueError(msg)


dets = {'e4m' : Detector_e4m}

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

