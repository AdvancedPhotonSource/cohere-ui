import numpy as np
import os
from abc import ABC, abstractmethod
import h5py
import beamlines.Petra3_P10.p10_scan_reader as p10sr
import cohere_core.utilities as ut


class Detector(ABC):
    """
    Abstract class representing detector.
    """

    def __init__(self, name="default"):
        self.name = name


    def dirs4scans(self, scans):
        """
        Finds info allowing to read data that correspond to given scans or scan ranges.
        The info can be directories where the data related to scans is stored.

        :param scans : list
            list of sub-lists defining scan ranges, ordered. For single scan a range has the same scan as beginning and end.
            one scan example:
            scans : [[2834, 2834]]
            returns : [[(2834, f'{path}/data_S2834)]]

            separate ranges example:
            ex1: [[2825, 2831], [2834, 2834], [2840, 2876]]
            returns: [[(2825, f'{path}/data_S2825'), (2828, f'{path}/data_S2828'), (2831, f'{path}/data_S2831')],
             [(2834, f'{path}/data_S2834)],
             [(2840, f'{path}/data_S2840'), (2843, f'{path}/data_S2843'), (2846, f'{path}/data_S2846'), (2849, f'{path}/data_S2849'),
              (2852, f'{path}/data_S2852'), (2855, f'{path}/data_S2855'), (2858, f'{path}/data_S2858'), (2861, f'{path}/data_S2861'),
              (2864, f'{path}/data_S2864'), (2867, f'{path}/data_S2867'), (2870, f'{path}/data_S2870'), (2873, f'{path}/data_S2873'),
              (2876, f'{path}/data_S2876')]]

        :return:
        list of sub-lists the input scans, or scans ranges with the corresponding directories
        """
        scans_dirs_ranges = [[] for _ in range(len(scans))]

        for sr_idx in range(len(scans)):
            scan_range = scans[sr_idx]
            scans_dirs = scans_dirs_ranges[sr_idx]
            for scan in range(scan_range[0], scan_range[1] + 1):
                scandir = ut.join(ut.join(self.data_dir, self.sample + '_{:05d}'.format(scan)))
                if not os.path.isdir(scandir):
                    continue
                if self.min_files is not None:
                    # exclude directories with fewer files than min_files
                    scanmeta = p10sr.P10Scan(self.data_dir, self.sample, scan, pathsave='', creat_save_folder=False)
                    nframes = int(scanmeta.get_command_infor()['motor1_step_num'])
                    if nframes < self.min_files:
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

        :param frame: 2D raw data file representing a frame
        :return: corrected frame
        """


class Detector_e4m(Detector):
    """
    Subclass of Detector. Encapsulates "34idcTIM1" detector.
    """
    name = "e4m"
    dims = (2167, 2070)
    roi = (0, 256, 0, 256)
    pixel = (75.0e-6, 75e-6)
    pixelorientation = ('x+', 'y-')  # in xrayutilities notation
    darkfield_filename = None
    darkfield = None
    data_dir = None
    min_files= None  # defines minimum frame scans in scan directory
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
    
    def __init__(self, **kwargs):
        super(Detector_e4m, self).__init__(self.name)
        # The detector attributes for background/whitefield/etc need to be set to read frames
        # this will capture things like data directory, whitefield_filename, etc.
        # keep parameters that are relevant to the detector
        self.data_dir = kwargs.get('data_dir')
        self.sample = kwargs.get('sample')
        if 'darkfield_filename' in kwargs:
            try:
                mask = np.load(kwargs['darkfield_filename'])
                mask[mask > 0] = np.nan
                mask[~np.isnan(mask)] = 1
                self.darkfield = mask
            except:
                print("Darkfield filename not set for e4m, will not correct")
                raise

            if kwargs.get('clear_asicbounds', True):
                for c in self.module_x:
                    for x in self.asic_x:
                        self.darkfield[:, np.s_[c + x - 2:c + x + 2]] = np.nan
                for c in self.module_y:
                    self.darkfield[np.s_[c + 256 - 2:c + 256 + 2], :] = np.nan

        self.min_files = kwargs.get('min_files', None)
        r=self.ROIS[kwargs.get('detector_module', 0)]
        self.slice=np.s_[:,r[1]:r[3],r[0]:r[2]]


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


def create_detector(det_name, **kwargs):
    if det_name == 'e4m':
        return Detector_e4m(**kwargs)
    else:
        print(f'detector {det_name} not defined.')
        return None


dets = {'e4m' : Detector_e4m}

def get_pixel(det_name):
    return dets[det_name].pixel


def get_pixel_orientation(det_name):
    return dets[det_name].pixelorientation

