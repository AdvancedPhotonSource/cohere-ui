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

    def dirs4scans(self, scans):
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
            ex1: [[2825, 2831], [2834, 2834], [2840, 2876]]
            returns: [[(2825, f'{path}/data_S2825'), (2828, f'{path}/data_S2828'), (2831, f'{path}/data_S2831')],
             [(2834, f'{path}/data_S2834)],
             [(2840, f'{path}/data_S2840'), (2843, f'{path}/data_S2843'), (2846, f'{path}/data_S2846'), (2849, f'{path}/data_S2849'),
              (2852, f'{path}/data_S2852'), (2855, f'{path}/data_S2855'), (2858, f'{path}/data_S2858'), (2861, f'{path}/data_S2861'),
              (2864, f'{path}/data_S2864'), (2867, f'{path}/data_S2867'), (2870, f'{path}/data_S2870'), (2873, f'{path}/data_S2873'),
              (2876, f'{path}/data_S2876')]]

        :return:
        list of sub-lists the input scans, or scans ranges with the corresponding info
        """
        # Below is an implementation from aps_34idc beamline. Look at the implementation for the esrf_id01 beamline
        # if using hdf5 file.

        # create empty results list that allocates a sub-list for each scan range
        scans_dirs_ranges = [[] for _ in range(len(scans))]
        sr_idx = 0
        scan_range = scans[sr_idx]
        scans_dirs = scans_dirs_ranges[sr_idx]
        
        #for scandir in sorted(glob.glob(os.path.join(self.data_dir, self.sample+'*'))):
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
                    scans_dirs.append((scan, scandir_full))
                    if scan == scan_range[-1]:
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

        arr= np.stack(ordered_slices, axis=-1)
        if self.maxcrop is not None:
            maxindx= np.unravel_index(arr.argmax(), arr.shape)
            mc0=int(self.maxcrop[0]/2)
            mc1=int(self.maxcrop[1]/2)
            roislice1 = slice(maxindx[0]-mc0, maxindx[0]+mc0)
            roislice2 = slice(maxindx[1]-mc1, maxindx[1]+mc1)
            arr=arr[roislice1, roislice2, :]
        return arr


    @abstractmethod
    def correct_frame(self, frame):
        """
        Applies the correction for detector.

        :param frame: 2D raw data file representing a frame
        :return: corrected frame
        """


class ASI(Detector):
    """
    Subclass of Detector. Encapsulates any detector. Values are based on "34idcTIM2" detector.
    """
    name = "ASI"
    dims = (518, 518)
    roi = (0, 512, 0, 512)
    pixel = (55.0e-6, 55e-6)
    pixelorientation = ('x+', 'y-')  # in xrayutilities notation
    whitefield = None
    maxcrop = None

    def __init__(self, params):
        super(ASI, self).__init__(self.name)
        # The detector attributes specific for the detector.
        # Can include data directory, whitefield_filename, roi, etc.

        if 'maxcrop' in params:
            self.maxcrop=params['maxcrop']
        # keep parameters that are relevant to the detector
        if 'roi' in params:
            self.roi = params.get('roi')
        if 'data_dir' in params:
            self.data_dir = params.get('data_dir')
        if 'whitefield_filename' in params:
            self.whitefield = ut.read_tif(params.get('whitefield_filename'))
            # the code below is specific to ASI detector
            self.wfavg = np.average(self.whitefield)
            self.wfstd = np.std(self.whitefield)
            self.whitefield = np.where(self.whitefield < self.wfavg - 3 * self.wfstd, 0, self.whitefield)
            self.Imult = params.get('Imult', self.wfavg)

    def correct_frame(self, frame_filename):
        """
        Applies correction for the detector.

        This example is based on aps_34idc beamline, TIM2 detector and applies darkfield, whitefield.

        :param frame: 2D raw data file representing a frame
        :return: corrected frame
        """
        roislice1 = slice(self.roi[0], self.roi[0] + self.roi[1])
        roislice2 = slice(self.roi[2], self.roi[2] + self.roi[3])
        frame = ut.read_tif(frame_filename)[roislice1, roislice2]

        if self.whitefield is not None:
            frame = frame / self.whitefield[roislice1, roislice2] * self.Imult
        else:
            print('whitefield_filename not given, not correcting')
            pass

        frame = np.where(np.isfinite(frame), frame, 0)

        return frame


def create_detector(det_name, kwargs):
    if det_name == 'ASI':
        return ASI(kwargs)
    else:
        print(f'detector {det_name} not defined.')
        return None


dets = {'ASI' : ASI}

def get_pixel(det_name):
    return dets[det_name].pixel


def get_pixel_orientation(det_name):
    return dets[det_name].pixelorientation

