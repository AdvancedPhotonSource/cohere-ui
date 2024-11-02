import numpy as np
import os
import re
import glob
import cohere_core.utilities as ut
from abc import ABC, abstractmethod


class Detector(ABC):
    """
    Abstract class representing detector.
    """

    def __init__(self, name="default"):
        self.name = name

    def dirs4scans(self, scans):
        """
        Finds existing sub-directories in data_dir that correspond to given scans and scan ranges.
        This function is the same for aps_34idc detectors as the naming scheme is consistent for this beamline.

        Parameters
        ----------
        scans : list
            list of tuples defining scan(s) and scan range(s), ordered

        Returns
        -------
        list
            a list of sublist, the sublist reflecting scan ranges or scans and containing tuples of existing scans
            and directory where the data for this scan is located
        """
        # create empty results list that allocates a sub-list for each scan range
        scans_dirs_ranges = [[] for _ in range(len(scans))]
        sr_idx = 0
        scan_range = scans[sr_idx]
        scans_dirs = scans_dirs_ranges[sr_idx]

        for scandir in sorted(os.listdir(self.data_dir)):
            scandir_full = ut.join(self.data_dir, scandir)
            if os.path.isdir(scandir_full):
                last_digits = re.search(r'\d+$', scandir_full)
                if last_digits is not None:
                    scan = int(last_digits.group())
                if scan < scan_range[0]:
                    continue
                elif scan <= scan_range[-1]:
                    # scan within range
                    if self.min_files is not None:
                        # exclude directories with fewer tif files than min_files
                        if len(glob.glob1(scandir, "*.tif")) < self.min_files:
                            continue
                    scans_dirs.append((scan, scandir_full))
                else:
                    # The scan exceeded range
                    # move to the next scan range
                    sr_idx += 1
                    if sr_idx > len(scans) - 1:
                        break
                    scan_range = scans[sr_idx]
                    scans_dirs = scans_dirs_ranges[sr_idx]

        # remove empty sub-lists
        scans_dirs_ranges = [e for e in scans_dirs_ranges if len(e) > 0]
        return scans_dirs_ranges

    def get_scan_array(self, dir):
        """
        Reads raw data files from scan directory, applies correction, and returns 3D corrected data for a single scan directory.
        The correction is detector dependent. It can be darkfield and/or whitefield correction.
        Parameters
        ----------
        dir : str
            directory of scan to read the raw files from
        Returns
        -------
        arr : ndarray
            3D array containing corrected data for one scan.
        """
        slices_files = {}
        for file_name in os.listdir(dir):
            if file_name.endswith('tif'):
                fnbase = file_name[:-4]
            else:
                continue
            # for aps_34idc the file names end with the slice number, followed by 'tif' extension
            last_digits = re.search(r'\d+$', fnbase)
            if last_digits is not None:
                key = int(last_digits.group())
                slices_files[key] = ut.join(dir, file_name)

        ordered_keys = sorted(list(slices_files.keys()))
        ordered_slices = [self.get_frame(slices_files[k]) for k in ordered_keys]

        return np.stack(ordered_slices, axis=-1)

    def get_raw_frame(self, filename):
        return ut.read_tif(filename)
        # try:
        #     self.raw_frame = ut.read_tif(filename)
        # except:
        #     print("problem reading raw file ", filename)
        #     raise

    @abstractmethod
    def get_frame(self, filename):
        """
        Reads raw 2D frame from a file. Concrete function in subclass applies correction for the specific detector. For example it could be darkfield correction or whitefield correction.

        Parameters
        ----------
        filename : str
            data file name
        roi : list
            detector area used to take image. If None the entire detector area will be used.
        Imult : int
            multiplier

        Returns
        -------
        ndarray
            frame after instrument correction

        """
        pass

    def get_pixel(self):
        """
        Returns detector pixel size.  Concrete function in subclass returns value applicable to the detector.

        Returns
        -------
        tuple
            size of pixel

        """
        return self.pixel


class Detector_34idcTIM1(Detector):
    """
    Subclass of Detector. Encapsulates "34idcTIM1" detector.
    """
    name = "34idcTIM1"
    dims = (256, 256)
    roi = (0, 256, 0, 256)
    pixel = (55.0e-6, 55e-6)
    pixelorientation = ('x+', 'y-')  # in xrayutilities notation
    darkfield_filename = None
    darkfield = None
    data_dir = None
    min_files = None  # defines minimum frame scans in scan directory
    Imult = 1.0

    def __init__(self, **kwargs):
        super(Detector_34idcTIM1, self).__init__()
        # The detector attributes for background/whitefield/etc need to be set to read frames
        # this will capture things like data directory, darkfield_filename, etc.
        for key, val in kwargs.items():
            setattr(self, key, val)

    def load_darkfield(self):
        """
        Reads darkfield file and save the frame as class member.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        try:
            self.darkfield = ut.read_tif(self.darkfield_filename)
        except:
            print("Darkfield filename not set for TIM1, will not correct")

    # TIM1 only needs bad pixels deleted.  Even that is optional.
    def get_frame(self, filename):
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
        if self.darkfield is None:
            if not self.darkfield_filename is None:
                self.load_darkfield()
            else:
                print("Darkfield filename not configured for TIM1, will not correct")
        roislice1 = slice(self.roi[0], self.roi[0] + self.roi[1])
        roislice2 = slice(self.roi[2], self.roi[2] + self.roi[3])
        raw_frame = self.get_raw_frame(filename)
        try:
            frame = np.where(self.darkfield[roislice1, roislice2] > 1, 0.0, raw_frame)
        except:
            frame = raw_frame

        return frame


class Detector_34idcTIM2(Detector):
    """
    Subclass of Detector. Encapsulates "34idcTIM2" detector.
    """
    name = "34idcTIM2"
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
        super(Detector_34idcTIM2, self).__init__()
        # The detector attributes for background/whitefield/etc need to be set to read frames
        # this will capture things like data directory, whitefield_filename, etc.
        for key, val in kwargs.items():
            setattr(self, key, val)

    def load_whitefield(self):
        """
        Reads whitefield file and save the frame as class member.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        try:
            self.whitefield = ut.read_tif(self.whitefield_filename)
        except:
            print("Whitefield filename not set for TIM2")
            raise
        try:
            self.whitefield[255:257,
            0:255] = 0  # wierd pixels on edge of seam (TL/TR). Kill in WF kills in returned frame as well.
            self.wfavg = np.average(self.whitefield)
            self.wfstd = np.std(self.whitefield)
            self.whitefield = np.where(self.whitefield < self.wfavg - 3 * self.wfstd, 0, self.whitefield)
        except:
            print("Corrections to the TIM2 whitefield image failed in detector module.")

    def load_darkfield(self):
        """
        Reads darkfield file and save the frame as class member.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        try:
            self.darkfield = ut.read_tif(self.darkfield_filename)
        except:
            print("Darkfield filename not set for TIM2")
            raise
        if type(self.whitefield) == np.ndarray:
            self.whitefield = np.where(self.darkfield > 1, 0, self.whitefield)  # kill known bad pixel

    def get_frame(self, filename):
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
        if self.darkfield is None:
            if not self.darkfield_filename is None:
                self.load_darkfield()
            else:
                print("darkfield filename not configured for TIM2, will not correct")
        if self.whitefield is None:
            if not self.whitefield_filename is None:
                self.load_whitefield()
            else:
                print("whitefield filename not configured for TIM2, will not correct")
        # roi is start,size,start,size
        # will be in imageJ coords, so might need to transpose,or just switch x-y
        # divide whitefield
        # blank out pixels identified in darkfield
        # insert 4 cols 5 rows if roi crosses asic boundary
        if self.roi is None:
            self.roi = Detector_34idcTIM2.roi
        if not type(self.darkfield) == np.ndarray:
            self.load_darkfield()
        if not type(self.whitefield) == np.ndarray:
            self.load_whitefield()
        if self.Imult is None:
            self.Imult = self.wfavg

        roislice1 = slice(self.roi[0], self.roi[0] + self.roi[1])
        roislice2 = slice(self.roi[2], self.roi[2] + self.roi[3])

        # some of this should probably be in try blocks
        raw_frame = self.get_raw_frame(filename)
        normframe = raw_frame / self.whitefield[roislice1, roislice2] * self.Imult
        normframe = np.where(self.darkfield[roislice1, roislice2] > 1, 0.0, normframe)
        normframe = np.where(np.isfinite(normframe), normframe, 0)

        frame, seam_added = self.insert_seam(normframe)
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


class default(Detector):
    """
    Subclass of Detector. Encapsulates any detector. Based on "34idcTIM2" detector.
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
        super(default, self).__init__()
        # The detector attributes for background/whitefield/etc need to be set to read frames
        # this will capture things like data directory, whitefield_filename, etc.
        for key, val in kwargs.items():
            setattr(self, key, val)

    def load_whitefield(self):
        """
        Reads whitefield file and save the frame as class member.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        try:
            self.whitefield = ut.read_tif(self.whitefield_filename)
        except:
            print("Whitefield filename not set for TIM2")
            raise
        try:
            self.whitefield[255:257,
            0:255] = 0  # wierd pixels on edge of seam (TL/TR). Kill in WF kills in returned frame as well.
            self.wfavg = np.average(self.whitefield)
            self.wfstd = np.std(self.whitefield)
            self.whitefield = np.where(self.whitefield < self.wfavg - 3 * self.wfstd, 0, self.whitefield)
        except:
            print("Corrections to the TIM2 whitefield image failed in detector module.")

    def load_darkfield(self):
        """
        Reads darkfield file and save the frame as class member.
        Parameters
        ----------
        none
        Returns
        -------
        nothing
        """
        try:
            self.darkfield = ut.read_tif(self.darkfield_filename)
        except:
            print("Darkfield filename not set for TIM2")
            raise
        if type(self.whitefield) == np.ndarray:
            self.whitefield = np.where(self.darkfield > 1, 0, self.whitefield)  # kill known bad pixel

    def get_frame(self, filename):
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
        if self.darkfield is None:
            if not self.darkfield_filename is None:
                self.load_darkfield()
            else:
                print("darkfield filename not configured for TIM2, will not correct")
        if self.whitefield is None:
            if not self.whitefield_filename is None:
                self.load_whitefield()
            else:
                print("whitefield filename not configured for TIM2, will not correct")
        # roi is start,size,start,size
        # will be in imageJ coords, so might need to transpose,or just switch x-y
        # divide whitefield
        # blank out pixels identified in darkfield
        # insert 4 cols 5 rows if roi crosses asic boundary
        if self.roi is None:
            self.roi = Detector_34idcTIM2.roi
        if not type(self.darkfield) == np.ndarray:
            self.load_darkfield()
        if not type(self.whitefield) == np.ndarray:
            self.load_whitefield()
        if self.Imult is None:
            self.Imult = self.wfavg

        roislice1 = slice(self.roi[0], self.roi[0] + self.roi[1])
        roislice2 = slice(self.roi[2], self.roi[2] + self.roi[3])

        # some of this should probably be in try blocks
        raw_frame = self.get_raw_frame(filename)
        normframe = raw_frame / self.whitefield[roislice1, roislice2] * self.Imult
        normframe = np.where(self.darkfield[roislice1, roislice2] > 1, 0.0, normframe)
        normframe = np.where(np.isfinite(normframe), normframe, 0)

        frame, seam_added = self.insert_seam(normframe)
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


def create_detector(det_name, **kwargs):
    if det_name == '34idcTIM1':
        return Detector_34idcTIM1(**kwargs)
    elif det_name == '34idcTIM2':
        return Detector_34idcTIM2(**kwargs)
    elif det_name is None:
        return default(**kwargs)
    else:
        print(f'detector {det_name} not defined.')
        return None


