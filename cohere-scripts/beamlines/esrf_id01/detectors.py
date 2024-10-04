import numpy as np
import h5py
from abc import ABC, abstractmethod


class Detector(ABC):
    def __init__(self, name="default"):
        self.name = name


    def nodes4scans(self, scans, h5file):
        """
        Finds nodes in hdf5 file that correspond to given scans and scan ranges.

        Parameters
        ----------
        scans : list
            list of tuples defining scan(s) and scan range(s), ordered
        h5file : str
            h5file containing the data
        Returns
        -------
        list
            a list of sublist, the sublist reflecting scan ranges or scans and containing tuples of existing scans
            and node where the data for this scan is located
        """
        scans_nodes_ranges = []
        for (start, stop) in scans:
            scans_nodes_ranges.append([(i, f"{i}.1/measurement/{self.name}") for i in range(start, stop+1)])

        return scans_nodes_ranges


    def get_scan_array(self, node, h5file):
        """
        Reads raw data files from scan node, applies correction, and returns 3D corrected data for a single scan.
        Parameters
        ----------
        node : str
            node in hd5 file of scan to read the raw files from
        h5file : str
            h5file containing the data
        Returns
        -------
        arr : ndarray
            3D array containing corrected data for one scan.
        """
        # TODO: need to find out how to parse roi from the h5file. For now it will return the full data.
        # It can be cropped during standard preprocessing
        with h5py.File(h5file, "r") as h5f:
            data = np.array(h5f[node]).T

        # apply correction if needed
        # I think the data already is corrected

        return data
    

    def get_pixel(self):
        """
        Returns detector pixel size.  Concrete function in subclass returns value applicable to the detector.

        Returns
        -------
        tuple
            size of pixel

        """
        pass


class Detector_mpxgaas(Detector):
    """
    Subclass of Detector. Encapsulates "mpxgaas" detector.
    """
    name = "mpxgaas"
    dims = (516, 516)
    roi = (516, 516)
    pixel = (55.0e-6, 55e-6)
    pixelorientation = ('x-', 'y-')  # in xrayutilities notation


    def __init__(self):
        super(Detector_mpxgaas, self).__init__(self.name)


def create_detector(det_name):
    if det_name == 'mpxgaas':
        return Detector_mpxgaas()
    else:
        print (f'detector {det_name} not defined.')
        return None
