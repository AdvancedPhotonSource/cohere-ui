# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import numpy as np
import h5py
from abc import ABC, abstractmethod

from cohere_core import data


class Detector(ABC):
    def __init__(self, name="default"):
        self.name = name


    def node4scan(self, scan):
        return f"{scan}.1/measurement/{self.name}"


    def nodes4scans(self, scans):
        """
        Finds nodes in hdf5 file that correspond to given scans and scan ranges.

        Parameters
        ----------
        scans : list
            list of lists defining scan(s) and scan range(s), ordered
        h5file : str
            h5file containing the data
        Returns
        -------
        list
            a list of sublist, the sublist reflecting scan ranges or scans and containing tuples of existing scans
            and nodes where the data for this scan is located
        """
        scans_nodes_ranges = []
        for (start, stop) in scans:
            # todo add check
            scans_nodes_ranges.append([(i, f"{i}.1/measurement/{self.name}") for i in range(start, stop+1)])

        return scans_nodes_ranges


    def get_scan_array(self, scans, h5file):
        """
        Reads raw rdata files from scan nodes, applies correction, and returns a dict with 3D corrected rdata
        for each node.
        Parameters
        ----------
        node : str
            node in hd5 file of scan to read the raw files from
        h5file : str
            h5file containing the rdata
        Returns
        -------
        arr : dict {str : ndarray}
            node : 3D array containing corrected rdata for one scan.
        """
        # TODO: need to find out how to parse roi from the h5file. For now it will return the full rdata.
        # It can be cropped during standard preprocessing
        data = {}
        with h5py.File(h5file, "r") as h5f:
            for s in scans:
                ar = np.array(h5f[self.node4scan(s)][:]).T
                data[s] = ar

        # print max
        print('printing max for full scan(s)')
        for s, d in data.items():
            print('scan, shape, max coordinates, max value', s, d.shape, np.unravel_index(np.argmax(d), d.shape), np.max(d))
        # # cut out roi region
        data = {s : d[self.roi[0] : self.roi[0] + self.roi[1], self.roi[2] : self.roi[2] + self.roi[3], :] for s,d in data.items()}
        print('printing max for scan(s) trimmed to roi')
        for s, d in data.items():
            print('scan, shape, max coordinates, max value', s, d.shape, np.unravel_index(np.argmax(d), d.shape), np.max(d))
            # apply correction if needed
            # the rdata already is corrected

        if self.max_crop is not None:
            for k in data.keys():
                arr = data[k]
                maxindx = np.unravel_index(arr.argmax(), arr.shape)
                while (arr[maxindx[0] + 1, maxindx[1], maxindx[2]] == 0
                       and arr[maxindx[0] - 1, maxindx[1], maxindx[2]] == 0
                       or arr[maxindx[0], maxindx[1] + 1, maxindx[2]] == 0
                       and arr[maxindx[0], maxindx[1] - 1, maxindx[2]] == 0):
                    arr[maxindx] = 0.0
                    maxindx = np.unravel_index(arr.argmax(), arr.shape)

                mc0 = int(self.max_crop[0] / 2)
                mc1 = int(self.max_crop[1] / 2)
                roislice1 = slice(maxindx[0] - mc0, maxindx[0] + mc0)
                roislice2 = slice(maxindx[1] - mc1, maxindx[1] + mc1)
                arr = arr[roislice1, roislice2, :]
                data[k] = arr

        return data


class Detector_mpxgaas(Detector):
    """
    Subclass of Detector. Encapsulates "mpxgaas" detector.
    """
    name = "mpxgaas"
    dims = (516, 516)
    roi = (0, 516, 0, 516)
    pixel = (55.0e-6, 55e-6)
    pixelorientation = ('x-', 'y-')  # in xrayutilities notation
    max_crop = None
    min_frames = None  # defines minimum frame scans in scan directory


    def __init__(self, conf_params):
        super(Detector_mpxgaas, self).__init__(self.name)
        for key, val in conf_params.items():
            if val is None:
                continue
            setattr(self, key, val)


def create_detector(det_name, params):
    for detector in Detector.__subclasses__():
        if detector.name == det_name:
            return  detector(params)
    msg = f'detector {det_name} not defined'
    raise ValueError(msg)


dets = {'mpxgaas' : Detector_mpxgaas}

def get_pixel(det_name):
    return dets[det_name].pixel


def get_pixel_orientation(det_name):
    return dets[det_name].pixelorientation
