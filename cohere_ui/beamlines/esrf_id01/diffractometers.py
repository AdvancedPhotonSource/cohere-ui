# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import numpy as np
import math as m
import xrayutilities.experiment as xuexp
import h5py
import cohere_ui.beamlines.esrf_id01.detectors as det
from abc import ABC, abstractmethod


class Diffractometer(ABC):

    """
    class Diffractometer(self, diff_name)
    ============================================

    Abstract class representing diffractometer. It keeps fields related to the specific diffractometer represented by a subclass.
        diff_name : str
            diffractometer name

    """
    name = None

    def __init__(self, diff_name):
        """
        Constructor.

        Parameters
        ----------
        diff_name : str
            diffractometer name

        """
        self.name = diff_name


class Diffractometer_id01(Diffractometer):
    """
    Subclass of Diffractometer. Encapsulates "id01" diffractometer.
    """
    name = "id01"
    sampleaxes = ('y-', 'x-', 'y-')  # in xrayutilities notation
    detectoraxes = ('y-', 'x-')
    incidentaxis = (0, 0, 1)
    sampleaxes_name = ('Mu', 'Eta', 'Phi')
    sampleaxes_mne = ('mu', 'eta', 'phi')
    detectoraxes_name = ('Nu', 'Delta')
    detectoraxes_mne = ('nu', 'delta')
    detectordist_name = 'distance'
    detectordist_mne = 'detdist'


    def __init__(self):
        super(Diffractometer_id01, self).__init__('id01')


    def parse_h5(self, h5file, scan, detector):
        """
        Reads parameters from h5 file for given scan.

        Parameters
        ----------
        h5file : str
            h5 file name

        scan : int
            scan number to use to recover the saved measurements

        diff : object
            diffractometer object

        Returns
        -------
        dict with delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist, detector_name, energy
        """
        h5_dict = {}

        # Scan numbers start at one but the list is 0 indexed
        h5f = h5py.File(h5file)
        info = h5f[f"{scan}.1"]

        try:
            h5_dict['detector'] = detector
            command = info['title'].asstr()[()].split(" ")
            if command[0] in ("ascan", "a2scan", "a3scan"):
                h5_dict['scanmot'] = command[1]
                h5_dict['scanmot_del'] = (float(command[3]) - float(command[2])) / int(command[4])
            else:
                raise IOError(f"{__name__}: Unknown scan type: {command[0]}")

            for mot_mne in self.sampleaxes_mne + self.detectoraxes_mne:
                h5_dict[mot_mne] = info[f'instrument/positioners/{mot_mne}'][()]

            h5_dict['detdist'] = info[f'instrument/{detector}/{self.detectordist_name}'][()]

            h5_dict['energy'] = info['instrument/monochromator/Energy'][()]
        except Exception as ex:
            print(f"{__name__}: {ex}")
            raise ex
        h5f.close()

        return h5_dict


    def get_geometry(self, shape, scan, conf_params):
        """
        Calculates geometry for given scan based on diffractometer's attributes and experiment parameters.
        :param shape: shape of the array
        :param scan: scan
        :param conf_params: Parameters reflecting configuration
        :return: tuple, geometry in reciprocal, geometry in direct space
        """
        params = self.parse_h5(conf_params['h5file'], scan, conf_params['detector'])
        params.update(conf_params)

        binning = params.get('binning', [1, 1, 1])
        pixel = det.get_pixel(params['detector'])
        px = pixel[0] * binning[0]
        py = pixel[1] * binning[1]

        detdist = params.get('detdist')
        scanmot = params.get('scanmot').strip()
        enfix = 1
        # if energy is given in kev convert to ev for xrayutilities
        energy = params['energy']
        if m.floor(m.log10(energy)) < 3:
            enfix = 1000
        energy = energy * enfix  # x-ray energy in eV

        if scanmot == 'en':
            scanen = np.array((energy, energy + params.get('scanmot_del') * enfix))
        else:
            scanen = np.array((energy,))
        qc = xuexp.QConversion(self.sampleaxes, self.detectoraxes, self.incidentaxis, en=scanen)

        # compute for 4pixel (2x2) detector
        pixelorientation = det.get_pixel_orientation(params['detector'])
        qc.init_area(pixelorientation[0], pixelorientation[1], shape[0], shape[1], 2, 2,
                     distance=detdist, pwidth1=px, pwidth2=py)

        if scanmot == 'en':  # seems en scans always have to be treated differently since init is unique
            q2 = np.array(qc.area(params.get('mu'), params.get('eta'), params.get('phi'),
                                  params.get('nu'), params.get('delta'), deg=False))
        elif scanmot in self.sampleaxes_mne:  # based on scanmot args are made for qc.area
            args = []
            for sampleax in self.sampleaxes_mne:
                if scanmot == sampleax:
                    scaninfo = params[scanmot]
                    # checking type, in 34-idc the 'th' type is float and it is the scanstart.
                    # array is built by adding scanmot_del for each step.
                    # for the esrf the 'eta' is an array, so in the 'else' clouse no need to build the array.
                    # adding hack to change the 'eta' attribute from array to the float - first element (scanstart)
                    if type(scaninfo) == float:
                        scanstart = params[scanmot]
                        args.append(np.array((scanstart, scanstart + params.get('scanmot_del') * binning[2])))
                    else:
                        args.append(scaninfo[::binning[2]])
                        params[scanmot] = params[scanmot][0]
                else:
                    args.append(params[sampleax])
            for axis in self.detectoraxes_mne:
                args.append(params[axis])
            q2 = np.array(qc.area(*args, deg=True))
        else:
            print(f"{__name__}: scanmot not in sample axes or energy, exiting")
            raise RuntimeError

        Astar = q2[:, 0, 1, 0] - q2[:, 0, 0, 0]
        Bstar = q2[:, 0, 0, 1] - q2[:, 0, 0, 0]
        Cstar = q2[:, 1, 0, 0] - q2[:, 0, 0, 0]

        # transform to lab coords from sample reference frame
        Astar = qc.transformSample2Lab(Astar, params['mu'], params['eta'], params['phi']) * 10.0  # convert to inverse nm.
        Bstar = qc.transformSample2Lab(Bstar, params['mu'], params['eta'], params['phi']) * 10.0
        Cstar = qc.transformSample2Lab(Cstar, params['mu'], params['eta'], params['phi']) * 10.0

        denom = np.dot(Astar, np.cross(Bstar, Cstar))
        A = 2 * m.pi * np.cross(Bstar, Cstar) / denom
        B = 2 * m.pi * np.cross(Cstar, Astar) / denom
        C = 2 * m.pi * np.cross(Astar, Bstar) / denom

        Trecip = np.zeros(9)
        Trecip.shape = (3, 3)
        Trecip[:, 0] = Astar
        Trecip[:, 1] = Bstar
        Trecip[:, 2] = Cstar

        Tdir = np.zeros(9)
        Tdir.shape = (3, 3)
        Tdir = np.array((A, B, C)).transpose()

        return (Trecip, Tdir)


def create_diffractometer(diff_name):
    for diff in Diffractometer.__subclasses__():
        if diff.name == diff_name:
            return diff()
    msg = f'diffractometor {diff_name} not defined'
    raise ValueError(msg)
