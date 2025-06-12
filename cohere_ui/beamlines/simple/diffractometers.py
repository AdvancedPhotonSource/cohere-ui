import numpy as np
import math as m
import xrayutilities.experiment as xuexp
import cohere_ui.beamlines.simple.detectors as det
from abc import ABC


class Diffractometer(ABC):
    """
    Abstract class representing diffractometer. It keeps fields related to the specific diffractometer represented by
    a subclass.

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


class Diffractometer_34idc(Diffractometer):
    """
    Subclass of Diffractometer. Encapsulates any diffractometer. Based on aps_34idc beamline.
    """
    name = "34idc"
    sampleaxes = ('y+', 'z-', 'y+')  # in xrayutilities notation
    detectoraxes = ('y+', 'x-')
    incidentaxis = (0, 0, 1)
    sampleaxes_name = ('Theta', 'Chi', 'Phi')
    sampleaxes_mne = ('th', 'chi', 'phi')
    detectoraxes_name = ('Delta', 'Gamma')
    detectoraxes_mne = ('delta', 'gamma')
    detectordist_name = 'camdist'
    detectordist_mne = 'detdist'

    def __init__(self):
        super(Diffractometer_34idc, self).__init__('34idc')


    def check_params(self, params):
        if 'detector' not in params:
            print('detector name not configured')
            raise KeyError('detector name not configured')
        if 'detdist' not in params:
            print('detdist not configured')
            raise KeyError('detdist not configured')
        if 'scanmot' not in params:
            print('scanmot not configured')
            raise KeyError('scanmot not configured')
        if 'energy' not in params:
            print('energy not configured')
            raise KeyError('energy not configured')
        if 'scanmot_del' not in params:
            print('scanmot_del not configured')
            raise KeyError('scanmot_del not configured')
        for ax in self.sampleaxes_mne:
            if ax not in params:
                print(f'{ax} not configured')
                raise KeyError (f'{ax} not configured')
        for ax in self.detectoraxes_mne:
            if ax not in params:
                print(f'{ax} and not configured')
                raise KeyError (f'{ax} not configured')


    def get_geometry(self, shape, scan, config_maps):
        """
        Calculates geometry based on diffractometer and detector attributes and experiment parameters.

        Typically, the delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist, detector_name,
        energy values are retrieved from experiment records (spec file or hdf5 file).
        They can be overridden by configuration.
        The mneminic used in this example is may vary by beamline.

        :param shape: tuple, shape of array
        :param scan: scan the geometry is calculated for
        :param det_obj: detector object, can be None
        :param config_maps: configuration maps
        :return: tuple
            (Trecip, Tdir)
        """
        # At the beginning the parameters would be parsed either from spec file or read from hdf5 file, or
        # can be read from some database depending on the beamline. The params dictionary holds the parsed
        # parameters. It is assumed here that all parameters are from kwargs for the diffractometer,
        # so the dictionary is empty.

        params = config_maps['config_instr']
        self.check_params(params)

        binning = config_maps['config_data'].get('binning', [1, 1, 1])
        pixel = det.get_pixel(params['detector'])
        px = pixel[0] * binning[0]
        py = pixel[1] * binning[1]

        detdist = params['detdist'] / 1000.0  # convert to meters
        scanmot = params['scanmot'].strip()
        enfix = 1
        # if energy is given in kev convert to ev for xrayutilities
        energy = params['energy']
        if m.floor(m.log10(energy)) < 3:
            enfix = 1000
        energy = energy * enfix  # x-ray energy in eV

        if scanmot == 'en':
            scanen = np.array((energy, energy + params['scanmot_del'] * enfix))
        else:
            scanen = np.array((energy,))
        qc = xuexp.QConversion(self.sampleaxes, self.detectoraxes, self.incidentaxis, en=scanen)

        # compute for 4pixel (2x2) detector
        pixelorientation = det.get_pixel_orientation(params['detector'])
        qc.init_area(pixelorientation[0], pixelorientation[1], shape[0], shape[1], 2, 2,
                     distance=detdist, pwidth1=px, pwidth2=py)

        # I think q2 will always be (3,2,2,2) (vec, scanarr, px, py)
        # should put some try except around this in case something goes wrong.
        if scanmot == 'en':  # seems en scans always have to be treated differently since init is unique
            q2 = np.array(qc.area(params['th'], params['chi'], params['phi'], params['delta'], params['gamma'], deg=True))
        elif scanmot in self.sampleaxes_mne:  # based on scanmot args are made for qc.area
            args = []
            for sampleax in self.sampleaxes_mne:
                if scanmot == sampleax:
                    scanstart = params[scanmot]
                    args.append(np.array((scanstart, scanstart + params['scanmot_del'] * binning[2])))
                else:
                    args.append(params[sampleax])
            for axis in self.detectoraxes_mne:
                args.append(params[axis])

            q2 = np.array(qc.area(*args, deg=True))
        else:
            print("scanmot not in sample axes or energy, exiting")
            raise RuntimeError

        # I think q2 will always be (3,2,2,2) (vec, scanarr, px, py)
        Astar = q2[:, 0, 1, 0] - q2[:, 0, 0, 0]
        Bstar = q2[:, 0, 0, 1] - q2[:, 0, 0, 0]
        Cstar = q2[:, 1, 0, 0] - q2[:, 0, 0, 0]

        # transform to lab coords from sample reference frame
        Astar = qc.transformSample2Lab(Astar, params['th'], params['chi'], params['phi']) * 10.0  # convert to inverse nm.
        Bstar = qc.transformSample2Lab(Bstar, params['th'], params['chi'], params['phi']) * 10.0
        Cstar = qc.transformSample2Lab(Cstar, params['th'], params['chi'], params['phi']) * 10.0

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
