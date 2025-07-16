# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import numpy as np
import math as m
import xrayutilities.experiment as xuexp
from xrayutilities.io import spec as spec
import cohere_ui.beamlines.aps_34idc.detectors as det
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
    Subclass of Diffractometer. Encapsulates "34idc" diffractometer.
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

    def __init__(self, params):
        super(Diffractometer_34idc, self).__init__('34idc')
        self.specfile = params.get('specfile')


    def parse_spec(self, scan):
        """
        Reads parameters from spec file for given scan.

        Parameters
        ----------
        scan : int
            scan number to use to recover the saved measurements

        Returns
        -------
        dict with delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist, detector_name, energy
        """
        spec_dict = {}

        # Scan numbers start at one but the list is 0 indexed
        try:
            ss = spec.SPECFile(self.specfile)[scan - 1]
        except Exception as ex:
            print(str(ex))
            print('Could not parse ' + self.specfile)
            return None

        try:
            command = ss.command.split()
            spec_dict['scanmot'] = command[1]
            spec_dict['scanmot_del'] = (float(command[3]) - float(command[2])) / int(command[4])
        except:
            pass

        for mot_mne, mot_name in zip(self.sampleaxes_mne + self.detectoraxes_mne,
                                     self.sampleaxes_name + self.detectoraxes_name):
            try:
                motname = "INIT_MOPO_{m}".format(m=mot_name)
                spec_dict[mot_mne] = ss.init_motor_pos[motname]
            except:
                pass

        try:
            motname = "INIT_MOPO_{m}".format(m=self.detectordist_name)
            spec_dict['detdist'] = ss.init_motor_pos[motname]
        except:
            pass

        try:
            spec_dict['energy'] = ss.init_motor_pos['INIT_MOPO_Energy']
        except:
            pass

        try:
            spec_dict['detector'] = str(ss.getheader_element('UIMDET'))
            if spec_dict['detector'].endswith(':'):
                spec_dict['detector'] = spec_dict['detector'][:-1]
        except Exception as ex:
            print(str(ex))

        return spec_dict


    def check_params(self, params):
        if 'detector' not in params:
            print('detector name not parsed from spec file and not configured')
            raise KeyError('detector name not parsed from spec file and not configured')
        if 'detdist' not in params:
            print('detdist not parsed from spec file and not configured')
            raise KeyError('detdist not parsed from spec file and not configured')
        if 'scanmot' not in params:
            print('scanmot not parsed from spec file and not configured')
            raise KeyError('scanmot not parsed from spec file and not configured')
        if 'energy' not in params:
            print('energy not parsed from spec file and not configured')
            raise KeyError('energy not parsed from spec file and not configured')
        if 'scanmot_del' not in params:
            print('scanmot_del not parsed from spec file and not configured')
            raise KeyError('scanmot_del not parsed from spec file and not configured')
        for ax in self.sampleaxes_mne:
            if ax not in params:
                print(f'{ax} not parsed from spec file and not configured')
                raise KeyError (f'{ax} not parsed from spec file and not configured')
        for ax in self.detectoraxes_mne:
            if ax not in params:
                print(f'{ax} not parsed from spec file and not configured')
                raise KeyError (f'{ax} not parsed from spec file and not configured')


    def get_geometry(self, shape, scan, conf_params, **kwargs):
        """
        Calculates geometry based on diffractometer and detector attributes and experiment parameters.

        Typically, the delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist, detector_name,
        energy values are retrieved from experiment spec file.
        They can be overridden by configuration.

        :param shape: tuple, shape of array
        :param scan: scan the geometry is calculated for
        :param conf_params: configuration parameters, can contain delta, gamma, theta, phi, chi, scanmot,
            scanmot_del, detdist, detector_name, energy.
        :return: tuple
            (Trecip, Tdir)
        """
        params = {}
        # parse spec
        if self.specfile is not None and scan is not None:
            params.update(self.parse_spec(scan))
        # override with config params
        params.update(conf_params)
        self.check_params(params)

        binning = params.get('binning', [1, 1, 1])
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

        xtal = kwargs.get('xtal', False)
        if xtal:
            Trecip_cryst = np.zeros(9)
            Trecip_cryst.shape = (3, 3)
            Trecip_cryst[:, 0] = Astar * 10
            Trecip_cryst[:, 1] = Bstar * 10
            Trecip_cryst[:, 2] = Cstar * 10
            return Trecip_cryst, None

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


def create_diffractometer(diff_name, params):
    for diff in Diffractometer.__subclasses__():
        if diff.name == diff_name:
            return diff(params)
    msg = f'diffractometor {diff_name} not defined'
    raise ValueError(msg)
