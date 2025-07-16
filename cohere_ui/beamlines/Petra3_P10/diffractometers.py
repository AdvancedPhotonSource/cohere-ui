# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import os.path
import cohere_core.utilities as ut
import numpy as np
import math as m
import xrayutilities.experiment as xuexp
import cohere_ui.beamlines.Petra3_P10.detectors as det
import cohere_ui.beamlines.Petra3_P10.p10_scan_reader as p10sr


class Diffractometer():
    """
    Parent class representing diffractometer. It keeps fields related to the specific diffractometer represented by
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


class Diffractometer_P10sixc(Diffractometer):
    """
    Subclass of Diffractometer. Encapsulates "P10sixc" diffractometer.
    """
    name = "P10sixc"
    sampleaxes = ('y+', 'x-', 'z+', 'y-')  # in xrayutilities notation
    detectoraxes = ('y+', 'x-')
    incidentaxis = (0, 0, 1)
    sampleaxes_name = ('mu', 'om', 'chi', 'phi')
    sampleaxes_mne = ('mu', 'om', 'chi', 'phi')
    detectoraxes_name = ('Gamma', 'Delta')
    detectoraxes_mne = ('gam','del')

    def __init__(self, params):
        super(Diffractometer_P10sixc, self).__init__(self.name)
        self.data_dir = params['data_dir']
        self.sample = params['sample']


    #Here the fiofile is the P10 fio object.  So no need to read a file.
    def parse_fio(self, scan):
        """
        Reads parameters from fio file for given scan. The fio file is derived from data_dir sample and scan.
        :param data_dir: directory where data along with fio file are saved
        :param sample: sample name that is used as subdirectory where the fio file is saved
        :param scan: scan defines the subdirectory
        :return: dict with optional params: scanmot, scanmot_del, detdist, detector, energy
        """
        # check if the values are meaningful
        if self.data_dir is None or self.sample is None:
            return {}
        if not os.path.isdir(self.data_dir):
            print (f"the data path {self.data_dir} does not exist, parsing not possible." )
            return {}
        if not os.path.isdir(ut.join(self.data_dir, self.sample + '_{:05d}'.format(scan))):
            print (f"the data/sample path {self.data_dir}/{self.sample + '_{:05d}'.format(scan)} does not exist, parsing not possible." )
            return {}
        fio_dict = {}
        scanmeta = p10sr.P10Scan(self.data_dir, self.sample, scan, pathsave='', creat_save_folder=False)
        command = scanmeta.command.split()
        fio_dict['scanmot'] = command[1]
        fio_dict['scanmot_del'] = (float(command[3]) - float(command[2])) / int(command[4])

        for mot_mne, mot_name in zip(self.sampleaxes_mne + self.detectoraxes_mne,
                                     self.sampleaxes_name + self.detectoraxes_name):
            fio_dict[mot_mne] = scanmeta.get_motor_pos(mot_mne)

        fio_dict['detdist'] = scanmeta.get_motor_pos('_distance')


        fio_dict['energy'] = scanmeta.get_motor_pos('fmbenergy')

        try:
            fio_dict['detector'] = scanmeta.get_motor_pos('_ccd')
        except Exception as ex:
            print(str(ex))

        return fio_dict


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


    def get_geometry(self, shape, scan, conf_params):
        """
        Calculates geometry based on diffractometer's and detector's attributes and experiment parameters.

        For the Petra3_P10 scanmot, scanmot_del, detdist, detector_name, energy values are parsed from fio file.
        They can be overridden by configuration.

        Parameters
        ----------
        shape : tuple
            shape of reconstructed array
        scan : int
            scan number the geometry is calculated for
        conf_params : reflect configuration, and could contain del, gam, theta, phi, chi, scanmot, scanmot_del,
        detdist, detector_name, energy.

        Returns
        -------
        tuple
            (Trecip, Tdir)
        """
        params = {}
        # parse fiofile
        if scan is not None:
            params.update(self.parse_fio(scan))
        # override with config params
        params.update(conf_params)

        binning = params.get('binning', [1, 1, 1])
        pixel = det.get_pixel(params['detector'])
        px = pixel[0] * binning[0]
        py = pixel[1] * binning[1]

        detdist = params.get('detdist') / 1000.0  # convert to meters
        scanmot = params.get('scanmot').strip()
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

        if scanmot == 'en':  # seems en scans always have to be treated differently since init is unique
            q2 = np.array(qc.area(params['mu'], params['om'], params['chi'], params['phi'], params['del'],
            params['gam'], deg=True))
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

        Astar = q2[:, 0, 1, 0] - q2[:, 0, 0, 0]
        Bstar = q2[:, 0, 0, 1] - q2[:, 0, 0, 0]
        Cstar = q2[:, 1, 0, 0] - q2[:, 0, 0, 0]

        # transform to lab coords from sample reference frame
        Astar = qc.transformSample2Lab(Astar, params['mu'], params['om'], params['chi'], params['phi']) * 10.0  # convert to inverse nm.
        Bstar = qc.transformSample2Lab(Bstar, params['mu'], params['om'], params['chi'], params['phi']) * 10.0
        Cstar = qc.transformSample2Lab(Cstar, params['mu'], params['om'], params['chi'], params['phi']) * 10.0

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


    @staticmethod
    def check_mandatory_params(params):
        """
        For the P10sixc diffractometer the data_dir, sample are mandatory parameters.

        :params: parameters needed to create detector
        :return: message indicating problem or empty message if all is ok
        """
        if  'data_dir' not in params:
            msg = 'data_dir parameter not configured, mandatory for P10sixc diffractometer.'
            raise ValueError(msg)
        data_dir = params['data_dir']
        if not os.path.isdir(data_dir):
            msg = f'data_dir directory {data_dir} does not exist.'
            raise ValueError(msg)

        if 'sample' not in params:
            msg = 'sample parameter not configured, mandatory for e4m detector.'
            raise ValueError(msg)


def create_diffractometer(diff_name, params):
    for diff in Diffractometer.__subclasses__():
        if diff.name == diff_name:
            return diff(params)
    msg = f'diffractometor {diff_name} not defined'
    raise ValueError(msg)


diffs = {'P10sixc' : Diffractometer_P10sixc}

def check_mandatory_params(diff_name, params):
    return diffs[diff_name].check_mandatory_params(params)
