# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import os
import cohere_ui.beamlines.Petra3_P10.diffractometers as diff
import cohere_ui.beamlines.Petra3_P10.detectors as det
import cohere_ui.beamlines.Petra3_P10.p10_scan_reader as p10sr
import cohere_core.utilities as ut


class Instrument:
    """
      This class encapsulates istruments: diffractometer and detector used for that experiment.
      It provides interface to get the classes encapsulating the diffractometer and detector.
    """

    def __init__(self, det_obj, diff_obj):
        """
        Constructor

        :param det_obj: detector object, can be None
        :param diff_obj: diffractometer object, can be None
        """
        self.det_obj = det_obj
        self.diff_obj = diff_obj


    def datainfo4scans(self):
        """
        Finds existing sub-directories in data_dir that correspond to given scans and scan ranges.
        Parameters
        ----------
        Returns
        -------
        list
        """
        return self.det_obj.dirs4scans(self.scan_ranges)


    def get_scan_array(self, scan_dir):
        return self.det_obj.get_scan_array(scan_dir)


    def get_geometry(self, shape, scan, conf_maps):
        """
        Calculates geometry based on diffractometer's and detctor's attributes and experiment parameters.

        For the Petra3_P10 typically the delta, gamma, theta, phi, chi, scanmot, scanmot_del,
        detdist, detector_name, energy values are parsed from fio file.
        They can be overridden by configuration.

        Parameters
        ----------
        :param  : tuple
            shape of reconstructed array
        :param  : int
            scan for which the geometry applies
        :param  : conf_params
            parameters typically parsed from config file, other

        :return: tuple of arrays containing geometry in reciprocal space and direct space
            (Trecip, Tdir)
        """
        if self.diff_obj is None:
            raise RuntimeError

        # get needed parameters into one flat dict
        conf_params = conf_maps['config_instr']
        conf_params['binning'] = conf_maps['config_data'].get('binning', [1,1,1])
        return self.diff_obj.get_geometry(shape, scan, conf_params)


def create_instr(configs, **kwargs):
    """
    Build factory for the Instrument class.

    Parameters
    ----------
    configs : dict of dicts
        the parameters parsed from config files

    Returns
    -------
    (str, Object)
        error msg, Instrument object or None
    """
    det_obj = None
    diff_obj = None
    scan_ranges = None
    # set parameters from config_instr
    config_params = configs['config_instr']

    scan = configs['config'].get('scan', None)
    if scan is not None:
        # 'scan' is configured as string. It can be a single scan, range, or combination separated by comma.
        # Parse the scan into list of scan ranges, defined by starting scan, and ending scan, inclusive.
        # The single scan has range defined as the same starting and ending scan.
        scan_ranges = []
        scan_units = [u for u in scan.replace(' ','').split(',')]
        for u in scan_units:
            if '-' in u:
                r = u.split('-')
                scan_ranges.append([int(r[0]), int(r[1])])
            else:
                scan_ranges.append([int(u), int(u)])

    det_name = config_params.get('detector', None)
    if det_name is None and scan is not None:
        # try to parse detector name
        # Find the first scan to parse detector params.
        first_scan = scan_ranges[0][0]
        # the directories for Petra are structured as follows: <data_dir>/<sample>scan
        # check here if that directory exist
        data_dir = config_params['data_dir']
        scan_subdir = config_params['sample'] + '_{:05d}'.format(int(scan))
        if not os.path.isdir(ut.join(data_dir, scan_subdir)):
            msg = "cannot parse det_name, the data/sample path does not exist"
            raise ValueError(msg)

        scanmeta = p10sr.P10Scan(config_params.get('data_dir'), config_params.get('sample'), first_scan, pathsave='', creat_save_folder=True)
        det_name = scanmeta.get_motor_pos('_ccd')

    if det_name is None:
        msg = 'detector name not configured and could not be parsed'
        raise ValueError(msg)

    diff_name = config_params.get('diffractometer', None)
    if diff_name is None:
        msg = 'diffractometer parameter not defined'
        raise ValueError(msg)
    else:
        diff.check_mandatory_params(diff_name, config_params)
        diff_obj = diff.create_diffractometer(diff_name, config_params)

    if 'need_detector' in kwargs and kwargs['need_detector']:
        # add parameters from the config_prep
        config_params.update(configs['config_prep'])
        # check for parameters
        det.check_mandatory_params(det_name, config_params)
        det_obj = det.create_detector(det_name, config_params)

    instr = Instrument(det_obj, diff_obj)
    instr.scan_ranges = scan_ranges

    return instr
