# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import cohere_ui.beamlines.aps_1ide.diffractometers as diff
import cohere_ui.beamlines.aps_1ide.detectors as det


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

        For the aps_34idc typically the delta, gamma, theta, phi, chi, scanmot, scanmot_del,
        detdist, detector_name, energy values are parsed from spec file.
        They can be overridden by configuration.

        Parameters
        ----------
        shape : tuple
            shape of reconstructed array
        scan : int
            scan to use to parse experiment parameters
        conf_params : dict
            reflect configuration, and can contain values of diffractometer parameters at the specific scan.

        Returns
        -------
        tuple
            (Trecip, Tdir)
        """
        # get needed parameters into one flat dict
        conf_params = conf_maps['config_instr']
        conf_params['binning'] = conf_maps['config_data'].get('binning', [1,1,1])
        return self.diff_obj.get_geometry(shape, scan, conf_params)


def create_instr(configs, **kwargs):
    """
    Build factory for the Instrument class.

    Parameters
    ----------
    configs : dict
        the parameters parsed from config file

    Returns
    -------
    (str, Object)
        error msg, Instrument object or None
    """
    det_obj = None
    diff_obj = None
    main_config_params = configs['config']

    det_name = configs['config_instr'].get('detector', None)
    if det_name is None:
        raise ValueError('detector name not configured and could not be parsed')

    if  'need_detector' in kwargs and kwargs['need_detector']:
        if 'config_prep' not in configs:
            raise ValueError('missing config_prep, required for beamline aps34-idc')
        # set detector parameters to configured parameters in config_prep
        det_params = configs['config_prep']
        # check for parameters, it will raise exception if not success
        det.check_mandatory_params(det_name, det_params)
        det_obj = det.create_detector(det_name, det_params)

    diff_name = configs['config_instr'].get('diffractometer', None)
    if diff_name is None:
        raise ValueError('diffractometer parameter not defined')
    else:
        diff_obj = diff.create_diffractometer(diff_name, configs['config_instr'])

    instr = Instrument(det_obj, diff_obj)
    # set scan ranges in instrument class
    scan_ranges = None
    scan = main_config_params.get('scan', None)
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
    instr.scan_ranges = scan_ranges

    return instr
