# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import cohere_ui.beamlines.aps_34idc.diffractometers as diff
import cohere_ui.beamlines.aps_34idc.detectors as det
from xrayutilities.io import spec


def parse_spec4roi(specfile, scan):
    """
    Returns detector name and detector area parsed from spec file for given scan.

    Parameters
    ----------
    specfile : str
        spec file name

    scan : int
        scan number to use to recover the saved measurements

    Returns
    -------
    dict
        dictionary of parameters; name : value
    """
    params = {}
    # Scan numbers start at one but the list is 0 indexed, so we subtract 1
    try:
        ss = spec.SPECFile(specfile)[scan - 1]
    except Exception as ex:
        print(str(ex))
        print('Could not parse ' + specfile)
        return params

    try:
        params['detector'] = str(ss.getheader_element('UIMDET'))
        if params['detector'].endswith(':'):
            params['detector'] = params['detector'][:-1]
    except Exception as ex:
        print(str(ex))

    try:
        params['roi'] = [int(n) for n in ss.getheader_element('UIMR5').split()]
    except Exception as ex:
        print (str(ex))

    return params


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


    def get_geometry(self, shape, scan, conf_maps, **kwargs):
        """
        Calculates geometry based on diffractometer's and detctor's attributes and experiment parameters.

        For the aps_34idc typically the delta, gamma, theta, phi, chi, scanmot, scanmot_del,
        detdist, detector_name, energy values are parsed from spec file.
        They can be overridden by configuration.

        Parameters
        ----------
        shape : tuple
            shape of reconstructed array
        scan : int or None
            scan to use to parse experiment parameters
        conf_params : configuration parameters, can contain delta, gamma, theta, phi, chi, scanmot, scanmot_del,
        detdist, detector_name, energy.
        kwargs:
            xtal : boolean
                request only reciprocal space geometry when True

        Returns
        -------
        tuple
            (Trecip, Tdir)
        """
        # get needed parameters into one flat dict
        conf_params = conf_maps['config_instr']
        conf_params['binning'] = conf_maps['config_data'].get('binning', [1,1,1])
        return self.diff_obj.get_geometry(shape, scan, conf_params, **kwargs)


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
    det_params = {}
    scan_ranges = None

    main_config_params = configs['config']
    instr_config_params = configs['config_instr']

    scan = main_config_params.get('scan', None)
    if 'config_mp' in configs:
        scan = configs['config_mp'].get('scan', None)
    if scan is not None:
        # 'scan' is configured as string. It can be a single scan, range, or combination separated by comma.
        # Parse the scan into list of scan ranges, defined by starting scan, and ending scan, inclusive.
        # The single scan has range defined as the same starting and ending scan.
        scan_ranges = []
        scan_units = [u for u in scan.replace(' ', '').split(',')]
        for u in scan_units:
            if '-' in u:
                r = u.split('-')
                scan_ranges.append([int(r[0]), int(r[1])])
            else:
                scan_ranges.append([int(u), int(u)])

        if 'specfile' in instr_config_params:
            # detector name and roi is parsed from specfile if one exists
            # Find the first scan to parse detector params.
            first_scan = scan_ranges[0][0]
            det_params = parse_spec4roi(instr_config_params.get('specfile'), first_scan)

    # override det_params with configured values in params
    det_params.update(instr_config_params)
    det_name = det_params.get('detector', None)
    if det_name is None:
        raise ValueError('detector name not configured and could not be parsed')
    if 'need_detector' in kwargs and kwargs['need_detector']:
        # add parameters from config_prep to det_params
        if 'config_prep' not in configs:
            msg = 'missing config_prep, required for beamline aps34-idc'
            raise ValueError(msg)
        det_params.update(configs['config_prep'])
        # check for parameters, it will raise exception if failed
        det.check_mandatory_params(det_name, det_params)

        det_obj = det.create_detector(det_name, det_params)
        if det_obj is None:
            msg = f'failed create {det_name} detector'
            raise ValueError(msg)

    diff_name = instr_config_params.get('diffractometer', None)
    if diff_name is None:
        msg = 'diffractometer parameter not defined'
        raise ValueError(msg)
    else:
        diff_obj = diff.create_diffractometer(diff_name, instr_config_params)
        if diff_obj is None:
            msg = f'failed create {diff_name} diffractometer'
            raise ValueError(msg)

    instr = Instrument(det_obj, diff_obj)
    instr.scan_ranges = scan_ranges

    return instr
