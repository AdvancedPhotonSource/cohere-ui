import beamlines.aps_1ide.diffractometers as diff
import beamlines.aps_1ide.detectors as det
from xrayutilities.io import spec
import re


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
        # print (str(ex))
        pass

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


    def get_geometry(self, shape, scan, **kwargs):
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
        xtal : boolean
            request only reciprocal space geometry when True
        The **kwargs reflect configuration, and could contain delta, gamma, theta, phi, chi, scanmot, scanmot_del,
        detdist, detector_name, energy.

        Returns
        -------
        tuple
            (Trecip, Tdir)
        """
        return self.diff_obj.get_geometry(shape, scan, **kwargs)


def create_instr(params):
    """
    Build factory for the Instrument class.

    Parameters
    ----------
    params : dict
        the parameters parsed from config file

    Returns
    -------
    (str, Object)
        error msg, Instrument object or None
    """
    det_obj = None
    diff_obj = None
    det_params = {}
    scan_ranges = None

    scan = params.get('scan', None)
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

        if 'specfile' in params:
            # detector name and roi is parsed from specfile if one exists
            # Find the first scan to parse detector params.
            first_scan = scan_ranges[0][0]
            det_params = parse_spec4roi(params.get('specfile'), first_scan)

   # override det_params with configured values in params
    det_params.update(params)
    det_name = det_params.get('detector', None)
    if det_name is not None:
        det_obj = det.create_detector(det_name, **det_params)
        if det_obj is None:
            return None
    diff_name = params.get('diffractometer', None)
    if diff_name is not None:
        diff_obj = diff.create_diffractometer(diff_name, specfile=params.get('specfile', None))
        if diff_obj is None:
            return None

    instr = Instrument(det_obj, diff_obj)
    instr.scan_ranges = scan_ranges

    return instr
