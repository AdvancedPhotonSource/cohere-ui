import beamlines.Petra3_P10.diffractometers as diff
import beamlines.Petra3_P10.detectors as det
import re
import beamlines.Petra3_P10.p10_scan_reader as p10sr


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

        For the Petra3_P10 typically the delta, gamma, theta, phi, chi, scanmot, scanmot_del,
        detdist, detector_name, energy values are parsed from fio file.
        They can be overridden by configuration.

        Parameters
        ----------
        :param  : tuple
            shape of reconstructed array
        :param  : int
            scan for which the geometry applies
        :param  : kwargs
            parameters typically parsed from config file, other

        :return: tuple of arrays containing geometry in reciprocal space and direct space
            (Trecip, Tdir)
        """
        if self.diff_obj is None:
            raise RuntimeError

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

    det_name = params.pop('detector', None)
    if det_name is None and 'scan' in params.keys():
        # try to parse detector name
        # Find the first scan to parse detector params.
        first_scan = scan_ranges[0][0]
        scanmeta = p10sr.P10Scan(params.get('data_dir'), params.get('sample'), first_scan, pathsave='', creat_save_folder=True)
        det_name = scanmeta.get_motor_pos('_ccd')
    if det_name is not None:
        det_obj = det.create_detector(det_name, **params)

    diff_name = params.get('diffractometer', None)
    if diff_name is not None:
        diff_obj = diff.create_diffractometer(diff_name, data_dir=params.get('data_dir', None), sample=params.get('sample', None))
        if diff_obj is None:
            return None
    instr = Instrument(det_obj, diff_obj)
    instr.scan_ranges = scan_ranges

    return instr
