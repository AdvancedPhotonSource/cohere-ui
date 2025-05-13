import beamlines.esrf_id01.diffractometers as diff
import beamlines.esrf_id01.detectors as det
import os
import cohere_core.utilities as ut


class Instrument:
    """
      This class encapsulates istruments: diffractometer and detector used for that experiment.
      It provides interface to get the classes encapsulating the diffractometer and detector.
    """

    def __init__(self, h5file, diff_obj, det_obj, detector):
        """
        The constructor.

        Parameters
        ----------
        params : dict
            <param name> : <param value>

        Returns
        -------
        str
            a string containing error message or empty
        """
        self.h5file = h5file
        self.diff_obj = diff_obj
        self.det_obj = det_obj
        self.detector = detector


    def datainfo4scans(self):
        """
        Finds nodes in hdf5 file that correspond to given scans and scan ranges.
        Parameters
        ----------
        Returns
        -------
        list
        """
        return self.det_obj.nodes4scans(self.scan_ranges)


    def get_scan_array(self, scan_node):
        return self.det_obj.get_scan_array(scan_node, self.h5file)


    def get_geometry(self, shape, scan, params):
        """
        Calculates geometry based on diffractometer's and detctor's attributes and experiment parameters.

        Parameters
        ----------
        shape : tuple
            shape of reconstructed array
        scan : scan number for which the geometry is calculated
        params : reflect configuration

        Returns
        -------
        tuple
            (Trecip, Tdir)
        """
        if self.diff_obj is None:
            raise RuntimeError

        return self.diff_obj.get_geometry(shape, scan, params)


def create_instr(params, **kwargs):
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

    h5file = params.get('h5file', None)
    if h5file is None:
        raise ValueError('h5file file must be provided to create Instrument for esrf_id01 beamline')
    # check if the file exist
    if not os.path.isfile(h5file):
        raise ValueError(f'h5file {h5file} does not exist.')

    diffractometer = params.get('diffractometer', None)
    if diffractometer is None:
        raise ValueError('diffractometer must be provided to create Instrument for esrf_id01 beamline')

    detector = params.get('detector', None)
    if detector is None:
        raise ValueError('detector must be provided to create Instrument for esrf_id01 beamline')

    diff_obj = diff.create_diffractometer(diffractometer)
    if diff_obj is None:
        raise ValueError(f'failed create diffractometer {diffractometer}')

    if 'need_detector' in kwargs:
        roi = params.get('roi', None)
        det_obj = det.create_detector(detector, roi=roi)
        if det_obj is None:
            raise ValueError(f'failed create detector {detector}')

    instr = Instrument(h5file, diff_obj, det_obj, detector)

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
        instr.scan_ranges = scan_ranges

    return instr
