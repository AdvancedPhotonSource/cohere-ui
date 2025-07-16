# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import cohere_ui.beamlines.esrf_id01.diffractometers as diff
import cohere_ui.beamlines.esrf_id01.detectors as det
import os


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

        # get needed parameters into one flat dict
        conf_params = conf_maps['config_instr']
        conf_params['binning'] = conf_maps['config_data'].get('binning', [1,1,1])
        return self.diff_obj.get_geometry(shape, scan, params)


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

    h5file = configs['config_instr'].get('h5file', None)
    if h5file is None:
        msg = 'h5file file must be provided to create Instrument for esrf_id01 beamline'
        raise ValueError(msg)
    # check if the file exist
    if not os.path.isfile(h5file):
        msg = f"h5file {h5file} does not exist"
        raise ValueError(msg)

    diffractometer = configs['config_instr'].get('diffractometer', None)
    if diffractometer is None:
        msg = 'diffractometer must be provided to create Instrument for esrf_id01 beamline'
        raise ValueError(msg)

    detector = configs['config_instr'].get('detector', None)
    if detector is None:
        msg = 'detector must be provided to create Instrument for esrf_id01 beamline'
        raise ValueError(msg)

    diff_obj = diff.create_diffractometer(diffractometer)

    if 'need_detector' in kwargs:
        det_obj = det.create_detector(detector, configs['config_prep'])

    instr = Instrument(h5file, diff_obj, det_obj, detector)

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
        instr.scan_ranges = scan_ranges

    return instr
