import numpy as np
import math as m
import h5py
import beamlines.esrf_id01.diffractometers as diff
import beamlines.esrf_id01.detectors as det



class Instrument:
    """
      This class encapsulates parameters defining experiment instruments and parameters defining geometry.
      It contains diffractometer attributes, detector attributes and parameters parsed from h5 file. The
      parsed parameters will be overridden with configured parameters in config_instr file.
    """

    def initialize(self, config, scan):
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
        # The calling code ensures diffractometer and h5file are configured
        self.diff_obj = diff.create_diffractometer(config['diffractometer'])
        self.h5file = config['h5file']

        self.binning = config.get('binning', [1, 1, 1])

        # set the attributes with values parsed from h5
        for key, val in parse_h5(self.h5file, scan, self.diff_obj).items():
            setattr(self, key, val)

        # save the instrument parameters from configuration
        # and override the parsed parameters with entries in config file
        # Note: the multipeak and separate scans configuration will not
        # include the h5 parsed parameters
        for attr in config:
            setattr(self, attr, config[attr])

        self.det_obj = det.create_detector(self.detector)


class Instrument:
    """
      This class encapsulates istruments: diffractometer and detector used for that experiment.
      It provides interface to get the classes encapsulating the diffractometer and detector.
    """

    def __init__(self, *args):
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
        (self.h5file, diffractometer, detector) = args
        self.diff_obj = diff.create_diffractometer(diffractometer)
        self.det_obj = det.create_detector(detector)
        self.detector = detector


    def datainfo4scans(self, scans):
        """
        Finds nodes in hdf5 file that correspond to given scans and scan ranges.
        Parameters
        ----------
        scans : list
            list of tuples defining scan(s) and scan range(s), ordered

        Returns
        -------
        list
        """
        return self.det_obj.nodes4scans(scans, self.h5file)


    def get_scan_array(self, scan_node):
        return self.det_obj.get_scan_array(scan_node, self.h5file)


    def get_geometry(self, shape, scan, xtal=False, **kwargs):
        """
        Calculates geometry based on diffractometer's and detctor's attributes and experiment parameters.

        For the aps_34idc typically the delta, gamma, theta, phi, chi, scanmot, scanmot_del,
        detdist, detector_name, energy values are parsed from spec file.
        They can be overridden by configuration.

        Parameters
        ----------
        shape : tuple
            shape of reconstructed array
        The *args for aps_34idc contain scan number.
        The **kwargs reflect configuration, and could contain delta, gamma, theta, phi, chi, scanmot, scanmot_del,
        detdist, detector_name, energy.

        Returns
        -------
        tuple
            (Trecip, Tdir)
        """
        if self.diff_obj is None:
            raise RuntimeError

        return self.diff_obj.get_geometry(shape, scan, self.h5file, xtal, self.detector)


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
    h5file = params.get('h5file', None)
    if h5file is None:
        print ('h5file file must be provided to create Instrument for esrf_id01 beamline')
        return None
    diffractometer = params.get('diffractometer', None)
    if diffractometer is None:
        print ('diffractometer must be provided to create Instrument for esrf_id01 beamline')
        return None
    detector = params.get('detector', None)
    if detector is None:
        print ('detector must be provided to create Instrument for esrf_id01 beamline')
        return None
    instr = Instrument(h5file, diffractometer, detector)

    return instr

