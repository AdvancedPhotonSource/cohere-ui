import os.path

import beamlines.simple.diffractometers as diff
import beamlines.simple.detectors as det
import re


class Instrument:
    """
      This class encapsulates istruments: diffractometer and detector used for that experiment.

      A detector class contains interfaces to retrieve data captured by the detector. A correction
      specific to the detector may be used.
      A diffractometer class provides interface to obtain the geometry that was in effect during the experiment.
      The geometry allows visualization of reconstructed object.
    """

    def __init__(self, det_obj, diff_obj):
        """
        Constructor

        :param det_obj: detector object, can be None
        :param diff_obj: diffractometer object, can be None
        """
        self.det_obj = det_obj
        self.diff_obj = diff_obj


    def datainfo4scans(self, scans):
        """
        Finds info allowing to read data that correspond to given scans or scan ranges.
        The info can be directories where the data related to scans is stored or nodes in hd5 file
        that contain the data, or other info specific to a beamline.

        :param scans : list
            list of sub-lists defining scan ranges, ordered. For single scan a range has the same scan as beginning and end.
            one scan example:
            scans : [[2834, 2834]]
            returns : [[(2834, f'{path}/data_S2834)]]

            separate ranges example:
            ex1: [[2825, 2831], [2834, 2834], [2840, 2876]]
            returns: [[(2825, f'{path}/data_S2825'), (2828, f'{path}/data_S2828'), (2831, f'{path}/data_S2831')],
             [(2834, f'{path}/data_S2834)],
             [(2840, f'{path}/data_S2840'), (2843, f'{path}/data_S2843'), (2846, f'{path}/data_S2846'), (2849, f'{path}/data_S2849'),
              (2852, f'{path}/data_S2852'), (2855, f'{path}/data_S2855'), (2858, f'{path}/data_S2858'), (2861, f'{path}/data_S2861'),
              (2864, f'{path}/data_S2864'), (2867, f'{path}/data_S2867'), (2870, f'{path}/data_S2870'), (2873, f'{path}/data_S2873'),
              (2876, f'{path}/data_S2876')]]

        :return:
        list of sub-lists the input scans, or scans ranges with the corresponding info
        """
        # The detector function is typically renamed to reflect the info.
        # if the info is directory, the function name would be dirs4scans
        # if the info is hdf5 file node, the function name would be nodes4scans
        return self.det_obj.datainfo4scans(scans)


    def get_scan_array(self, scan_info):
        """
        Gets the data for the scan. The data is corrected for the detector.

        :param scan_info: info allowing detector to retrieve data for a scan
        :return: corrected data array
        """
        return self.det_obj.get_scan_array(scan_info)


    def get_geometry(self, shape, scan, **kwargs):
        """
        Calculates geometry based on diffractometer's and detctor's attributes and experiment parameters.

        Geometry may be different by scan and depends on array shape.
        The parameters needed for geometry calculation can be parsed by some mechanism (from spec or from
        hdf5 file or other).
        Another way to pass parameters is through kwargs.

        The parameters can include for example delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist,
        detector_name, energy, wave length.

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
        return self.diff_obj.get_geometry(shape, scan, self.det_obj, **kwargs)


def create_instr(params):
    """
    Build factory for the Instrument class.

    :param : dict
        the parameters typically parsed from config file

    Returns
    -------
    Object or None
        Instrument object or None
    """
    det_obj = None
    diff_obj = None
    det_name = params.get('detector', None)
    if det_name is not None:
        det_obj = det.create_detector(det_name, **params)
        if det_obj is None:
            return None
    diff_name = params.get('diffractometer', None)
    if diff_name is not None:
        diff_obj = diff.create_diffractometer(diff_name)
        if diff_obj is None:
            return None

    instr = Instrument(det_obj, diff_obj)

    return instr
