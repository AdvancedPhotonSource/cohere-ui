import cohere_ui.beamlines.simple.diffractometers as diff
import cohere_ui.beamlines.simple.detectors as det


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


    def datainfo4scans(self):
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
            scans: [[2825, 2831], [2834, 2834], [2840, 2846]]
            returns: [[(2825, f'{path}/data_S2825'), (2828, f'{path}/data_S2828'), (2831, f'{path}/data_S2831')],
             [(2834, f'{path}/data_S2834)],
             [(2840, f'{path}/data_S2840'), (2843, f'{path}/data_S2843'), (2846, f'{path}/data_S2846')]]

        :return:
        list of sub-lists, each sublist containing tuples with the input scans and corresponding data info
         within scan ranges.
        """
        # The detector function is typically renamed to reflect the info.
        # if the info is directory, the function name would be dirs4scans
        # if the info is hdf5 file node, the function name would be nodes4scans
        if self.det_obj is None:
            print('detector object not created, check config parameters')
        return self.det_obj.datainfo4scans(self.scan_ranges)


    def get_scan_array(self, scan):
        """
        Gets the data for the scan. The data is corrected for the detector.

        :param scan_info: info allowing detector to retrieve data for a scan
        :return: corrected data array
        """
        return self.det_obj.get_scan_array(scan)


    def get_geometry(self, shape, scan, conf_params):
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
        :param  : conf_params
            parameters parsed from config file

        :return: tuple of arrays containing geometry in reciprocal space and direct space
            (Trecip, Tdir)
        """
        return self.diff_obj.get_geometry(shape, scan, conf_params)


def create_instr(configs, **kwargs):
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
    instr_config_params = configs['config_instr']
    det_name = instr_config_params.get('detector', None)
    if det_name is None:
        raise ValueError('detector name not configured in config_instr')

    if 'need_detector' in kwargs and kwargs['need_detector']:
        if 'config_prep' not in configs:
            msg = 'missing config_prep'
            raise ValueError(msg)
        det_params = instr_config_params
        det_params.update(configs['config_prep'])

        det_obj = det.create_detector(det_name, det_params)

    diff_name = instr_config_params.get('diffractometer', None)
    if diff_name is None:
        msg = 'detector name not configured in config_instr'
        raise ValueError(msg)

    diff_obj = diff.create_diffractometer(diff_name)

    instr = Instrument(det_obj, diff_obj)
    main_conf = configs['config']
    if 'scan' in main_conf:
        instr.scan_ranges = [[int(main_conf['scan']), int(main_conf['scan'])]]

    return instr
