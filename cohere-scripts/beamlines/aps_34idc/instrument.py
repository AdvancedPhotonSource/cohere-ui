import beamlines.aps_34idc.diffractometers as diff
import beamlines.aps_34idc.detectors as det
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
    params_values = {}
    # Scan numbers start at one but the list is 0 indexed, so we subtract 1
    try:
        ss = spec.SPECFile(specfile)[scan - 1]
    except Exception as ex:
        print(str(ex))
        print('Could not parse ' + specfile)
        return params_values

    try:
        params_values['detector'] = str(ss.getheader_element('UIMDET'))
        if params_values['detector'].endswith(':'):
            params_values['detector'] = params_values['detector'][:-1]
    except Exception as ex:
        print(str(ex))

    try:
        params_values['roi'] = [int(n) for n in ss.getheader_element('UIMR5').split()]
    except Exception as ex:
        print (str(ex))

    return params_values


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
        (self.specfile, diffractometer) = args
        self.diff_obj = diff.create_diffractometer(diffractometer)
        self.det_obj = None


    def init_detector(self, *args, **kwargs):
       # the detector is parsed from specfile, and therefore scan number must be given
        # parse the frame size (roi) at the same time
        (scan,) = args
        det_pars = parse_spec4roi(self.specfile, scan)

        det_name = kwargs.pop('detector', None)
        if det_name is None:
            det_name = det_pars.pop('detector', None)

        if det_name is None:
            return 'detector name unknown'

        kwargs.update(det_pars)

        self.det_obj = det.create_detector(det_name, **kwargs)
        if self.det_obj is None:
            raise RuntimeError


    def datainfo4scans(self, scans):
        """
        Finds existing sub-directories in data_dir that correspond to given scans and scan ranges.
        Parameters
        ----------
        scans : list
            list of tuples defining scan(s) and scan range(s), ordered

        Returns
        -------
        list
        """
        return self.det_obj.dirs4scans(scans)


    def get_scan_array(self, scan_dir):
        return self.det_obj.get_scan_array(scan_dir)


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

        kwargs.pop('specfile', None)
        return self.diff_obj.get_geometry(shape, scan, self.specfile, xtal, self.det_obj, **kwargs)


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
    specfile = params.get('specfile', None)
    if specfile is None:
        print ('spec file must be provided to create Instrument for aps-34idc beamline')
        return None
    diffractometer = params.get('diffractometer', None)
    if diffractometer is None:
        print ('diffractometer must be provided to create Instrument for aps-34idc beamline')
        return None
    instr = Instrument(specfile, diffractometer)

    if 'scan' in params:
        # This is executed when preprocessing
        # Find first scan to set detector. Pass preprocessor configuration (conf_prep) parameters to init the detector.
        first_scan = int(re.search(r'\d+', params.get('scan')).group())
        instr.init_detector(first_scan, **params)

    return instr
