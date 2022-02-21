import numpy as np
from cohere.beamlines.viz import CXDViz
import math as m
import xrayutilities.experiment as xuexp
from xrayutilities.io import spec as spec

def parse_spec(specfile, scan):
    """
    Reads parameters necessary to run visualization from spec file for given scan.
    Parameters
    ----------
    specfile : str
        spec file name
         
    scan : int
        scan number to use to recover the saved measurements
    Returns
    -------
    delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist, detector_name, energy
    """
    # Scan numbers start at one but the list is 0 indexed
    try:
        ss = spec.SPECFile(specfile)[scan - 1]
    except  Exception as ex:
        print(str(ex))
        print ('Could not parse ' + specfile )
        return None,None,None,None,None,None,None,None,None,None

    # Stuff from the header
    try:
        detector_name = str(ss.getheader_element('UIMDET'))
    except:
        detector_name = None
    try:
        command = ss.command.split()
        scanmot = command[1]
        scanmot_del = (float(command[3]) - float(command[2])) / int(command[4])
    except:
        scanmot = None
        scanmot_del = None

    # Motor stuff from the header
    try:
        delta = ss.init_motor_pos['INIT_MOPO_Delta']
    except:
        delta = None
    try:
        gamma = ss.init_motor_pos['INIT_MOPO_Gamma']
    except:
        gamma = None
    try:
        theta = ss.init_motor_pos['INIT_MOPO_Theta']
    except:
        theta = None
    try:
        phi = ss.init_motor_pos['INIT_MOPO_Phi']
    except:
        phi = None
    try:
        chi = ss.init_motor_pos['INIT_MOPO_Chi']
    except:
        chi = None
    try:
        detdist = ss.init_motor_pos['INIT_MOPO_camdist']
    except:
        detdist = None
    try:
        energy = ss.init_motor_pos['INIT_MOPO_Energy']
    except:
        energy = None

    # returning the scan motor name as well.  Sometimes we scan things
    # other than theta.  So we need to expand the capability of the display
    # code.
    return delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist, detector_name, energy


class DispalyParams:
    """
      This class encapsulates parameters defining image display. The parameters are read from config file on construction.
      This class is basically an information agglomerator for the viz generation.
    """

    def __init__(self, config):
        """
        The constructor gets config file and fills out the class members.

        Parameters
        ----------
        config : str
            configuration file name

        Returns
        -------
        none
        """
        self.detector = None
        deg2rad = np.pi / 180.0
        if 'specfile' in config and 'last_scan' in config:
            specfile = config['specfile']
            last_scan = config['last_scan']
            # get stuff from the spec file.
            self.delta, self.gamma, self.th, self.phi, self.chi, self.scanmot, self.scanmot_del, self.detdist, self.detector, self.energy = parse_spec(specfile, last_scan)
        # drop the ':' from detector name
        if self.detector is not None and self.detector.endswith(':'):
            self.detector = self.detector[:-1]

        if 'diffractometer' in config:
            self.diffractometer = config['diffractometer']
        else:
            print('diffractometer name not in config file')

        # override the parsed parameters with entries in config file
        if 'detector' in config:
            self.detector = config['detector']
        else:
            if self.detector is None:
                raise ValueError('detector not in spec, please configure')
        if 'energy' in config:
            self.energy = config['energy']
        else:
            if self.energy is None:
                raise ValueError('energy not in spec, please configure')
        if 'delta' in config:
            self.delta = config['delta']
        else:
            if self.delta is None:
                raise ValueError('delta not in spec, please configure')
        if 'gamma' in config:
            self.gamma = config['gamma']
        else:
            if self.gamma is None:
                raise ValueError('gamma not in spec, please configure')
        if 'detdist' in config:
            self.detdist = config['detdist']
        else:
            if self.detdist is None:
                raise ValueError('detdist not in spec, please configure')
        if 'theta' in config:
            self.th = config['theta']
        else:
            if self.th is None:
                raise ValueError('theta not in spec, please configure')
        if 'chi' in config:
            self.chi = config['chi']
        else:
            if self.chi is None:
                raise ValueError('chi not in spec, please configure')
        if 'phi' in config:
            self.phi = config['phi']
        else:
            if self.phi is None:
                raise ValueError('phi not in spec, please configure')
        if 'scanmot' in config:
            self.scanmot = config['scanmot']
        else:
            if self.scanmot is None:
                raise ValueError('scanmot not in spec, please configure')
        if 'scanmot_del' in config:
            self.scanmot_del = config['scanmot_del']
        else:
            if self.scanmot_del is None:
                raise ValueError('scanmot_del not in spec, please configure')

        if 'rampups' in config:
            self.rampups = config['rampups']
        else:
            self.rampups = 1

        if 'binning' in config:
            self.binning = []
            binning = list(config['binning'])
            for i in range(len(binning)):
                self.binning.append(binning[i])
            for _ in range(3 - len(self.binning)):
                self.binning.append(1)
        else:
            self.binning = [1, 1, 1]
        if 'crop' in config:
            self.crop = []
            crop = list(config['crop'])
            for i in range(len(crop)):
                if crop[i] > 1:
                    crop[i] = 1.0
                self.crop.append(crop[i])
            for _ in range(3 - len(self.crop)):
                self.crop.append(1.0)
            crop[0], crop[1] = crop[1], crop[0]
        else:
            self.crop = (1.0, 1.0, 1.0)


    def set_instruments(self, detector, diffractometer):
        # for beamline aps_34idc both detector and diffractometer must be defined
        if detector is None:
            print ('detector must be defined')
            return False
        if diffractometer is None:
            print ('diffractometer must be defined')
            return False

        for attr in diffractometer.__class__.__dict__.keys():
            if not attr.startswith('__'):
                self.__dict__[attr] = diffractometer.__class__.__dict__[attr]
        for attr in diffractometer.__dict__.keys():
            if not attr.startswith('__'):
                self.__dict__[attr] = diffractometer.__dict__[attr]

        for attr in detector.__class__.__dict__.keys():
            if not attr.startswith('__'):
                self.__dict__[attr] = detector.__class__.__dict__[attr]
        for attr in detector.__dict__.keys():
            if not attr.startswith('__'):
                self.__dict__[attr] = detector.__dict__[attr]

        return True


def set_geometry(shape, p):
    """
    Sets geometry.

    Parameters
    ----------
    shape : tuple
        shape of reconstructed array

    p : DisplayParmas object

    Returns
    -------
    nothing
    """
    # DisplayParams is not expected to do any modifications of params (units, etc)
    px = p.pixel[0] * p.binning[0]
    py = p.pixel[1] * p.binning[1]
    detdist = p.detdist / 1000.0  # convert to meters
    scanmot = p.scanmot.strip()
    enfix = 1
    # if energy is given in kev convert to ev for xrayutilities
    if m.floor(m.log10(p.energy)) < 3:
        enfix = 1000
    energy = p.energy * enfix  # x-ray energy in eV

    if scanmot == 'en':
        scanen = np.array((energy, energy + p.scanmot_del * enfix))
    else:
        scanen = np.array((energy,))
    qc = xuexp.QConversion(p.sampleaxes, p.detectoraxes, p.incidentaxis, en=scanen)

    # compute for 4pixel (2x2) detector
    qc.init_area(p.pixelorientation[0], p.pixelorientation[1], shape[0], shape[1], 2, 2, distance=detdist,
                      pwidth1=px, pwidth2=py)
    # I think q2 will always be (3,2,2,2) (vec, scanarr, px, py)
    # should put some try except around this in case something goes wrong.
    if scanmot == 'en':  # seems en scans always have to be treated differently since init is unique
        q2 = np.array(qc.area(p.th, p.chi, p.phi, p.delta, p.gamma, deg=True))
    elif scanmot in p.sampleaxes_name:  # based on scanmot args are made for qc.area
        args = []
        axisindex = p.sampleaxes_name.index(scanmot)
        for n in range(len(p.sampleaxes_name)):
            if n == axisindex:
                scanstart = p.__dict__[scanmot]
                args.append(np.array((scanstart, scanstart + p.scanmot_del * p.binning[2])))
            else:
                args.append(p.__dict__[p.sampleaxes_name[n]])
        for axis in p.detectoraxes_name:
            args.append(p.__dict__[axis])
        q2 = np.array(qc.area(*args, deg=True))
    else:
        print("scanmot not in sample axes or energy, exiting")
        raise RuntimeError

    # I think q2 will always be (3,2,2,2) (vec, scanarr, px, py)
    Astar = q2[:, 0, 1, 0] - q2[:, 0, 0, 0]
    Bstar = q2[:, 0, 0, 1] - q2[:, 0, 0, 0]
    Cstar = q2[:, 1, 0, 0] - q2[:, 0, 0, 0]

    # transform to lab coords from sample reference frame
    Astar = qc.transformSample2Lab(Astar, p.th, p.chi, p.phi) * 10.0  # convert to inverse nm.
    Bstar = qc.transformSample2Lab(Bstar, p.th, p.chi, p.phi) * 10.0
    Cstar = qc.transformSample2Lab(Cstar, p.th, p.chi, p.phi) * 10.0

    denom = np.dot(Astar, np.cross(Bstar, Cstar))
    A = 2 * m.pi * np.cross(Bstar, Cstar) / denom
    B = 2 * m.pi * np.cross(Cstar, Astar) / denom
    C = 2 * m.pi * np.cross(Astar, Bstar) / denom

    Trecip = np.zeros(9)
    Trecip.shape = (3, 3)
    Trecip[:, 0] = Astar
    Trecip[:, 1] = Bstar
    Trecip[:, 2] = Cstar

    Tdir = np.zeros(9)
    Tdir.shape = (3, 3)
    Tdir = np.array((A, B, C)).transpose()

    return (Trecip, Tdir)
