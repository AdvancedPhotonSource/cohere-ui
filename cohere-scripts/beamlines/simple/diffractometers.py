import numpy as np
import math as m
import xrayutilities.experiment as xuexp
import beamlines.aps_34idc.detectors as det
from abc import ABC


class Diffractometer(ABC):
    """
    Abstract class representing diffractometer. It keeps fields related to the specific diffractometer represented by
    a subclass.

    diff_name : str
        diffractometer name
    """
    name = None

    def __init__(self, diff_name):
        """
        Constructor.

        Parameters
        ----------
        diff_name : str
            diffractometer name

        """
        self.diff_name = diff_name


class Default(Diffractometer):
    """
    Subclass of Diffractometer. Encapsulates any diffractometer. is based on 34-idc.
    """
    name = "default"
    sampleaxes = ('y+', 'z-', 'y+')  # in xrayutilities notation
    detectoraxes = ('y+', 'x-')
    incidentaxis = (0, 0, 1)
    sampleaxes_name = ('Theta', 'Chi', 'Phi')
    sampleaxes_mne = ('th', 'chi', 'phi')
    detectoraxes_name = ('Delta', 'Gamma')
    detectoraxes_mne = ('delta', 'gamma')
    detectordist_name = 'camdist'
    detectordist_mne = 'detdist'

    def __init__(self):
        super(Default, self).__init__('default')


    def get_geometry(self, shape, scan, det_obj, **kwargs):
        """
        Calculates geometry based on diffractometer and detector attributes and experiment parameters.

        Typically the delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist, detector_name,
        energy values are retrieved from experiment records (spec file or hdf5 file).
        They can be overridden by configuration.
        The mneminic used in this example is may vary by beamline.

        :param shape: tuple
            shape of reconstructed array
        :param scan: int
            scan number the geometry is calculated for
        :param det_obj: Object
            detector object or None
        :param kwargs: The **kwargs reflect configuration, and could contain delta, gamma, theta, phi, chi, scanmot,
            scanmot_del, detdist, detector_name, energy.
        :return: tuple
            (Trecip, Tdir)
        """
        # At the beginning the parameters would be parsed either from spec file or read from hdf5 file, or
        # can be read from some data base if this apply to the beamline.
        # It is assumed here that all parameters are from kwargs.
        attrs = kwargs
        binning = kwargs.get('binning', [1, 1, 1])

        # set the attributes with values parsed from spec and then possibly overridden by configuration
        for attr in attrs:
            setattr(self, attr, attrs[attr])

        if det_obj is None:
            det_obj = det.create_detector(attrs.get('detector'))
        px = det_obj.pixel[0] * binning[0]
        py = det_obj.pixel[1] * binning[1]

        detdist = attrs.get('detdist') / 1000.0  # convert to meters
        scanmot = attrs.get('scanmot').strip()
        enfix = 1
        # if energy is given in kev convert to ev for xrayutilities
        if m.floor(m.log10(attrs.get('energy'))) < 3:
            enfix = 1000
        energy = attrs.get('energy') * enfix  # x-ray energy in eV

        if scanmot == 'en':
            scanen = np.array((energy, energy + attrs.get('scanmot_del') * enfix))
        else:
            scanen = np.array((energy,))
        qc = xuexp.QConversion(self.sampleaxes, self.detectoraxes, self.incidentaxis, en=scanen)

        # compute for 4pixel (2x2) detector
        qc.init_area(det_obj.pixelorientation[0], det_obj.pixelorientation[1], shape[0], shape[1], 2, 2,
                     distance=detdist, pwidth1=px, pwidth2=py)

        # I think q2 will always be (3,2,2,2) (vec, scanarr, px, py)
        # should put some try except around this in case something goes wrong.
        if scanmot == 'en':  # seems en scans always have to be treated differently since init is unique
            q2 = np.array(qc.area(attrs.get('th'), attrs.get('chi'), attrs.get('phi'), attrs.get('delta'),
                                  attrs.get('gamma'), deg=True))
        elif scanmot in self.sampleaxes_mne:  # based on scanmot args are made for qc.area
            args = []
            axisindex = self.sampleaxes_mne.index(scanmot)
            for n in range(len(self.sampleaxes_mne)):
                if n == axisindex:
                    scanstart = getattr(self, scanmot)
                    args.append(np.array((scanstart, scanstart + attrs.get('scanmot_del') * binning[2])))
                else:
                    args.append(self.__dict__[self.sampleaxes_mne[n]])
            for axis in self.detectoraxes_mne:
                args.append(getattr(self, axis))
            q2 = np.array(qc.area(*args, deg=True))
        else:
            print("scanmot not in sample axes or energy, exiting")
            raise RuntimeError

        # I think q2 will always be (3,2,2,2) (vec, scanarr, px, py)
        Astar = q2[:, 0, 1, 0] - q2[:, 0, 0, 0]
        Bstar = q2[:, 0, 0, 1] - q2[:, 0, 0, 0]
        Cstar = q2[:, 1, 0, 0] - q2[:, 0, 0, 0]

        # transform to lab coords from sample reference frame
        Astar = qc.transformSample2Lab(Astar, self.th, self.chi, self.phi) * 10.0  # convert to inverse nm.
        Bstar = qc.transformSample2Lab(Bstar, self.th, self.chi, self.phi) * 10.0
        Cstar = qc.transformSample2Lab(Cstar, self.th, self.chi, self.phi) * 10.0

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


def create_diffractometer(diff_name):
    if diff_name == 'default':
        d = Default()
        return d
    else:
        print (f'diffractometer {diff_name} not defined.')
        return None