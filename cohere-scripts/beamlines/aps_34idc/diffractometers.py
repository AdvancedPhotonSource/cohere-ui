import numpy as np
import math as m
import xrayutilities.experiment as xuexp
from xrayutilities.io import spec as spec
import beamlines.aps_34idc.detectors as det
from abc import ABC, abstractmethod



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


class Diffractometer_34idc(Diffractometer):
    """
    Subclass of Diffractometer. Encapsulates "34idc" diffractometer.
    """
    name = "34idc"
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
        super(Diffractometer_34idc, self).__init__('34idc')

    def parse_spec(self, specfile, scan):
        """
        Reads parameters from spec file for given scan.

        Parameters
        ----------
        specfile : str
            spec file name

        scan : int
            scan number to use to recover the saved measurements

        diff : object
            diffractometer object

        Returns
        -------
        dict with delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist, detector_name, energy
        """
        spec_dict = {}

        # Scan numbers start at one but the list is 0 indexed
        try:
            ss = spec.SPECFile(specfile)[scan - 1]
        except Exception as ex:
            print(str(ex))
            print('Could not parse ' + specfile)
            return None

        try:
            command = ss.command.split()
            spec_dict['scanmot'] = command[1]
            spec_dict['scanmot_del'] = (float(command[3]) - float(command[2])) / int(command[4])
        except:
            pass

        for mot_mne, mot_name in zip(self.sampleaxes_mne + self.detectoraxes_mne,
                                     self.sampleaxes_name + self.detectoraxes_name):
            try:
                motname = "INIT_MOPO_{m}".format(m=mot_name)
                spec_dict[mot_mne] = ss.init_motor_pos[motname]
            except:
                pass

        try:
            motname = "INIT_MOPO_{m}".format(m=self.detectordist_name)
            spec_dict['detdist'] = ss.init_motor_pos[motname]
        except:
            pass

        try:
            spec_dict['energy'] = ss.init_motor_pos['INIT_MOPO_Energy']
        except:
            pass

        try:
            spec_dict['detector'] = str(ss.getheader_element('UIMDET'))
            if spec_dict['detector'].endswith(':'):
                spec_dict['detector'] = spec_dict['detector'][:-1]
        except Exception as ex:
            print(str(ex))

        return spec_dict


    def get_geometry(self, shape, scan, specfile, xtal, det_obj, **kwargs):
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
            scan number the geometry is calculated for
        specfile : str
            specfile name
        xtal : boolean
            a switch
        binning : list
            binning
        det_obj : Object
            detector object
        The **kwargs reflect configuration, and could contain delta, gamma, theta, phi, chi, scanmot, scanmot_del,
        detdist, detector_name, energy.

        Returns
        -------
        tuple
            (Trecip, Tdir)
        """
        attrs = self.parse_spec(specfile, scan)
        attrs.update(kwargs)
        binning = attrs.get('binning', [1, 1, 1])

        # set the attributes with values parsed from spec and then possibly overridden by configuration
        for attr in attrs:
            setattr(self, attr, attrs[attr])

        if det_obj is None:
            det_obj = det.create_detector(self.detector)
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

        if xtal:
            Trecip_cryst = np.zeros(9)
            Trecip_cryst.shape = (3, 3)
            Trecip_cryst[:, 0] = Astar * 10
            Trecip_cryst[:, 1] = Bstar * 10
            Trecip_cryst[:, 2] = Cstar * 10
            return Trecip_cryst, None

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
    if diff_name is None:
        print('diffractometer name not provided')
        return None
    if diff_name == '34idc':
        d = Diffractometer_34idc()
        return d
    else:
        print (f'diffractometer {diff_name} not defined.')
        return None