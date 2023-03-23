import numpy as np
import math as m
import xrayutilities.experiment as xuexp
from xrayutilities.io import spec as spec
import beamlines.aps_34idc.diffractometers as diff
import beamlines.aps_34idc.detectors as det


def parse_spec(specfile, scan, diff):
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
        spec_dict['detector'] = str(ss.getheader_element('UIMDET'))
        if spec_dict['detector'].endswith(':'):
            spec_dict['detector'] = spec_dict['detector'][:-1]
    except:
        pass

    try:
        command = ss.command.split()
        spec_dict['scanmot'] = command[1]
        spec_dict['scanmot_del'] = (float(command[3]) - float(command[2])) / int(command[4])
    except:
        pass

    for mot_mne, mot_name in zip(diff.sampleaxes_mne + diff.detectoraxes_mne, diff.sampleaxes_name + diff.detectoraxes_name):
        try:
            motname = "INIT_MOPO_{m}".format(m=mot_name)
            spec_dict[mot_mne] = ss.init_motor_pos[motname]
        except:
            pass

    try:
        motname = "INIT_MOPO_{m}".format(m=diff.detectordist_name)
        spec_dict['detdist'] = ss.init_motor_pos[motname]
    except:
        pass

    try:
        spec_dict['energy'] = ss.init_motor_pos['INIT_MOPO_Energy']
    except:
        pass

    try:
        spec_dict['det_area'] = [int(n) for n in ss.getheader_element('UIMR5').split()]
    except:
        pass

    return spec_dict


class Instrument:
    """
      This class encapsulates parameters defining experiment instruments and parameters defining geometry.
      It contains diffractometer attributes, detector attributes and parameters parsed from spec file. The
      parsed parameters will be overridden with configured parameters in config_instr file.
    """

    def initialize(self, config):
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
        if 'diffractometer' in config:
            try:
                self.diff_obj = diff.create_diffractometer(config['diffractometer'])
            except:
                return('cannot create diffractometer', config['diffractometer'])
        else:
            return('diffractometer name not in config file')

        if not 'specfile' in config or not 'last_scan' in config:
            return('missing spec file or last_scan')

        if 'binning' in config:
            self.binning = config['binning']
        else:
            self.binning = [1, 1, 1]

        specfile = config['specfile']
        last_scan = config['last_scan']
        attrs = parse_spec(specfile, last_scan, self.diff_obj)

        # set the attributes with values parsed from spec
        for attr in attrs:
            setattr(self, attr, attrs[attr])

        # override the parsed parameters with entries in config file
        for attr in config:
            setattr(self, attr, config[attr])

        self.det_obj = det.create_detector(self.detector)
        if self.det_obj is None:
            return 'detector ' + self.detector + ' not defined in detectors.py file.'

        return ''


    def get_geometry(self, shape, xtal=False):
        """
        Calculates geometry based on diffractometer's attributes and experiment parameters.

        Parameters
        ----------
        shape : tuple
            shape of reconstructed array

        Returns
        -------
        tuple
            (Trecip, Tdir)
        """
        px = self.det_obj.pixel[0] * self.binning[0]
        py = self.det_obj.pixel[1] * self.binning[1]

        detdist = self.detdist / 1000.0  # convert to meters
        scanmot = self.scanmot.strip()
        enfix = 1
        # if energy is given in kev convert to ev for xrayutilities
        if m.floor(m.log10(self.energy)) < 3:
            enfix = 1000
        energy = self.energy * enfix  # x-ray energy in eV

        if scanmot == 'en':
            scanen = np.array((energy, energy + self.scanmot_del * enfix))
        else:
            scanen = np.array((energy,))
        qc = xuexp.QConversion(self.diff_obj.sampleaxes, self.diff_obj.detectoraxes, self.diff_obj.incidentaxis, en=scanen)

        # compute for 4pixel (2x2) detector
        qc.init_area(self.det_obj.pixelorientation[0], self.det_obj.pixelorientation[1], shape[0], shape[1], 2, 2,
                     distance=detdist, pwidth1=px, pwidth2=py)
        # I think q2 will always be (3,2,2,2) (vec, scanarr, px, py)
        # should put some try except around this in case something goes wrong.
        if scanmot == 'en':  # seems en scans always have to be treated differently since init is unique
            q2 = np.array(qc.area(self.th, self.chi, self.phi, self.delta, self.gamma, deg=True))
        elif scanmot in self.diff_obj.sampleaxes_mne:  # based on scanmot args are made for qc.area
            args = []
            axisindex = self.diff_obj.sampleaxes_mne.index(scanmot)
            for n in range(len(self.diff_obj.sampleaxes_mne)):
                if n == axisindex:
                    scanstart = getattr(self, scanmot)
                    args.append(np.array((scanstart, scanstart + self.scanmot_del * self.binning[2])))
                else:
                    args.append(self.__dict__[self.diff_obj.sampleaxes_mne[n]])
            for axis in self.diff_obj.detectoraxes_mne:
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