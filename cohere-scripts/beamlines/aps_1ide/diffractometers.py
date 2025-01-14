import numpy as np
import math as m
import xrayutilities.experiment as xuexp
from xrayutilities.io import spec as spec
import beamlines.aps_1ide.detectors as det
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


class Diffractometer_1ide(Diffractometer):
    """
    Subclass of Diffractometer. Encapsulates "34idc" diffractometer.
    """
    name = "1ide"
    #sampleaxes = ('y+', 'z-', 'y+')  # in xrayutilities notation
    sampleaxes=('y-')  #omega is postive down
    detectoraxes=('z+','ty','tx')
    #detectoraxes = ('y+', 'x-')
    incidentaxis = (0, 0, 1)
    #motors from spec file.
    sampleaxes_name = ('AeroTech',)
    sampleaxes_mne = ('aero',)
    detectoraxes_name = ('vff_eta', 'vff_r')
    detectoraxes_mne = ('vff_eta', 'vff_r')
    #det dist will be in the config file.  Combination of dist to eta and x95 offset to back.

    def __init__(self, **kwargs):
        super(Diffractometer_1ide, self).__init__('1ide')
        self.specfile = kwargs.get('specfile')


    def parse_spec(self, scan):
        """
        Reads parameters from spec file for given scan.

        Parameters
        ----------
        scan : int
            scan number to use to recover the saved measurements

        Returns
        -------
        dict with delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist, detector_name, energy
        """
        spec_dict = {}

        # Scan numbers start at one but the list is 0 indexed
        try:
            ss = spec.SPECFile(self.specfile)[scan - 1]
        except Exception as ex:
            print(str(ex))
            print('Could not parse ' + self.specfile)
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
                print("failed from spec", mot_mne, mot_name)

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


    def get_geometry(self, shape, scan, **kwargs):
        """
        Calculates geometry based on diffractometer and detector attributes and experiment parameters.

        Typically, the delta, gamma, theta, phi, chi, scanmot, scanmot_del, detdist, detector_name,
        energy values are retrieved from experiment spec file.
        They can be overridden by configuration.

        :param shape: tuple, shape of array
        :param scan: scan the geometry is calculated for
        :param kwargs: The **kwargs reflect configuration, and could contain delta, gamma, theta, phi, chi, scanmot,
            scanmot_del, detdist, detector_name, energy.
        :return: tuple
            (Trecip, Tdir)
        """
        params = {}
        # parse spec
        if self.specfile is not None and scan is not None:
            params.update(self.parse_spec(scan))
        # override with config params
        params.update(kwargs)
        binning = params.get('binning', [1, 1, 1])
        pixel = det.get_pixel(params['detector'])
        px = pixel[0] * binning[0]
        py = pixel[1] * binning[1]

        detdist = params['detdist'] #/ 1000.0  # convert to meters
        scanmot = params['scanmot'].strip()
        enfix = 1
        # if energy is given in kev convert to ev for xrayutilities
        energy = params['energy']
        if m.floor(m.log10(energy)) < 3:
            enfix = 1000
        energy = energy * enfix  # x-ray energy in eV
        scanen = np.array((energy,))
        qc = xuexp.QConversion(self.sampleaxes, self.detectoraxes, self.incidentaxis, en=scanen)

        # compute for 4pixel (2x2) detector
        pixelorientation = det.get_pixel_orientation(params['detector'])
        qc.init_area(pixelorientation[0], pixelorientation[1], shape[0], shape[1], 2, 2,
                     distance=detdist, pwidth1=px, pwidth2=py)

        if scanmot in self.sampleaxes_mne:  # based on scanmot args are made for qc.area
            args = []
            for sampleax in self.sampleaxes_mne:
                if scanmot == sampleax:
                    scanstart = params[scanmot]
                    args.append(np.array((scanstart, scanstart + params['scanmot_del'] * binning[2])))
                else:
                    args.append(params[sampleax])
            args.append(params['vff_eta'])
            args.append(params['vff_r']/1000+params['vff_r_offset'])
            args.append(params['vff_eta_offset'])

            q2 = np.array(qc.area(*args, deg=True))
        else:
            print("scanmot not in sample axes or energy, exiting")
            raise RuntimeError

        # I think q2 will always be (3,2,2,2) (vec, scanarr, px, py)
        Astar = q2[:, 0, 1, 0] - q2[:, 0, 0, 0]
        Bstar = q2[:, 0, 0, 1] - q2[:, 0, 0, 0]
        Cstar = q2[:, 1, 0, 0] - q2[:, 0, 0, 0]

        xtal = kwargs.get('xtal', False)
        if xtal:
            Trecip_cryst = np.zeros(9)
            Trecip_cryst.shape = (3, 3)
            Trecip_cryst[:, 0] = Astar * 10
            Trecip_cryst[:, 1] = Bstar * 10
            Trecip_cryst[:, 2] = Cstar * 10
            return Trecip_cryst, None

        # transform to lab coords from sample reference frame
        Astar = qc.transformSample2Lab(Astar, params['aero']) * 10.0  # convert to inverse nm.
        Bstar = qc.transformSample2Lab(Bstar, params['aero']) * 10.0
        Cstar = qc.transformSample2Lab(Cstar, params['aero']) * 10.0

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


def create_diffractometer(diff_name, **kwargs):
    if diff_name == '1ide':
        d = Diffractometer_1ide(**kwargs)
        return d
    else:
        print (f'diffractometer {diff_name} not defined.')
        return None