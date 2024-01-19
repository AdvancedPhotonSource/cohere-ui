class Diffractometer(object):
    """
    class Diffractometer(self, diff_name)
    ============================================

    Abstract class representing diffractometer. It keeps fields related to the specific diffractometer represented by a subclass.
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

    def __init__(self):
        super(Diffractometer_34idc, self).__init__('34idc')


def create_diffractometer(diff_name):
    if diff_name == '34idc':
        return Diffractometer_34idc()
    else:
        print (f'diffractometer {diff_name} not defined.')


def verify_diffractometer(diff_name):
    if diff_name == '34idc':
        return True
    else:
        return False


