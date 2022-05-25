from cohere import Diffractometer


class Diffractometer_34idc(Diffractometer):
    """
    Subclass of Diffractometer. Encapsulates "34idc" diffractometer.
    """
    name = "34idc"
    sampleaxes = ('y+', 'z-', 'y+')  # in xrayutilities notation
    detectoraxes = ('y+', 'x-')
    incidentaxis = (0, 0, 1)
    sampleaxes_name = ('th', 'chi', 'phi')  # using the spec mnemonics for scan id.
    detectoraxes_name = ('delta', 'gamma')

    def __init__(self):
        super(Diffractometer_34idc, self).__init__('34idc')


def create_diffractometer(diff_name):
    if diff_name == '34idc':
        return Diffractometer_34idc()
    else:
        print ('diffractometer ' + diff_name + ' not defined.')


def verify_diffractometer(diff_name):
    if diff_name == '34idc':
        return True
    else:
        return False


