from xrayutilities.io import spec as spec

def parse_spec(params, specfile, scan):
    """
    Reads parameters necessary to run visualization from spec file for given scan.
    Parameters
    ----------
    specfile : str
        spec file name

    scan : int
        scan number to use to recover the saved measurements

    params_values : list
        list of parameters to find values

    Returns
    -------
    dict
        dictionary of parameters; name : value
    """
    params_values = {}
    # Scan numbers start at one but the list is 0 indexed
    try:
        ss = spec.SPECFile(specfile)[scan - 1]
    except  Exception as ex:
        print(str(ex))
        print('Could not parse ' + specfile)
        return params_values

    # Stuff from the header
    if 'detector_name' in params:
        try:
            params_values['detector_name'] = str(ss.getheader_element('UIMDET'))
            if params_values['detector_name'].endswith(':'):
                params_values['detector_name'] = params_values['detector_name'][:-1]

        except:
            pass

    if 'scanmot' in params or 'scanmot_del' in params:
        try:
            command = ss.command.split()
            params_values['scanmot'] = command[1]
            params_values['scanmot_del'] = (float(command[3]) - float(command[2])) / int(command[4])
        except:
            pass

    # Motor stuff from the header
    if 'delta' in params:
        try:
            params_values['delta'] = ss.init_motor_pos['INIT_MOPO_Delta']
        except:
            pass
    if 'gamma' in params:
        try:
            params_values['gamma'] = ss.init_motor_pos['INIT_MOPO_Gamma']
        except:
            pass
    if 'theta' in params:
        try:
            params_values['theta'] = ss.init_motor_pos['INIT_MOPO_Theta']
        except:
            pass
    if 'phi' in params:
        try:
            params_values['phi'] = ss.init_motor_pos['INIT_MOPO_Phi']
        except:
            pass
    if 'chi' in params:
        try:
            params_values['chi'] = ss.init_motor_pos['INIT_MOPO_Chi']
        except:
            pass
    if 'detdist' in params:
        try:
            params_values['detdist'] = ss.init_motor_pos['INIT_MOPO_camdist']
        except:
            pass
    if 'energy' in params:
        try:
            params_values['energy'] = ss.init_motor_pos['INIT_MOPO_Energy']
        except:
            pass

    return params_values

def set_spec_attrs(obj, attr_list, specfile, last_scan):
    """
    This function fills out the class members of obj instance with values parsed from spec.
    Parameters
    ----------
    config : str
        configuration file name
    Returns
    -------
    none
    """
    # get stuff from the spec file.
    attrs = parse_spec(attr_list, specfile, last_scan)
    for attr in attrs:
        if attr in attrs.keys():
            setattr(obj, attr, attrs[attr])
        else:
            setattr(obj, attr, None)
