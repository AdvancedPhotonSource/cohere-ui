# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.config_verifier
===========================

Verification of configuration parameters.
"""
__author__ = "Dave Cyl"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['verify']


config_prep_error = {'File':['No configuration file',
                             'cannot read configuration file',
                             'Parsing error, check parenthesis,quotation syntax'],
                     'Roi':['roi parameter should be a list of int'],
                     'Datadir':['data_dir parameter should be string',
                                'missing mandatory parameter data_dir'],
                     'Darkfield':['darkfield_filename parameter should be string',
                                  'darkfield_filename parameter parsing error'],
                     'Whitefield':['whitefield_filename parameter should be string',
                                   'whitefield_filename parameter parsing error'],
                     'Excludescans':['exclude scans should be a list'],
                     'MinFiles':['min_frames should be int',
                                 'min_frames parameter parsing error']}
# config_disp_error = {'File':['No configuration file',
#                              'Cannot read configuration file',
#                              'Parsing error, check parenthesis,quotation syntax'],
#                      'Resultsdir':['results_dir parameter should be string'],
#                      'Crop':['crop should be list',
#                              'crop should be a list of int or float'],
#                      'Rampups':['rampups should be int']}
#
config_instr_error = { 'Diffractometer':['missing mandatory diffractometer parameter',
                                         'diffractometer parameter should be string'],
                       'Specfile': ['missing specfile parameter. The program will use detector configured in config_instr and roi in config_prep',
                                    'specfile parameter should be string',
                                    'specfile parameter parsing error'],
                       'Detector':['detector parameter should be string'],
                       'Energy':['energy should be float',
                                 'energy parameter parsing error'],
                       'Delta':['delta should be float',
                                'delta parameter parsing error'],
                       'Gamma':['gamma should be float',
                                'gamma parameter parsing error'],
                       'Detdist':['detdist should be float',
                                  'detdist parameter parsing error'],
                       'Dth':['dth should be float',
                              'dth parameter parsing error']
                    }

config_map_names = {'config_prep_error_map_file':config_prep_error,
                   # 'config_disp_error_map_file':config_disp_error,
                    'config_instr_error_map_file':config_instr_error}

def ver_list_int(param_name, param_value):
    """
    This function verifies if all elements in a given list are int.

    Parameters
    ----------
    param_name : str
        the parameter being evaluated

    param_value : list
        the list to evaluate for int values

    Returns
    -------
    eval : boolean
        True if all elements are int, False otherwise
    """
    if not issubclass(type(param_value), list):
        print(f'{param_name} is not a list')
        return False
    for e in param_value:
        if type(e) != int:
            print(f'{param_name} should be list of integer values')
            return False
    return True


def ver_list_float(param_name, param_value):
    """
    This function verifies if all elements in a given list are float.

    Parameters
    ----------
    param_name : str
        the parameter being evaluated

    param_value : list
        the list to evaluate for float values

    Returns
    -------
    eval : boolean
        True if all elements are float, False otherwise
    """
    if not issubclass(type(param_value), list):
        print(f'{param_name} is not a list')
        return False
    for e in param_value:
        if type(e) != float:
            print(f'{param_name} should be list of float values')
            return False
    return True


def get_config_error_message(config_file_name, map_file, config_parameter, config_error_no):
    """
    This function returns an error message string for this config file from the error map file using
    the parameter and error number as references for the error.

    :param config_file_name: The config file being verified
    :param map_file: The dictionary of error dictionary files
    :param config_parameter: The particular config file parameter being tested
    :param config_error_no: The error sequence in the test
    :return: An error string describing the error and where it was found
    """
    config_map_dic = config_map_names.get(map_file)

    error_string_message = config_map_dic.get(config_parameter)[config_error_no]
    # presented_message = "File=" + config_file_name, "Parameter=" + config_parameter, "Error=" + error_string_message

    return (error_string_message)


def ver_config_prep(config_map):
    """
    This function verifies experiment config_prep file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
    error_message : str
        message describing parameter error or empty string if all parameters are verified
    """
    config_map_file = 'config_prep_error_map_file'
    fname = 'config_prep'

    config_parameter = 'Roi'
    if 'roi' in config_map:
        if not ver_list_int('roi', config_map['roi']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print (error_message)
            return (error_message)

    config_parameter = 'Datadir'
    if 'data_dir' in config_map:
        data_dir = config_map['data_dir']
        if type(data_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return error_message
    else:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print (error_message)
        return (error_message)

    config_parameter = 'Darkfield'
    if 'darkfield_filename' in config_map:
        darkfield_filename = config_map['darkfield_filename']
        if type(darkfield_filename) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Whitefield'
    if 'whitefield_filename' in config_map:
        whitefield_filename = config_map['whitefield_filename']
        if type(whitefield_filename) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Excludescans'
    if 'exclude_scans' in config_map:
        if not ver_list_int('exclude_scans', config_map['exclude_scans']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'MinFiles'
    if 'min_frames' in config_map:
        min_frames = config_map['min_frames']
        if type(min_frames) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    return ("")

#
# def ver_config_disp(config_map):
#     """
#     This function verifies experiment config_disp file
#
#     Parameters
#     ----------
#     fname : str
#         configuration file name
#
#     Returns
#     -------
#     error_message : str
#         message describing parameter error or empty string if all parameters are verified
#     """
#     config_map_file = 'config_disp_error_map_file'
#     fname = 'config_disp'
#
#     config_parameter = 'Resultsdir'
#     if 'results_dir' in config_map:
#         results_dir = config_map['results_dir']
#         if type(results_dir) != str:
#             config_error = 0
#             error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
#             print('results_dir parameter should be string')
#             return (error_message)
#
#     config_parameter = 'Crop'
#     if 'crop' in config_map:
#         crop = config_map['crop']
#         if not issubclass(type(crop), list):
#             config_error = 0
#             error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
#             print('crop should be list')
#             return (error_message)
#         for e in crop:
#             if type(e) != int and type(e) != float:
#                 config_error = 1
#                 error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
#                 print('crop should be a list of int or float')
#                 return (error_message)
#
#     config_parameter = 'Rampups'
#     if 'rampups' in config_map:
#         rampups = config_map['rampups']
#         if type(rampups) != int:
#             config_error = 0
#             error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
#             print('rampups should be float')
#             return (error_message)
#
#     return ("")
#

def ver_config_instr(config_map):
    """
    This function verifies experiment config_disp file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
    error_message : str
        message describing parameter error or empty string if all parameters are verified
    """
    config_map_file = 'config_instr_error_map_file'
    fname = 'config_instr'

    config_parameter = 'Diffractometer'
    if 'diffractometer' in config_map:
        diffractometer = config_map['diffractometer']
        if type(diffractometer) != str:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
    else:
        config_error = 0
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print(error_message)
        return ''  # in some cases the parameters might be given

    config_parameter = 'Specfile'
    if 'specfile' in config_map:
        specfile = config_map['specfile']
        if type(specfile) != str:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('specfile parameter should be string')
            return (error_message)
    else:
        config_error = 0
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print(error_message)
        return ''

    config_parameter = 'Detector'
    if 'detector' in config_map:
        detector = config_map['detector']
        if type(detector) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Crop'
    if 'crop' in config_map:
        crop = config_map['crop']
        if not issubclass(type(crop), list):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        for e in crop:
            if type(e) != int and type(e) != float:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

    config_parameter = 'Energy'
    if 'energy' in config_map:
        energy = config_map['energy']
        if type(energy) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Delta'
    if 'delta' in config_map:
        delta = config_map['delta']
        if type(delta) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Gamma'
    if 'gamma' in config_map:
        gamma = config_map['gamma']
        if type(gamma) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Detdist'
    if 'detdist' in config_map:
        detdist = config_map['detdist']
        if type(detdist) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Dth'
    if 'dth' in config_map:
        dth = config_map['dth']
        if type(dth) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    return ("")

def verify(file_name, conf_map):
    """
    Verifies parameters.

    Parameters
    ----------
    file_name : str
        name of file the parameters are related to. Supported: config_prep, config_data, config_rec, config_disp

    conf_map : dict
        parameters

    Returns
    -------
    str
        a message with description of error or empty string if no error
    """
    if file_name == 'config_prep':
        return ver_config_prep(conf_map)
    # elif file_name == 'config_disp':
    #     return ver_config_disp(conf_map)
    elif file_name == 'config_instr':
        return ver_config_instr(conf_map)
    elif file_name == 'config_mp':
        return ''
    else:
        return ('verifier has no function to check config file named', file_name)
