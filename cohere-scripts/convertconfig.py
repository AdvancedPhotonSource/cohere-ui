import sys
import os
import argparse

# version must be increased after each modification of configuration file(s)
version = 0

# Map file of before/after keys to remap
config_map = {'beamline': 'beamline', 'specfile': 'specfile'}
config_prep_map = {'darkfile': 'darkfield_filename', 'whitefile': 'whitefield_filename'}
config_rec_map = {'samples': 'reconstructions', 'beta' : 'hio_beta', 'amp_support_trigger': 'shrink_wrap_trigger',
                  'support_type': 'shrink_wrap_type', 'support_threshold' : 'shrink_wrap_threshold',
                  'support_sigma' : 'shrink_wrap_gauss_sigma', 'support_area' : 'initial_support_area',
                  'pcdi_trigger' : 'pc_interval', 'pc_trigger' : 'pc_interval', 'partial_coherence_type' : 'pc_type',
                  'partial_coherence_iteration_num' : 'pc_LUCY_iterations', 'partial_coherence_normalize' : 'pc_normalize',
                  'partial_coherence_roi' : 'pc_LUCY_kernel', 'phase_min' : 'phm_phase_min', 'phase_max' : 'phm_phase_max',
                  'iter_res_sigma_range' : 'lowpass_filter_sw_sigma_range', 'iter_res_det_range' : 'lowpass_filter_range',
                  'generations' : 'ga_generations', 'ga_support_thresholds' : 'ga_shrink_wrap_thresholds',
                  'ga_support_sigmas' : 'ga_shrink_wrap_gauss_sigmas', 'ga_low_resolution_sigmas' : 'ga_lowpass_filter_sigmas',
                  'gen_pcdi_start' : 'ga_gen_pc_start'}
config_disp_map = {'arm': 'detdist', 'dth': 'scanmot_del'}
config_data_map = {'aliens': 'aliens', 'amp_threshold' : 'intensity_threshold'}

beamlinedefaultvalue = ' "aps_34idc"'

config_maps = {'config': config_map,
               'config_prep': config_prep_map,
               'config_rec': config_rec_map,
               'config_disp': config_disp_map,
               'config_data': config_data_map}


def get_version():
    """
    Returns current version of this script. The version is an integer number and it must be updated after
    each modification of the script.

    Parameters
    ----------
    data : ndarray
        an array with experiment data
    config : Object
        configuration object providing access to configuration parameters
    data_dir : str
        a directory where 'alien_analysis' subdirectory will be created to save results of analysis if configured
    Returns
    -------
    data : ndarray
        data array without aliens
    """
    return version


def versionfile(file_spec, vtype='copy'):
    # Prior to any change make a backup of the original file
    import os
    import shutil

    file_spec = file_spec.replace(os.sep, '/')
    if os.path.isfile(file_spec):
        # or, do other error checking:
        if vtype not in ('copy', 'rename'):
            vtype = 'copy'

        # Determine root filename so the extension doesn't get longer
        n, e = os.path.splitext(file_spec)
        # Is e an integer?
        try:
            num = int(e)
            root = n
        except ValueError:
            root = file_spec + "_backup"
            shutil.copy(file_spec, root)
            return 0


def returnconfigdictionary(cfile_path, cfile):
    """
    This function takes a config file name and creates a dictionary of all the parameters defined in the config
    file using the = to split key,value pairs.
    There is a special case where values can be enclosed in () with a line for each value.
    Parameters
    ----------
    cfile : str
        configuration file name
    startdir : str
        a directory with configuration files to be converted
    Returns
    -------
    currentdic : dict
        a dictionary with the configuration parameters
    """
    param = ''
    value = ''
    extend = False
    currentdic = {}

    cfile_path = cfile_path.replace(os.sep, '/')
    cfile = cfile.replace(os.sep, '/')
    # Check if file exists, if it does make a backup copy of file with the new name config_backup
    if os.path.exists(cfile_path):
        if os.access(os.path.dirname(cfile_path), os.W_OK):
            backupfile = cfile + "_backup"
            versionfile(cfile_path)
    else:
        return currentdic

    input = open(cfile_path, 'r')
    str = input.readline()
    while str:
        # Ignore comment lines and move along
        str = str.rstrip()
        if '//' in str:
            str = input.readline()
            continue
        elif extend == False:
            # Test if = is in the string otherwise skip this line, if = in line split on = only if this is not an
            # extended string seperated by the () brackets
            if "=" not in str:
#                print("strange string in config=", str)
                str = input.readline()
                continue
            param, value = str.split('=')
            param = param.rstrip()
            currentdic[param] = value
        else:
            #  in the extended string area capture the string and check if at begining or end
            value = value + str
        if value == " (":
            extend = True
        elif str == ")":
            # At the end of extended string push string into dictionary and end loop
            currentdic[param] = value
            extend = False
        str = input.readline()
    input.close()
    return currentdic


def replace_keys(dic, cfile):
    for (k, v) in config_maps[cfile].items():
        if k in dic.keys():
            # Key is in the dictionary, replace with new value
            holdingvalue = dic.pop(k)
            dic[v] = holdingvalue
        else:
            # Key is not in the dictionary, do nothing
            continue
    return dic


def convert_dict(conf_dict, cfile):
    if cfile == 'config':
        mappingfile = config_map
    elif cfile == 'config_prep':
        mappingfile = config_prep_map
    elif cfile == 'config_data':
        mappingfile = config_data_map
    elif cfile == 'config_rec':
        mappingfile = config_rec_map
    elif cfile == 'config_disp':
        mappingfile = config_disp_map
    # Check if key is in the dictionary for this config file

    for (k, v) in mappingfile.items():
        if k in conf_dict.keys():
            # Key is in the dictionary, replace with new value
            holdingvalue = conf_dict.pop(k)
            conf_dict[v] = holdingvalue
#            print("The value of", k, "has been changed to", v, "and the value is", holdingvalue)
        else:
            # Key is not in the dictionary, do nothing
            continue

    # Some special cases if beamline has no value then it was never defined so set it to the default
    # if specfile is in config_prep then remove it a add it to config
    # if the config_data file has the older aliens format of coordinates/file then update to new layout
    if cfile == 'config':
        if 'beamline' in conf_dict.keys():
            beamlinevalue = conf_dict['beamline']
        else:
            beamlinevalue = beamlinedefaultvalue
            conf_dict['beamline'] = beamlinevalue
            conf_dict['converter_ver'] = get_version()
    # if specfile is in config_prep move it to config with the same value
    elif cfile == 'config_prep':
        if 'specfile' in conf_dict.keys():
            savedspecfilevalue = conf_dict.pop('specfile')
            conf_dict['specfile'] = savedspecfilevalue
    # Look to see if aliens is set and if it is a directory or a block of coordinates
    elif cfile == 'config_data':
        # if alien_alg is defined then this is current and no change is needed.
        if 'alien_alg' in conf_dict.keys():
            pass
        elif 'aliens' in conf_dict.keys():
            savedAlien = conf_dict.pop('aliens')
            if "(" in savedAlien:
                conf_dict['alien_alg'] = ' "block_aliens"'
                conf_dict['aliens'] = savedAlien
            else:
                conf_dict['alien_alg'] = ' "alien_file"'
                conf_dict['alien_file'] = savedAlien
    elif cfile == 'config_rec':
        import ast
        def add_iter(el, s):
            if len(el) == 2:
                s = s + str(el[0] * el[1][1]) + '*' + el[1][0]
            elif len(el) > 2:
                s = s + str(el[0]) + '*('
                for i in range(1, len(el)):
                    s = s + str(el[i][1]) + '*' + el[i][0]
                    if i == len(el) - 1:
                        last_char = ')'
                    else:
                        last_char = '+'
                    s = s + last_char
            return s

        alg_seq = conf_dict['algorithm_sequence'].replace(' ','')
        if alg_seq.startswith('('):    # old format
            s = '"'
            alg_seq = ast.literal_eval(alg_seq)
            for i in range(len(alg_seq)):
                s = add_iter(alg_seq[i], s)
                if i < len(alg_seq)-1:
                    s = s + '+'
            s = s + '"'
            conf_dict['algorithm_sequence'] = s

        pc_interval = conf_dict['pc_interval'].replace(' ','')
        if not pc_interval.isnumeric():
            pc_interval = ast.literal_eval(pc_interval)[1]
        conf_dict['pc_interval'] = str(pc_interval)

    return conf_dict


def get_conf_dict(cfile_path, cfile):
    """
    This function takes a config file name and creates a dictionary of all the parameters defined in the config
    file using the = to split key,value pairs.
    There is a special case where values can be enclosed in () with a line for each value.
    Parameters
    ----------
    cfile : str
        configuration file name
    startdir : str
        a directory with configuration files to be converted
    Returns
    -------
    currentdic : dict
        a dictionary with the configuration parameters
    """
    cfile_path = cfile_path.replace(os.sep, '/')
    cdict = returnconfigdictionary(cfile_path, cfile)
    cdict = replace_keys(cdict, cfile)
    cdict = convert_dict(cdict, cfile)

    return cdict


def convert(startdir):
    """
    This script will convert old config files to the newer format using the following critera

    Config file

    Move specfile from config_prep
    Add beamline
    Add/replace converter_ver

    config_prep

    change darkfile to darkfield_filename
    change whitefile to whitefield_filename

    config_rec

    changed samples to reconstructions
    changed amp_support_trigger to shrink_wrap_trigger
    changed support_type to shrink_wrap_type
    +

    config_disp

    changed arm to detdist
    changed dth to theta

    The path where the config files are located is one of the parameters passed to the script via the -p variable,
    new files and backup files are written back to the same path as supplied.

    Prior to any work, a backup of the original file is made to file_backup

    Newly mapped files then replace the original file with the new parameters which are then mapped to the original
    values.

    How it works: Each config file in the list config_file_names is read in and a dictionary of the file parameters
    (keys) is assigned a value. All config files are read into one large
    dictionary of dictionaries. Each config file key,value pairs is then compared with the map file associated with that
    config file using the config_map_tuple to steer the mapping.
    If a match is found the value of the key replaces the original key. The remapped key,value pair is then written out
    to the same file name thereby replacing the original file.

    There are special cases that needed to be added to this general mapping strategy, If the config file had the
    "beamline" parameter already defined then keep its value if not then I
    picked a default value of aps_sec34id-c. If the specfile parameter was found in config_prep it was moved over to the
    config file.
    All non-relevant data in the files are stripped out including spaces and comments.
    If an aliens parameter is found id the config_data
    file it is tested to see if it is a file or a set of coordinates and the alien_alg is set to the correct type.

    Parameters
    ----------
    startdir : str
        a directory with configuration files to be converted
    Returns
    -------
    nothing
    """
    startdir = startdir.replace(os.sep, '/')
    # First check to see if directory exists, if not then exit
    if not os.path.exists(startdir):
        # there is nothing to convert
        return

    conf_files = []
    for cfile in config_maps.keys():
        # check if file exist
        if not os.path.isfile(startdir +'/' + cfile):
            continue
        conf_files.append(cfile)

        # Now go into the content of each config  file and create a dictionary of line items to work with
        thisdic = returnconfigdictionary(startdir + '/' + cfile, cfile)

        # Use map file to see what items need to change
        # Use the map file to determine what parameters need to be changed.
        # if the key is found then do the remapp, if not then skip.
        converted_dict = replace_keys(thisdic, cfile)

        # Some special cases if beamline has no value then it was never defined so set it to the default
        # if specfile is in config_prep then remove it a add it to config
        # if the config_data file has the older aliens format of coordinates/file then update to new layout
        converted_dict = convert_dict(converted_dict, cfile)

        # Write the data out to the same-named file
        writepath = startdir + '/' + cfile
        fileobj = open(writepath, 'w')
        for subkey in converted_dict.keys():
            string2write = subkey + " =" + converted_dict[subkey] + '\n'
            fileobj.write(string2write)
        fileobj.close()


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("conv_dir", help="path to directory with configuration files that will be converted")
    args = parser.parse_args()
    convert(args.conv_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
