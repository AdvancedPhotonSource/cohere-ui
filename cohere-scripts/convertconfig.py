import sys
import os
import argparse
import shutil
import util.util as ut

# version must be increased after each modification of configuration file(s)
version = 1

# Map file of before/after keys to remap
config_map = {}
config_prep_map = {'darkfile': 'darkfield_filename',
                   'whitefile': 'whitefield_filename'}
config_rec_map = {'samples': 'reconstructions',
                  'beta': 'hio_beta',
                  'amp_support_trigger': 'shrink_wrap_trigger',
                  'support_type': 'shrink_wrap_type',
                  'support_threshold': 'shrink_wrap_threshold',
                  'support_sigma': 'shrink_wrap_gauss_sigma',
                  'support_area': 'initial_support_area',
                  'pcdi_trigger': 'pc_interval',
                  'pc_trigger' : 'pc_interval',
                  'partial_coherence_type' : 'pc_type',
                  'partial_coherence_iteration_num' : 'pc_LUCY_iterations',
                  'partial_coherence_normalize' : 'pc_normalize',
                  'partial_coherence_roi' : 'pc_LUCY_kernel',
                  'phase_min' : 'phm_phase_min',
                  'phase_max' : 'phm_phase_max',
                  'phase_support_trigger' : 'phm_trigger',
                  'resolution_trigger' : 'lowpass_filter_trigger',
                  'iter_res_det_range' : 'lowpass_filter_range',
                  'generations' : 'ga_generations',
                  'ga_support_thresholds' : 'ga_sw_thresholds',
                  'ga_support_sigmas' : 'ga_sw_gauss_sigmas',
                  'ga_low_resolution_sigmas' : 'ga_lowpass_filter_sigmas',
                  'gen_pcdi_start' : 'ga_gen_pc_start'}
config_disp_map = {'arm': 'detdist',
                   'dth': 'scanmot_del'}
config_data_map = {'aliens': 'aliens',
                   'amp_threshold' : 'intensity_threshold'}
config_instr_map = {}
config_mp_map = {}

beamlinedefaultvalue = '"aps_34idc"'

config_maps = {'config': config_map,
               'config_prep': config_prep_map,
               'config_rec': config_rec_map,
               'config_disp': config_disp_map,
               'config_data': config_data_map,
               'config_instr': config_instr_map,
               'config_mp': config_mp_map}

# the key is the configuration file parameters are removed from
# the parameters in list are inserted into the configuration file of subdict key
move_dict = {'config':{'config_instr':['specfile']},
             'config_prep':{'config':['separate_scans', 'separate_scan_ranges'],
                            'config_instr':['specfile']},
             'config_disp':{'config_instr':['energy', 'delta', 'gamma', 'detdist', 'th', 'chi', 'phi', 'scanmot',
                                            'scanmot_del', 'detector', 'diffractometer']}}

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


def versionfile(file_spec):
    # Prior to any change make a backup of the original file
    import os
    import shutil

    file_spec = file_spec.replace(os.sep, '/')
    if os.path.isfile(file_spec):
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


def convert_dict(conf_dicts, prev_ver=0):
    if 'config' in conf_dicts.keys():
        conf_dict = conf_dicts['config']
        if not 'beamline' in conf_dict.keys():
            conf_dict['beamline'] = beamlinedefaultvalue
        conf_dict['converter_ver'] = get_version()
    # Look to see if aliens is set and if it is a directory or a block of coordinates
    if 'config_data' in conf_dicts.keys():
        conf_dict = conf_dicts['config_data']
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
    if 'config_rec' in conf_dicts.keys():
        import ast
        conf_dict = conf_dicts['config_rec']
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

    return conf_dicts


# def get_conf_dict(cfile_path, cfile):
#     """
#     This function takes a config file name and creates a dictionary of all the parameters defined in the config
#     file using the = to split key,value pairs.
#     There is a special case where values can be enclosed in () with a line for each value.
#     Parameters
#     ----------
#     cfile : str
#         configuration file name
#     startdir : str
#         a directory with configuration files to be converted
#     Returns
#     -------
#     currentdic : dict
#         a dictionary with the configuration parameters
#     """
#     cfile_path = cfile_path.replace(os.sep, '/')
#     cdict = get_config_dict(cfile_path, cfile)
#     cdict = replace_keys(cdict, cfile)
#     cdict = convert_dict(cdict, cfile)
#
#     return cdict
#

def convert(conf_dir, save=True):
    """
    This script will convert old config files to the newer format using the following critera

    config_instr

    create if does not exist
    move specfile from config to config_instr
    move energy, delta, gamma, detdist, th, chi, phi, scanmot, scanmot_del, detector, diffractometer
        from config_disp to config_instr

    Config file

    Add beamline
    Move separate_scans, separate_scan_ranges from config_prep to config
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

    change arm to detdist
    change dth to theta

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
    picked a default value of aps_34idc. If the specfile parameter was found in config_prep it was moved over to the
    config file.
    All non-relevant data in the files are stripped out including spaces and comments.
    If an aliens parameter is found id the config_data
    file it is tested to see if it is a file or a set of coordinates and the alien_alg is set to the correct type.

    Parameters
    ----------
    conf_dir : str
        a directory with configuration files to be converted
    Returns
    -------
    nothing
    """
    conf_dir = conf_dir.replace(os.sep, '/')
    # First check to see if directory exists, if not then exit
    if not os.path.exists(conf_dir):
        # there is nothing to convert
        print('configuration directory', conf_dir, 'does not exist')
        return None

    # read main config and check the converter version
    main_conf = ut.read_config(conf_dir + '/config')
    if main_conf is None:
        return None
    if 'converter_ver' in main_conf:
        conf_version = main_conf['converter_ver']
    else:
        conf_version = None
    if conf_version == get_version():
        return None

    config_dicts = {}
    if not os.path.isfile(conf_dir + '/config_instr'):
        config_dicts['config_instr'] = {}
    for cfile in config_maps.keys():
        conf_file = conf_dir + '/' + cfile
        # check if file exist
        if not os.path.isfile(conf_file):
            continue
        if os.access(os.path.dirname(conf_dir), os.W_OK):
            shutil.copy(conf_file, conf_file + '_backup')

        config_dicts[cfile] = ut.read_config(conf_file)

        # Use map file to see what items need to change
        # Use the map file to determine what parameters need to be changed.
        # if the key is found then do the remap, if not then skip.
        config_dicts[cfile] = replace_keys(config_dicts[cfile], cfile)

    # move parameters between files
    for k,v in move_dict.items():
        for nk, sv in v.items():
            for p in sv:
                if p in config_dicts[k]:
                    config_dicts[nk][p] = config_dicts[k][p]

    # Some special cases:
    # Now only applies if the configuration version is None
    # config: if beamline has no value then it was never defined so set it to the default
    #         set the converter version to current version
    # config_data: if it has the older aliens format of coordinates/file then update to new layout
    # config_rec: algorithm_sequence and pc_interval changed
    if conf_version is None:
        config_dicts = convert_dict(config_dicts)

    # Write the data out to the same-named file
    if save:
        for k, v in config_dicts.items():
            file_name = conf_dir + '/' + k
            ut.write_config(v, file_name)

    return config_dicts


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("conv_dir", help="path to directory with configuration files that will be converted")
    args = parser.parse_args()
    convert(args.conv_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
