import sys
import os
#import cohere_core.utilities as ut
import cohere_core as cohere
import convertconfig as conv


def get_lib(proc):
    lib = 'np'
    err_msg = ''

    # find which library to run it on, default is numpy ('np')
    if sys.platform == 'darwin':
        return err_msg, lib

    if proc == 'auto':
        try:
            import cupy
            lib = 'cp'
        except:
            try:
                import torch
                lib = 'torch'
            except:
               pass
    elif proc == 'cp':
        try:
            import cupy
            lib = 'cp'
        except:
            masg = 'cupy is not installed, select different library (proc)'
    elif proc == 'torch':
        try:
            import torch
            lib = 'torch'
        except:
            masg = 'pytorch is not installed, select different library (proc)'
    elif proc == 'np':
        pass  # lib set to 'np'
    else:
        masg = 'invalid "proc" value ' + proc + ' is not supported'

    return err_msg, lib


def get_config_maps(experiment_dir, configs, rec_id, debug):
    err_msg = ''
    maps = []

    # always get main config
    # convert configuration files if needed
    main_conf = os.path.join(experiment_dir, 'conf', 'config').replace(os.sep, '/')
    if not os.path.isfile(main_conf):
        err_msg = "info: missing " + main_conf + " configuration file"
        return err_msg, maps

    main_config_map = cohere.read_config(main_conf)
    if 'converter_ver' not in main_config_map or conv.get_version() is None or conv.get_version() < main_config_map['converter_ver']:
        main_config_map = conv.convert(conf_dir, 'config')
    # verify main config file
    err_msg = cohere.verify('config', main_config_map)
    if len(err_msg) > 0:
        # the error message is printed in verifier
        if not debug:
            return err_msg, maps

    maps.append(main_config_map)

    for conf in configs:
        if rec_id is not None and (conf == 'config_rec' or conf == 'config_disp'):
            conf_file = os.path.join(experiment_dir, 'conf', conf + '_' + rec_id).replace(os.sep, '/')
        else:
            conf_file = os.path.join(experiment_dir, 'conf', conf).replace(os.sep, '/')

        if not os.path.isfile(conf_file):
            err_msg = "info: missing " + conf_file + " configuration file"
            return err_msg, maps

        config_map = cohere.read_config(conf_file)

        # verify configuration
        err_msg = cohere.verify(conf, config_map)
        if len(err_msg) > 0:
            # the error message is printed in verifier
            if not debug:
                return err_msg, maps
        
        maps.append(config_map)
 
    return err_msg, maps

