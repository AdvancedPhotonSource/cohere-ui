# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import sys
import os
import cohere_core.utilities as ut
import cohere_ui.api.convertconfig as conv


def get_config_maps(experiment_dir, configs, **kwargs):
    """
    Reads the configuration files included in configs list and returns dictionaries.
    It will check for missing main config, for converter version. If needed it will convert
    to the latest version.

    :param experiment_dir: str
        directory where the experiment files are loacted
    :param configs: list str
        list of configuaration files key names requested by calling function
        The main config is always processed.
    :param kwargs: ver parameters
        may contain:
        - rec_id : reconstruction id, pointing to alternate config
        - no_verify : boolean switch to determine if the verification error is returned
    :return:
        error message
        configuration dictionaries
        boolean value telling if conversion happened
    """
    no_verify = kwargs.pop('no_verify', False)
    maps = {}
    # always get main config
    conf_dir = ut.join(experiment_dir, 'conf')
    main_conf = ut.join(conf_dir, 'config')
    if not os.path.isfile(main_conf):
        # return 'no main config, exiting.', maps, None
        raise ValueError('no main config, exiting.')
    main_config_map = ut.read_config(main_conf)

    msg = ut.verify('config', main_config_map)
    if len(msg) > 0:
        if not no_verify:
            raise ValueError(msg)
           # return msg, maps, None

    converted = False

    # convert configuration files if different converter version
    if 'converter_ver' not in main_config_map or conv.get_version() is None or conv.get_version() > main_config_map['converter_ver']:
        conv.convert(conf_dir)
        main_config_map = ut.read_config(main_conf)
        converted = True

    maps['config'] = main_config_map

    if 'config_instr' in configs or 'config_mp' in configs:
        # the configuration file applies to specific beamline and needs to be imported
        beamline = main_config_map.get('beamline', None)
        if beamline is None:
            raise ValueError(f'cannot import cohere_ui.beamlines.{beamline} module, exiting.')
            # return f'cannot import cohere_ui.beamlines.{beamline} module, exiting.', maps, None
        import importlib
        beam_ver = importlib.import_module(f'cohere_ui.beamlines.{beamline}.beam_verifier')
    else:
        beam_ver = None

    verifier_map = {'config_data' : ut, 'config_rec' : ut, 'config_instr' : beam_ver,
                    'config_prep' : beam_ver, 'config_disp' : ut, 'config_mp' : beam_ver}

    rec_id = kwargs.get('rec_id')
    for conf in configs:
        # special case for rec_id
        if rec_id is not None and conf == 'config_rec':
            conf_file = ut.join(experiment_dir, 'conf', f'{conf}_{rec_id}')
        else:
            conf_file = ut.join(experiment_dir, 'conf', conf)
        if not os.path.isfile(conf_file):
            continue
        config_map = ut.read_config(conf_file)
        # verify the config map, for beamline specific conf file the verifier has to be imported
        msg = verifier_map[conf].verify(conf, config_map)
        if len(msg) > 0:
            if not no_verify:
                raise ValueError(msg)

        maps[conf] = config_map

    return maps, converted


def get_pkg(proc, dev):
    pkg = 'np'

    if proc == 'auto':
        try:
            import cupy
            pkg = 'cp'
            if dev == [-1]:
                raise ValueError('cupy processing is available, define device')
        except:
            try:
                import torch
                pkg = 'torch'
            except:
                pass  # lib set to 'np'
    elif proc == 'cp':
        if sys.platform == 'darwin':
            raise ValueError('cupy is not supported by Mac, select different processing')
        try:
            import cupy
            if dev == [-1]:
                raise ValueError('when using cupy processing, define device')
            pkg = 'cp'
        except:
            raise ValueError('cupy is not installed, select different processing')
    elif proc == 'torch':
        try:
            import torch
            pkg = 'torch'
        except:
            raise ValueError('torch is not installed, select different processing')
    elif proc == 'np':
        pass  # lib set to 'np'
    else:
        err_msg = f'invalid "processing" value, {proc} is not supported'
        raise ValueError(err_msg)

    return pkg
