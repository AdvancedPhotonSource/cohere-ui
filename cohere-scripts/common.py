import sys
import os
import convertconfig as conv
import cohere_core.utilities as ut


def get_config_maps(experiment_dir, configs, config_id=None):
    """
    Reads the configuration files included in configs list and returns dictionaries.
    It will check for missing main config, for converter version. If needed it will convert
    to the latest version.

    :param experiment_dir: str
        directory where the experiment files are loacted
    :param configs: list str
        list of configuaration files key names requested by calling function
        The main config is always processed.
    :param config_id: str
        the string identifying alternate configuration
    :return:
        error message
        configuration dictionaries
        boolean value telling if conversion happened
    """
    maps = {}
    # always get main config
    conf_dir = ut.join(experiment_dir, 'conf')
    main_conf = ut.join(conf_dir, 'config')
    if not os.path.isfile(main_conf):
        return maps, None

    converted = False
    main_config_map = ut.read_config(main_conf)
    # convert configuration files if different converter version
    if 'converter_ver' not in main_config_map or conv.get_version() is None or conv.get_version() > main_config_map['converter_ver']:
        conv.convert(conf_dir)
        main_config_map = ut.read_config(main_conf)
        converted = True

    maps['config'] = main_config_map

    for conf in configs:
        # special case for rec_id
        if config_id is not None and (conf == 'config_rec' or conf == 'config_disp'):
            conf_file = ut.join(experiment_dir, 'conf', f'{conf}_{config_id}')
        else:
            conf_file = ut.join(experiment_dir, 'conf', conf)

        if not os.path.isfile(conf_file):
            continue

        config_map = ut.read_config(conf_file)

        maps[conf] = config_map

    return maps, converted


def get_pkg(proc, dev):
    pkg = 'np'
    err_msg = ''

    if proc == 'auto':
        if sys.platform == 'darwin':
            return err_msg, pkg
        try:
            import cupy
            pkg = 'cp'
        except:
            try:
                import torch
                pkg = 'torch'
            except:
                pass
    elif proc == 'cp':
        if sys.platform == 'darwin':
            return 'cupy is not supported by Mac, running with numpy', pkg
        if dev == [-1]:
            return 'when using cupy processing, define device', pkg
        try:
            import cupy
            pkg = 'cp'
        except:
            err_msg = 'cupy is not installed, select different processing'
    elif proc == 'torch':
        try:
            import torch
            pkg = 'torch'
        except:
            err_msg = 'pytorch is not installed, select different processing'
    elif proc == 'np':
        pass  # lib set to 'np'
    else:
        err_msg = f'invalid "processing" value, {proc} is not supported'

    return err_msg, pkg
