# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import sys
import os
import contextlib
import importlib
import cohere_ui.api.convertconfig as conv
import cohere_core.utilities as ut
import cohere_core.utilities.schemas.beam_prep_schema as prep_schema
import cohere_core.utilities.schemas.exp_schema as exp_schema
import cohere_core.utilities.schemas.post_schema as post_schema
import cohere_core.utilities.schemas.recon_schema as recon_schema
import cohere_core.utilities.schemas.standard_prep_schema as st_prep_schema
import cohere_core.utilities.schemas.mp_schema as mp_schema


@contextlib.contextmanager
def preserve_devlib():
    """Snapshot and restore the cohere_core dvc_utils device-lib global.

    The dvc_utils helpers (cross-correlation, fast_shift, lucy_deconvolution,
    remove_ramp, ...) read a process-global ``cohere_core.utilities.dvc_utils.devlib``,
    so a stage that needs them on the CPU pins it to numpy. In a long-lived
    kernel that pin would persist and silently force any later in-kernel
    dvc_utils call (or an advanced-user Rec) onto numpy. Wrap the numpy-pinning
    region with this so the prior lib (a GPU backend, or unset on a fresh
    kernel) is put back on exit.

    Usable as a context manager (``with preserve_devlib():``) or a decorator
    (``@preserve_devlib()``).
    """
    import cohere_core.utilities.dvc_utils as dvut
    _missing = object()
    saved = getattr(dvut, 'devlib', _missing)
    try:
        yield
    finally:
        if saved is _missing:
            # Nothing was set before this region; remove what we pinned.
            if hasattr(dvut, 'devlib'):
                del dvut.devlib
        else:
            dvut.devlib = saved


def get_config_maps(experiment_dir, configs, **kwargs):
    """
    Reads the configuration files included in configs list and returns dictionaries.
    It will check for missing main config, for converter version. If needed it will convert
    to the latest version.

    :param experiment_dir: str
        directory where the experiment files are loacted
    :param configs: list str
        list of configuaration files key names requested by calling function
        The main config is always processed, thus not present in the list.
    :param kwargs: ver parameters
        may contain:
        - rec_id : reconstruction id, pointing to alternate config
        - no_verify : boolean switch to determine if the verification error is returned
    :return:
        error message
        configuration dictionaries
        boolean value telling if conversion happened
    """
    schema_imports = {'config_prep': prep_schema,
                        'config': exp_schema,
                        'config_disp': post_schema,
                        'config_rec': recon_schema,
                        'config_data': st_prep_schema,
                        'config_mp': mp_schema,
                      }
    maps = {}
    errs = {} # verification results
    # always get main config
    conf_dir = ut.join(experiment_dir, 'conf')
    main_conf = ut.join(conf_dir, 'config')
    if not os.path.isfile(main_conf):
        # return 'no main config, exiting.', maps, None
        raise ValueError('no main config, exiting.')
    main_config_map = ut.read_config(main_conf)

    # verifying types and mandatory params
    schema = exp_schema.get_config_schema()
    msg_type_err = ut.verify_types(schema, main_config_map)
    msg_param_err = ut.verify_params('config', main_config_map)
    err = ''
    if len(msg_type_err) > 0:
        err += msg_type_err + '\n'
    if len(msg_param_err) > 0:
        err += msg_param_err + '\n'
    errs['config'] = err

    converted = False

    # convert configuration files if different converter version
    if 'converter_ver' not in main_config_map or conv.get_version() is None or conv.get_version() > main_config_map['converter_ver']:
        conv.convert(conf_dir)
        main_config_map = ut.read_config(main_conf)
        converted = True
    maps['config'] = main_config_map

    if 'config_mp' in configs and not main_config_map.get('multipeak', False):
        configs.remove('config_mp')

    rec_id = kwargs.get('rec_id')
    for conf in configs:
        if conf == 'config_instr':
            # the configuration file applies to specific beamline and needs to be imported
            beamline = main_config_map.get('beamline', None)
            if beamline is None:
                raise ValueError(f'cannot import cohere_beamlines.{beamline} module, exiting.')
            instr_schema_mod = importlib.import_module(f'cohere_beamlines.{beamline}.instr_schema')
            schema =  instr_schema_mod.get_config_schema()
        else:
            schema = schema_imports[conf].get_config_schema()
        # special case for rec_id
        if rec_id is not None and conf == 'config_rec':
            conf_file = ut.join(experiment_dir, 'conf', f'{conf}_{rec_id}')
        else:
            conf_file = ut.join(experiment_dir, 'conf', conf)
        if not os.path.isfile(conf_file):
            continue
        config_map = ut.read_config(conf_file)
        # verify the config map
        msg_type_err = ut.verify_types(schema, config_map)
        msg_param_err = ut.verify_params(conf, config_map)
        err = ''
        if len(msg_type_err) > 0:
            err += msg_type_err + '\n'
        if len(msg_param_err) > 0:
            err += msg_param_err + '\n'

        errs[conf] = err
        maps[conf] = config_map

    return maps, converted, errs


def get_pkg(proc, dev, **kwargs):
    pkg = 'np'
    hpc = kwargs.get('hpc', False)
    if proc == 'auto':
        try:
            import cupy
            if dev == [-1] and not hpc:
                raise ValueError('cupy processing is available, define device')
            pkg = 'cp'
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
        except:
            raise ValueError('cupy is not installed, select different processing')
        if dev == [-1] and not hpc:
            raise ValueError('when using cupy processing, define a valid device')
        pkg = 'cp'
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
