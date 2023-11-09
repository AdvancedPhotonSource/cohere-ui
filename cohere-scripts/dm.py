#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This script is used in the automated data flows.
The cohere-ui experiment directory structure (i.e. <working_directory>/<id>_<scan> must exist in the
beamline experiment directory. The cohere-ui experiment must have config file defined.
The config_instr, config_prep, and config_data are generated in this script.
The result of this script is preprocessed data, including beamline and standard preprocessing.
The resulting file is placed in the beamline experiment directory structure under directory monitored
by the DM process. (for 34-idc the monitored directory root is ~/34idc-data/hpc_data).
The process will discover data ready for phasing and will trigger data transfer to HPC.
"""

__author__ = "Barbara Frosik"
__docformat__ = 'restructuredtext en'
__all__ = ['handle_prep',
           'main']

import argparse
import sys
import importlib
import convertconfig as conv
import cohere_core as cohere
import cohere_core.utilities as ut
import beamline_preprocess as bp


def handle_prep(experiment_dir, **kwargs):
    """
    Reads the configuration files and accrdingly creates prep_data.tif file in <experiment_dir>/prep directory or multiple
    prep_data.tif in <experiment_dir>/<scan_<scan_no>>/prep directories.
    Parameters
    ----------
    experimnent_dir : str
        directory with experiment files
    Returns
    -------
    experimnent_dir : str
        directory with experiment files
    """
    print('preparing data')
    # check configuration
    main_conf_file = experiment_dir + '/conf/config'
    main_conf_map = ut.read_config(main_conf_file)
    if main_conf_map is None:
        print('cannot read configuration file ' + main_conf_file)
        return 'cannot read configuration file ' + main_conf_file
    # convert configuration files if needed
    if 'converter_ver' not in main_conf_map or conv.get_version() is None or conv.get_version() > main_conf_map[
        'converter_ver']:
        conv.convert(experiment_dir + '/conf')
        # re-parse config
        main_conf_map = ut.read_config(main_conf_file)

    er_msg = cohere.verify('config', main_conf_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        debug = 'debug' in kwargs and kwargs['debug']
        if not debug:
            return er_msg

    main_conf_map = ut.read_config(main_conf_file)
    if 'beamline' in main_conf_map:
        beamline = main_conf_map['beamline']
        try:
            beam_prep = importlib.import_module('beamlines.' + beamline + '.prep')
            beam_dm = importlib.import_module('beamlines.' + beamline + '.dm_params')
        except Exception as e:
            print(e)
            print('cannot import beamlines.' + beamline + '.prep module.')
            return 'cannot import beamlines.' + beamline + '.prep module.'
    else:
        print('Beamline must be configured in configuration file ' + main_conf_file)
        return 'Beamline must be configured in configuration file ' + main_conf_file

    # generate prep_conf_map and instr_conf_map
    instr_conf_map = {}
    prep_conf_map = {}

    instr_conf_map['diffractometer'] = '34idc'
    (prep_conf_map['data_dir'], instr_conf_map['specfile']) = \
        beam_dm.DM_params.get_data_dir_spec(main_conf_map['working_dir'])
    (prep_conf_map['darkfield_filename'], prep_conf_map['whitefield_filename']) = \
        beam_dm.DM_params.get_corrections(main_conf_map['working_dir'])

    # create BeamPrepData object defined for the configured beamline
    conf_map = main_conf_map
    conf_map.update(prep_conf_map)
    conf_map.update(instr_conf_map)
    if 'multipeak' in main_conf_map and main_conf_map['multipeak']:
        conf_map.update(ut.read_config(experiment_dir + '/conf/config_mp'))
    prep_obj = beam_prep.BeamPrepData()
    msg = prep_obj.initialize(experiment_dir, conf_map)
    if len(msg) > 0:
        print(msg)
        return msg

    # auto_data
    outliers_scans = bp.find_outlier_scans(experiment_dir, prep_obj)
    if len(outliers_scans) > 0:
        prep_obj.outliers_scans = outliers_scans
        # save configuration with the auto found outliers
        prep_conf_map['outliers_scans'] = outliers_scans
    # write configuration files
    prep_file_name = experiment_dir + '/conf/config_prep'
    ut.write_config(prep_conf_map, prep_file_name)
    ut.write_config(instr_conf_map, experiment_dir + '/conf/config_instr')

    msg = bp.prep_data(prep_obj, experiment_dir=experiment_dir)
    if len(msg) > 0:
        print(msg)
        return msg
    print('finished beamline preprocessing, starting standard preprocessing')

    data_conf_map = {}
    data_conf_map['data_dir'] = beam_dm.DM_params.get_dm_data_dir(experiment_dir)
    cohere.prep(prep_file_name, True, data_conf_map)
    ut.write_config(data_conf_map, experiment_dir + '/conf/config_data')

    # define the config_disp here
    disp_conf_map = {}
    disp_conf_map['results_dir'] = beam_dm.DM_params.get_dm_results_dir(experiment_dir)
    ut.write_config(disp_conf_map, experiment_dir + '/conf/config_disp')

    return ''


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir",
                        help="directory where the configuration files are located")
    args = parser.parse_args()
    handle_prep(args.experiment_dir)


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
