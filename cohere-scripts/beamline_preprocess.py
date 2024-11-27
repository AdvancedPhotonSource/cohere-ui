#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script reads raw data, applies correction related to instrument, and saves prepared data.
This script is written for a specific APS beamline. It reads multiple raw data files in each scan directory, applies
darkfield and whitefield correction if applicable, creates 3D stack for each scan, then alignes and combines with
other scans.
"""

__author__ = "Barbara Frosik"
__docformat__ = 'restructuredtext en'
__all__ = ['handle_prep',
           'main']

import argparse
import importlib
import cohere_core.utilities as ut
import auto_data as ad
from multiprocessing import Process
import common as com
import multipeak as mp


def handle_prep(experiment_dir, **kwargs):
    """
    Reads the configuration files and accrdingly creates prep_data.tif file in <experiment_dir>/prep directory or multiple
    prep_data.tif in <experiment_dir>/<scan_<scan_no>>/prep directories.
    Parameters
    ----------
    experimnent_dir : str
        directory with experiment files
    kwargs: ver parameters
        may contain:
        - rec_id : reconstruction id, pointing to alternate config
        - no_verify : boolean switch to determine if the verification error is returned
        - debug : boolean switch not used in this code
    Returns
    -------
    experimnent_dir : str
        directory with experiment files
    """
    print('pre-processing data')

    # requesting the configuration files that provide parameters for preprocessing
    conf_list = ['config_prep', 'config_instr', 'config_mp']
    err_msg, conf_maps, converted = com.get_config_maps(experiment_dir, conf_list, **kwargs)
    if len(err_msg) > 0:
        return err_msg

    # check the maps
    if 'config_instr' not in conf_maps.keys():
        return 'missing config_instr file, exiting'
    if 'config_prep' not in conf_maps.keys():
        print('info: no config_prep file, continue with no parameters')

    main_conf_map = conf_maps['config']

    # checked already if beamline is configured
    beamline = main_conf_map['beamline']
    try:
        instr_module = importlib.import_module(f'beamlines.{beamline}.instrument')
        preprocessor = importlib.import_module(f'beamlines.{beamline}.preprocessor')
    except Exception as e:
        print(e)
        print(f'cannot import beamlines.{beamline} module.')
        return f'cannot import beamlines.{beamline} module.'

    prep_conf_map = conf_maps.get('config_prep', {})

    # config_instr contain all parameters needed to create instrument object
    # add main config and mutipeak config to parameters, as the beamline might need them
    # example: beamline 34idc needs scan number to parse spec file
    all_params = {k:v for d in conf_maps.values() for k,v in d.items()}
    instr_obj = instr_module.create_instr(all_params)
    if instr_obj is None:
        return 'cannot create instrument, check configuration'

    scan = all_params.get('scan', None)
    if scan is None:
        print('scan not defined in configuration')
        return ('scan not defined in configuration')

    # get the settings from main config
    auto_data = main_conf_map.get('auto_data', False)
    separate_scans = main_conf_map.get('separate_scans', False)
    separate_scan_ranges = main_conf_map.get('separate_scan_ranges', False)
    multipeak = main_conf_map.get('multipeak', False)

    # 'scan' is configured as string. It can be a single scan, range, or combination separated by comma.
    # Parse the scan into list of scan ranges, defined by starting scan, and ending scan, inclusive.
    # The single scan has range defined as the same starting and ending scan.
    scan_ranges = []
    scan_units = [u for u in scan.replace(' ','').split(',')]
    for u in scan_units:
        if '-' in u:
            r = u.split('-')
            scan_ranges.append([int(r[0]), int(r[1])])
        else:
            scan_ranges.append([int(u), int(u)])

    # get tuples of (scan, data info) for for each scan in the scan ranges.
    # The scan_datainfo is a list of sub-lists, each sub-list reflects scan range.
    # If there is a single scan, the output will be: [[(scan, data info)]].
    #
    # Note: For aps_34idc the data info is a directory path to the data.
    # For esrf_id01 the data info is a node in hdf5 file.
    scans_datainfo = instr_obj.datainfo4scans(scan_ranges)

    # remove exclude_scans from the scans_dirs
    remove_scans = prep_conf_map.get('remove_scans', None)
    if remove_scans is not None:
        scans_datainfo = [[s_d for s_d in batch if s_d[0] not in remove_scans] for batch in scans_datainfo]

    # auto_data should not be configured for separate scans
    if auto_data and not separate_scans:
        outliers_scans = ad.find_outlier_scans(experiment_dir, instr_obj.get_scan_array, scans_datainfo, separate_scan_ranges or multipeak)
        if len(outliers_scans) > 0:
            # remove outliers_scans from the scans_dirs
            scans_datainfo = [[s_d for s_d in batch if s_d[0] not in outliers_scans] for batch in scans_datainfo]
        # save configuration with the auto found outliers. Save even if no outliers found to show it.
        prep_conf_map['outliers_scans'] = outliers_scans
        ut.write_config(prep_conf_map, ut.join(experiment_dir, 'conf', 'config_prep'))

    if separate_scans:
        # get all (scan, data info) tuples, process each scan and save the data in scans directories.
        single_scans_datainfo = [s_d for batch in scans_datainfo for s_d in batch]
        # passing in lambda: instr_obj.get_scan_array as the function is instrument dependent
        ad.process_separate_scans(instr_obj.get_scan_array, single_scans_datainfo, experiment_dir)
    elif separate_scan_ranges:
        # combine scans within ranges, save the data in scan ranges directories.
        processes = []
        for batch in scans_datainfo:
            indx = str(batch[0][0])
            indx = f'{indx}-{str(batch[-1][0])}'
            save_file = ut.join(experiment_dir, f'scan_{indx}', 'preprocessed_data', 'prep_data.tif')
            p = Process(target=preprocessor.process_batch,
                        args=(instr_obj.get_scan_array, batch, save_file, experiment_dir))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    elif multipeak:
        mp.preprocess(preprocessor, instr_obj, scans_datainfo, experiment_dir, conf_maps['config_mp'])
    else:
        # combine all scans
        scans_datainfo = [e for batch in scans_datainfo for e in batch]
        save_file = ut.join(experiment_dir, 'preprocessed_data', 'prep_data.tif')
        preprocessor.process_batch(instr_obj.get_scan_array, scans_datainfo, save_file, experiment_dir)

    print('finished beamline preprocessing')
    return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir",
                        help="directory where the configuration files are located")
    parser.add_argument("--no_verify", action="store_true",
                        help="if True the verifier has no effect on processing, error is always printed when incorrect configuration")
    parser.add_argument("--debug", action="store_true",
                        help="not used currently, available to developer for debugging")
    args = parser.parse_args()
    handle_prep(args.experiment_dir, no_verify=args.no_verify, debug=args.debug)


if __name__ == "__main__":
    exit(main())
