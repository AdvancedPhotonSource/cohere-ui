#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################
"""
This script is used to create a cohere experiment directory structure with predefined configuration files.
The main configuration file is created and the other files are copied into the newly created experiment space.

This script is typically used to create an initial cohere experiment after data collection.

To run this script from command line::

    copy_setup <id> <scan_no> <conf_dir> --specfile <specfile> --copy_prep

One can use --help to get explanation of command line parameters.
"""

__author__ = "Ross Harder"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['setup_example',
           'main']

import argparse
import os
import shutil
import glob
import cohere_core.utilities as ut


######################################################################
def setup_example(prefix, scan, conf_dir, **kwargs):
    """
    Creates the new cohere experiment directory, creates main configuration files, and copies other configuration files.

    :param id: literal part ofnew experiment id
    :param scan: scan(s) number od the new experiment
    :param conf_dir: directory from where the configuration files will be copied
    :param specfile: optional, specfile configuration to write to config_instr file
    :param copy_prep: optional, if True, the preprocessed file is also copied
        
    """
    conf_dir = conf_dir.replace(os.sep, '/')
    if not os.path.isdir(conf_dir):
        print(f'configured directory {conf_dir} does not exist')
        return

    main_conf = ut.join(conf_dir, 'config')
    config_map = ut.read_config(main_conf)
    if config_map is None:
        return None

    id = prefix + '_' + scan

    if 'working_dir' in config_map:
        working_dir = config_map['working_dir'].replace(os.sep, '/')
    else:
        working_dir = os.getcwd().replace(os.sep, '/')

    new_experiment_conf_dir = ut.join(working_dir, id, 'conf')
    # copy configuration directory, it will fail if directory already exists
    shutil.copytree(conf_dir, new_experiment_conf_dir)

    # fix main config file
    config_map = ut.read_config(ut.join(new_experiment_conf_dir, 'config'))
    config_map['experiment_id'] = prefix
    config_map['scan'] = scan
    config_map['working_dir'] = working_dir
    ut.write_config(config_map, ut.join(new_experiment_conf_dir, 'config'))

    # delete results_dir as it points to the original results directory
    disp_config_map = ut.read_config(ut.join(new_experiment_conf_dir, 'config_disp'))
    if 'results_dir' in disp_config_map:
        disp_config_map.pop('results_dir')
        ut.write_config(disp_config_map, ut.join(new_experiment_conf_dir, 'config_disp'))

    # include specfile in config_instr if given
    specfile = kwargs.get('specfile')
    if specfile is not None:
        instr_config_map = ut.read_config(ut.join(new_experiment_conf_dir, 'config_instr'))
        instr_config_map['specfile'] = specfile
        ut.write_config(instr_config_map, ut.join(new_experiment_conf_dir, 'config_instr'))

    if 'copy_prep' in kwargs:
        copy_prep = kwargs['copy_prep']
    if copy_prep:
        # use abspath to get rid of trailing dir sep if it is there
        other_exp_dir = os.path.split(os.path.abspath(conf_dir))[0]
        new_exp_dir = os.path.split(os.path.abspath(new_experiment_conf_dir))[0]

        # get case of single scan or summed
        prep_dir_list = glob.glob(ut.join(other_exp_dir, 'preprocessed_data'), recursive=True)
        for dir in prep_dir_list:
            shutil.copytree(dir.replace(os.sep, '/'), ut.join(new_exp_dir, 'preprocessed_data'))

            # get case of split scans
        prep_dir_list = glob.glob(other_exp_dir  = '/scan*/preprocessed_data', recursive=True)
        for dir in prep_dir_list:
            scandir = os.path.basename(os.path.split(dir.replace(os.sep, '/'))[0]).replace(os.sep, '/')
            shutil.copytree(dir.replace(os.sep, '/'), ut.join(new_exp_dir, scandir, 'preprocessed_data'))

    return ut.join(working_dir, id)
        #################################################################################


def main():
    """
    An entry function that takes command line parameters. It invokes the processing function setup_example with
    the parameters. The command line parameters: id, scan, conf_dir, --specfile, --copy_prep.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("id", help="prefix to name of the experiment/data reconstruction")
    parser.add_argument("scan", help="a range of scans to prepare data from")
    parser.add_argument("conf_dir", help="directory where the configuration files are located")
    parser.add_argument('--specfile', action='store')
    parser.add_argument('--copy_prep', action='store_true')

    # would be nice to have specfile as optional arg?
    args = parser.parse_args()
    scan = args.scan
    id = args.id
    conf_dir = args.conf_dir

    return setup_example(id, scan, conf_dir, copy_prep=args.copy_prep, specfile=args.specfile)


if __name__ == "__main__":
    exit(main())
