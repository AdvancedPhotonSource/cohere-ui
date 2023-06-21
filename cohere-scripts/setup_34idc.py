# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script create experiment directory space and copies configuration files.

After the script is executed the experiment directory will contain "conf" subdirectory with configuration files, copies of files from given directory.
"""

__author__ = "Ross Harder"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['copy_conf',
           'setup_rundirs',
           'main']

import argparse
import sys
import os
import shutil
import glob
import util.util as ut


######################################################################
def copy_conf(src, dest, specfile):
    """
    Copies configuration files from src directory to dest directory.

    Parameters
    ----------
    src : str
        source directory containing configuration files
    dest : str
        directory where the files will be saved
    
    Returns
    -------
    nothing
    """
    try:
        conf_prep = src + '/config_prep'
        shutil.copy(conf_prep, dest)
    except:
        pass
    try:
        conf_data = src + '/config_data'
        shutil.copy(conf_data, dest)
    except:
        pass
    try:
        conf_rec = src + '/config_rec'
        shutil.copy(conf_rec, dest)
    except:
        pass
    try:
        conf_disp = src + '/config_disp'
        shutil.copy(conf_disp, dest)
    except:
        pass
    try:
        conf_instr = src + '/config_instr'
        if specfile is None:
            shutil.copy(conf_instr, dest)
        else:
            instr_config_map = ut.read_config(conf_instr)
            instr_config_map['specfile'] = specfile
            ut.write_config(instr_config_map, conf_instr)
    except:
        pass


######################################################################
def setup_rundirs(prefix, scan, conf_dir, **kwargs):
    """
    Concludes the experiment directory, creates main configuration files, and calls function to copy other configuration files.

    Parameters
    ----------
    prefix : str
        prefix to name of the experiment/data reconstruction
    scan : str
        a range of scans to prepare data from
    conf_dir : str
        directory from where the configuration files will be copied
    specfile : str
        optional, from kwargs, specfile configuration to write to config file
    copy_prep : bool
        optional, from kwargs, if sets to True, the prepared file is also copied
        
    """
    conf_dir = conf_dir.replace(os.sep, '/')
    if not os.path.isdir(conf_dir):
        print('configured directory ' + conf_dir + ' does not exist')
        return

    main_conf = conf_dir + '/config'
    config_map = ut.read_config(main_conf)
    if config_map is None:
        return None

    id = prefix + '_' + scan

    if 'working_dir' in config_map:
        working_dir = config_map['working_dir'].replace(os.sep, '/')
    else:
        working_dir = os.getcwd().replace(os.sep, '/')

    experiment_dir = working_dir + '/' + id
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    experiment_conf_dir = experiment_dir + '/conf'
    if not os.path.exists(experiment_conf_dir):
        os.makedirs(experiment_conf_dir)

    # override the config_map with values for new experiment
    config_map['working_dir'] = working_dir
    config_map['experiment_id'] = prefix
    config_map['scan'] = scan
    ut.write_config(config_map, experiment_conf_dir + '/config')

    # here we want the command line to be used if present
    if 'specfile' in kwargs and kwargs['specfile'] is not None:
        specfile = kwargs['specfile']
    else:
        specfile = None

    copy_conf(conf_dir, experiment_conf_dir, specfile)

    if 'copy_prep' in kwargs:
        copy_prep = kwargs['copy_prep']
    if copy_prep:
        # use abspath to get rid of trailing dir sep if it is there
        other_exp_dir = os.path.split(os.path.abspath(conf_dir))[0]
        new_exp_dir = os.path.split(os.path.abspath(experiment_conf_dir))[0]

        # get case of single scan or summed
        prep_dir_list = glob.glob(other_exp_dir + '/preprocessed_data', recursive=True)
        for dir in prep_dir_list:
            shutil.copytree(dir.replace(os.sep, '/'), new_exp_dir + '/preprocessed_data')

            # get case of split scans
        prep_dir_list = glob.glob(other_exp_dir  = '/scan*/preprocessed_data', recursive=True)
        for dir in prep_dir_list:
            scandir = os.path.basename(os.path.split(dir.replace(os.sep, '/'))[0]).replace(os.sep, '/')
            shutil.copytree(dir.replace(os.sep, '/'), new_exp_dir + '/' + scandir + '/preprocessed_data')
    return experiment_dir
        #################################################################################


def main(arg):
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

    return setup_rundirs(id, scan, conf_dir, copy_prep=args.copy_prep, specfile=args.specfile)


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
