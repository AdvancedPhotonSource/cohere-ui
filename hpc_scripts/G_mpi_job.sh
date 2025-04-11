#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This script is a wrapper calling cohere_scripts module controlling reconstructions using genetic algorithm (GA).
"""
import argparse
from cohere_scripts.inner_scripts.reconstruction_GA import reconstruction
import cohere_core.utilities.utils as ut


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['main']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", help="experiment directory")
    args = parser.parse_args()

    exp_dir = args.exp_dir
    conf_file = ut.join(exp_dir, 'conf', 'config_rec')
    datafile = ut.join(exp_dir, 'phasing_data', 'data.tif')

    reconstruction('cp', conf_file, datafile, exp_dir, None, None, True)


if __name__ == "__main__":
    exit(main())