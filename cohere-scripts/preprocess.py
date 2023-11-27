# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script invokes the beamline_preprocess and standard_preprocess scripts.
This script uses configuration parameters from the experiment configuration files.
"""

author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['main']

import beamline_preprocess as prep
import standard_preprocess as dt
import sys
import os
import argparse


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory")

    args = parser.parse_args()
    experiment_dir = args.experiment_dir.replace(os.sep, '/')
   
    prep.handle_prep(experiment_dir)
    dt.format_data(experiment_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
