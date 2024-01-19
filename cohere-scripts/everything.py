# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script invokes all cohere-scripts needed to present reconstructed data from raw data: run_prep_34idc, format_data, run_rec, run_disp.
This script is written for specific beamline, as it invokes targetted cohere-scripts.
This script uses configuration parameters from the experiment configuration files.
"""

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['run_all',
           'main']

import beamline_preprocess as prep
import standard_preprocess as dt
import run_reconstruction as rec
import beamline_visualization as dsp
import os
import argparse

def run_all(experiment_dir, **kwargs):
    """
    Creates a "config_prep" file with some parameters commented out.

    Parameters
    ----------
    dev : str
        processing library, choices are: cpu, cuda, opencl
    experiment_dir : str
        directory where the experiment files are loacted
    config_id : str
        optional, if given, alternate configuration file will be used for reconstruction, (i.e. <config_id>_config_rec)

    Returns
    -------
    nothing
    """
    experiment_dir = experiment_dir.replace(os.sep, '/')
    prep.handle_prep(experiment_dir, **kwargs)
    dt.format_data(experiment_dir, **kwargs)
    rec.manage_reconstruction(experiment_dir, **kwargs)
    dsp.handle_visualization(experiment_dir, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory")
    parser.add_argument("--config_id", help="reconstruction id, a prefix to '_results' directory")
    parser.add_argument("--debug", action="store_true",
                        help="if True the vrifier has no effect on processing")

    args = parser.parse_args()
    run_all(args.experiment_dir, config_id=args.config_id, debug=args.debug)


if __name__ == "__main__":
    main()

