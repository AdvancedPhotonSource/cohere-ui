import sys
import os
import argparse
import cohere_core as cohere
import util.util as ut

def manage_reconstruction(exp_dir, rec_id=None):
    """
    This function starts the interruption discovery process and continues the recontruction processing.
    It reads configuration file defined as <experiment_dir>/conf/config_rec.
    If multiple generations are configured, or separate scans are discovered, it will start concurrent reconstructions.
    It creates image.npy file for each successful reconstruction.
    Parameters
    ----------
    experiment_dir : str
        directory where the experiment files are loacted
    rec_id : str
        optional, if given, alternate configuration file will be used for reconstruction, (i.e. <rec_id>_config_rec)
    Returns
    -------
    nothing
    """
    print('starting reconstruction')
    exp_dir = exp_dir.replace(os.sep, '/')
    config_map = ut.read_config(exp_dir + "/conf/config")
    config_map.update(ut.read_config(exp_dir + "/conf/config_rec"))

    if 'device' in config_map:
        dev = config_map['device']
    else:
        dev = [-1]

    # find which library to run it on, default is numpy ('np')
    if 'processing' in config_map:
        proc = config_map['processing']
    else:
        proc = 'auto'

    lib = 'np'
    if proc == 'auto':
        try:
            import cupy
            lib = 'cp'
        except:
            try:
                import torch
                lib = 'torch'
            except:
                pass
    elif proc == 'cp':
        try:
            import cupy
            lib = 'cp'
        except:
            print('cupy is not installed, select different library (proc)')
            return
    elif proc == 'torch':
        try:
            import torch
            lib = 'torch'
        except:
            print('pytorch is not installed, select different library (proc)')
            return
    elif proc == 'np':
        pass  # lib set to 'np'
    else:
        print('invalid "proc" value', proc, 'is not supported')
        return
    peak_dirs = []
    for dir in os.listdir(exp_dir):
        if dir.startswith('mp'):
            peak_dirs.append(exp_dir + '/' + dir)
    cohere.reconstruction_coupled.reconstruction(lib, config_map, peak_dirs, dev)

    print('finished reconstruction')


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory.")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir

    manage_reconstruction(experiment_dir)


if __name__ == "__main__":
    main(sys.argv[1:])

# python run_reconstruction.py opencl experiment_dir
