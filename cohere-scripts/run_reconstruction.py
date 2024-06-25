# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script manages reconstruction(s).
Depending on configuration it starts either single reconstruction, GA, or multiple reconstructions. In multiple reconstruction scenario or split scans the script runs concurrent reconstructions.
"""

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['get_job_size',
           'split_resources',
           'process_scan_range',
           'manage_reconstruction',
           'main']

import os
import argparse
from multiprocessing import Process, Queue
import cohere_core as cohere
import cohere_core.utilities as ut
import common as com
from time import time


def get_job_size(size, method, pc_in_use=False):
    if method is None:
        factor = 170
        const = 100
    elif method == 'ga_fast':
        factor = 184
        const = 428
    elif method == 'populous':
        factor = 250
        const = 0

    # the memory size needed for the operation is in MB
    job_size = size * factor / 1000000. + const
    if pc_in_use:
        job_size = job_size * 2

    return job_size


def split_resources(hostfile, devs, no_scans):
    # get available hosts and number of devices for use on then
    with open(hostfile) as f:
        hosts_no_devs = [line.strip().split(':') for line in f]
        hosts_no_devs = [[s[0], int(s[1])] for s in hosts_no_devs]

    # each scan will have own hostfile
    hostfiles = [hostfile + str(i) for i in range(no_scans)]

    # distribute the hosts/devices between scans
    current_host_idx = 0
    linesep = os.linesep
    for i in range(no_scans):
        with open(hostfiles[i], mode='w+') as f:
            need_assign = devs
            while need_assign > 0:
                host, no_devs = hosts_no_devs[current_host_idx]
                if no_devs >= need_assign:
                    host_assigned_no = need_assign
                else:
                    host_assigned_no = no_devs
                f.write(f'{host}:{str(host_assigned_no)}{linesep}')
                need_assign -= host_assigned_no
                hosts_no_devs[current_host_idx][1] -= host_assigned_no
                if hosts_no_devs[current_host_idx][1] == 0:
                    current_host_idx += 1

    # delete hostfile
    os.remove(hostfile)

    return hostfiles


def process_scan_range(ga_method, lib, conf_file, datafile, dir, picked_devs, hostfile, q=None):
    """
    Calls the reconstruction function in a module identified by parameter. After the reconstruction is finished, it enqueues th eprocess id wit associated list of gpus.
    Parameters
    ----------
    ga_method : str
        defines what type of GA was requested, or None
    lib : str
        defines library to run reconstruction with
    conf_file : str
        configuration file with reconstruction parameters
    datafile : str
        name of file containing data
    dir : str
        parent directory to the <prefix>/results, or results directory
    picked_devs : list
       a list of gpus that will be used for reconstruction
    hostfile : str
       name of hostfile if cluster configuration was used
    q : Queue
       a queue that returns tuple of procees id and associated gpu list after the reconstruction process is done
    Returns
    -------
    nothing
    """
    if len(picked_devs) == 1:
        cohere.reconstruction_single.reconstruction(lib, conf_file, datafile, dir, picked_devs)
    elif ga_method is None or ga_method == 'ga_fast':
        cohere.mpi_cmd.run_with_mpi(ga_method, lib, conf_file, datafile, dir, picked_devs, hostfile)
    else:
        cohere.reconstruction_populous_GA.reconstruction(lib, conf_file, datafile, dir, picked_devs)

    if q is not None:
        q.put((os.getpid(), picked_devs, hostfile))


def manage_reconstruction(experiment_dir, config_id, debug):
    """
    This function starts the interruption discovery process and continues the recontruction processing.
    It reads configuration file defined as <experiment_dir>/conf/config_rec.
    If multiple generations are configured, or separate scans are discovered, it will start concurrent reconstructions.
    It creates image.npy file for each successful reconstruction.
    Parameters
    ----------
    experiment_dir : str
        directory where the experiment files are loacted
    config_id : str
        optional, if given, alternate configuration file will be used for reconstruction, (i.e. <rec_id>_config_rec)
    Returns
    -------
    nothing
    """
    print('started reconstruction')
    sr_time = time()

    conf_list = ['config_rec', 'config_mp']
    err_msg, conf_maps, converted = com.get_config_maps(experiment_dir, conf_list, debug, config_id)
    if len(err_msg) > 0:
        return err_msg

    main_config_map = conf_maps['config']
    rec_config_map = conf_maps['config_rec']
    mp_config_map = conf_maps['config_mp']
    # find which library to run it on, default is numpy ('np')
    if 'processing' in rec_config_map:
        proc = rec_config_map['processing']
    else:
        proc = 'auto'
    err_msg, lib = com.get_lib(proc)
    if len(err_msg) > 0:
        return err_msg

    # for multipeak reconstruction divert here
    if 'config_mp' in conf_maps:
        config_map = conf_maps['config_mp']
        config_map.update(main_config_map)
        config_map.update(rec_config_map)
        config_map.update({"save_dir": f"{experiment_dir}/results_phasing"})
        if 'device' in config_map:
            dev = config_map['device']
        else:
            dev = [-1]

        # peak_dirs = []
        # for dir in os.listdir(experiment_dir):
        #     if dir.startswith('mp'):
        #         peak_dirs.append(ut.join(experiment_dir, dir))
        peak_dirs = [ut.join(experiment_dir, d) for d in os.listdir(experiment_dir) if d.startswith("mp")]
        return cohere.reconstruction_coupled.reconstruction(lib, config_map, peak_dirs, dev)

    # exp_dirs_data list hold pairs of data and directory, where the directory is the root of phasing_data/data.tif file, and
    # data is the data.tif file in this directory.
    exp_dirs_data = []
    # experiment may be multi-scan(s) in which case reconstruction will run for each scan, or scan range
    for dir in os.listdir(experiment_dir):
        if dir.startswith('scan'):
            datafile = ut.join(experiment_dir,dir,'phasing_data', 'data.tif')
            if os.path.isfile(datafile):
                exp_dirs_data.append((datafile, ut.join(experiment_dir, dir)))
     # if there are no scan directories, assume it is combined scans experiment
    if len(exp_dirs_data) == 0:
        # in typical scenario data_dir is not configured, and it is defaulted to <experiment_dir>/data
        # the data_dir is ignored in multi-scan scenario
        if 'data_dir' in rec_config_map:
            data_dir = rec_config_map['data_dir']
        else:
            data_dir = ut.join(experiment_dir, 'phasing_data')
        datafile = ut.join(data_dir, 'data.tif')
        if os.path.isfile(datafile):
            exp_dirs_data.append((datafile, experiment_dir))

    no_scan_ranges = len(exp_dirs_data)
    if no_scan_ranges == 0:
        print('did not find data.tif file(s). ')
        return 'did not find data.tif file(s). '

    if config_id is None:
        conf_file = ut.join(experiment_dir, 'conf', 'config_rec')
    else:
        conf_file = ut.join(experiment_dir, 'conf', f'config_rec_{config_id}')

    ga_method = None
    if 'ga_generations' in rec_config_map and rec_config_map['ga_generations'] > 1:
        if 'ga_fast' in rec_config_map and rec_config_map['ga_fast']:
            ga_method = 'ga_fast'
        else:
            ga_method = 'populous'

    if 'reconstructions' in rec_config_map:
        reconstructions = rec_config_map['reconstructions']
    else:
        reconstructions = 1

    # number of wanted devices to accomodate all reconstructions is a product of no_scan_ranges and reconstructions
    want_dev_no = no_scan_ranges * reconstructions

    if lib == 'np' or 'device' not in rec_config_map:
        devices = [-1]
    else:
        devices = rec_config_map['device']

    # if device is [-1] it will be run on cpu
    if devices == [-1]:
        # for now run locally on cpu, will be enhanced to support cluster conf
        picked_devs, avail_jobs, hostfile = devices * want_dev_no, want_dev_no, None
    else:
        # based on configured devices find what is available
        # this code below assigns jobs for GPUs
        data_size = cohere.read_tif(exp_dirs_data[0][0]).size
        job_size = get_job_size(data_size, ga_method, 'pc' in rec_config_map['algorithm_sequence'])
        picked_devs, avail_jobs, hostfile = ut.get_gpu_use(devices, want_dev_no, job_size)

    if hostfile is not None:
        picked_devs = sum(picked_devs, [])

    if no_scan_ranges == 1:
            datafile, dir = exp_dirs_data[0]
            process_scan_range(ga_method, lib, conf_file, datafile, dir, picked_devs, hostfile)
    else: # multiple scans or scan ranges
        q = None
        if avail_jobs >= want_dev_no:
            # there is enough resources to run all reconstructions simultanuesly
            # assign the resources to scans
            no_concurrent_scans = no_scan_ranges
        else:
            # not enough resources to run all reconstructions simultaneously, recycle devices
            no_concurrent_scans = int(avail_jobs // reconstructions)
            picked_devs = picked_devs[: no_concurrent_scans * reconstructions]
            # create queue to reuse devices/hostfiles
            q = Queue()

        scan_picked_devs = [picked_devs[x:x+reconstructions] for x in range(0, len(picked_devs), reconstructions)]
        if hostfile is None:
            hostfiles = [None] * no_scan_ranges
        else:
            hostfiles = split_resources(hostfile, reconstructions, no_concurrent_scans)

        pr = {}
        for i in range(no_concurrent_scans):
            datafile, dir = exp_dirs_data[i]
            # run concurrently
            p = Process(target=process_scan_range, args=(ga_method, lib, conf_file, datafile, dir, scan_picked_devs[i], hostfiles[i], q))
            p.start()
            pr[p.pid] = p

        i += 1
        while i < no_scan_ranges:
            datafile, dir = exp_dirs_data[i]
            i += 1
            pid, devs, hostfile = q.get()
            if pid in pr.keys():
               del pr[pid]
            p = Process(target=process_scan_range, args=(ga_method, lib, conf_file, datafile, dir, devs, hostfile, q))
            p.start()
            pr[p.pid] = p

        for p in pr.values():
            p.join()

        if q is not None:
            while not q.empty():
                q.get()
               
        # delete host files
        for hf in hostfiles:
            if hf is not None:
                os.remove(hf)

    sp_time = time()
    print('finished reconstruction in', sp_time - sr_time, 'seconds')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory.")
    parser.add_argument("--rec_id", action="store", help="reconstruction id, a postfix to 'results_phasing_' directory")
    parser.add_argument("--debug", action="store_true",
                        help="if True the vrifier has no effect on processing")
    args = parser.parse_args()
    manage_reconstruction(args.experiment_dir, config_id=args.rec_id, debug=args.debug)


if __name__ == "__main__":
    main()
