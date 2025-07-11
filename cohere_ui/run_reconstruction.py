# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script manages reconstruction(s).
Depending on configuration it starts either single reconstruction, GA, or multiple reconstructions. In multiple reconstruction scenario or split scans the script runs parallel reconstructions.
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
import sys
import argparse
from multiprocessing import Process, Queue, Pool
import cohere_core.controller as rec
import cohere_core.utilities as ut
import cohere_ui.api.common as com
import cohere_ui.api.balancer as balancer
import cohere_ui.api.mpi_cmd as mpi_cmd
import cohere_ui.api.reconstruction_populous as reconstruction_populous
import cohere_ui.api.reconstruction_populous_ga as ga
import cohere_ui.api.multipeak as multipeak


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

    return hostfiles


def reconstruction_single(pkg, conf_file, datafile, dir, dev, **kwargs):
    """
    Controls single reconstruction according to parameters and conf_file.

    :param pkg:  str
        library acronym to use for reconstruction. Supported:
        'np' - to use numpy,
        'cp' - to use cupy,
        'torch' - to use torch
    :param conf_file: str
        configuration file name
    :param datafile: str
        data file name
    :param dir: str
        a parent directory that holds the reconstructions. For example experiment directory or scan directory.
    :param dev: int
        id defining GPU the this reconstruction will be utilizing, or -1 if running cpu
    :param kwargs: var parameters
        may contain:
        debug : if True the exceptions are not handled
    :return:
    """
    pars = ut.read_config(conf_file)

    pars['init_guess'] = pars.get('init_guess', 'random')
    if pars['init_guess'] == 'AI_guess':
        import cohere_core.controller.AI_guess as ai
        ai_dir = ai.start_AI(pars, datafile, dir)
        if ai_dir is None:
            return 'failed AI guess'
        pars['continue_dir'] = ai_dir

    if 'save_dir' in pars:
        save_dir = pars['save_dir']
    else:
        filename = conf_file.split('/')[-1]
        save_dir = ut.join(dir, filename.replace('config_rec', 'results_phasing'))

    if dev is None:
        device = pars.get('device', -1)
    else:
        device = dev[0]

    worker = rec.create_rec(pars, datafile, pkg, device, **kwargs)
    if worker is None:
        print('Could not create Rec object, check config_rec parameters.')
        return 'Could not create Rec object, check config_rec parameters.'

    ret_code = worker.iterate()
    if ret_code < 0:
        print ('reconstruction failed during iterations')
        return 'reconstruction failed during iterations'

    worker.save_res(save_dir)


def process_scan_range(ga_method, pkg, conf_file, datafile, dir, picked_devs, hostfile=None, q=None, debug=None):
    """
    Calls the reconstruction function in a module identified by parameter. After the reconstruction is finished, it enqueues th eprocess id wit associated list of gpus.
    Parameters
    ----------
    ga_method : str
        defines what type of GA was requested, or None
    pkg : str
        defines library to run reconstruction with
    conf_file : str
        configuration file with reconstruction parameters
    datafile : str
        name of file containing data
    dir : str
        parent directory to the <prefix>/results, or results directory
    picked_devs : list
       a list of gpus that will be used for reconstruction
    kwargs : dict
        may contain:
        hostfile : name of hostfile if cluster configuration was used
    Returns
    -------
    nothing
    """
    if len(picked_devs) == 1:
        reconstruction_single(pkg, conf_file, datafile, dir, picked_devs, debug=debug)
    elif ga_method is None:
        reconstruction_populous.reconstruction(pkg, conf_file, datafile, dir, picked_devs)
    elif ga_method == 'ga_fast':
        mpi_cmd.run_with_mpi(pkg, conf_file, datafile, dir, picked_devs, hostfile)
    else:
        # populous ga reconstruction
        ga.reconstruction(pkg, conf_file, datafile, dir, picked_devs)

    if q is not None:
        q.put((os.getpid(), picked_devs, hostfile))

    return ''


def manage_reconstruction(experiment_dir, **kwargs):
    """
    This function starts the interruption discovery process and continues the recontruction processing.
    It reads configuration file defined as <experiment_dir>/conf/config_rec.
    If multiple generations are configured, or separate scans are discovered, it will start parallel reconstructions.
    It creates image.npy file for each successful reconstruction.
    Parameters
    ----------
    experiment_dir : str
        directory where the experiment files are loacted
    :param kwargs: ver parameters
        may contain:
        - rec_id : reconstruction id, pointing to alternate config
        - no_verify : boolean switch to determine if the verification error is returned
        - debug : boolean switch to determine whether exception shell be handled during reconstruction
    Returns
    -------
    nothing
    """
    print('started reconstruction')

    conf_list = ['config_rec', 'config_mp']
    conf_maps, converted = com.get_config_maps(experiment_dir, conf_list, **kwargs)

    # check the maps
    rec_id = kwargs.pop('rec_id', None)
    if 'config_rec' not in conf_maps.keys():
        con = 'config_rec'
        if rec_id is not None:
            con = f'{con}_{rec_id}'
        raise ValueError (f'missing {con} file, exiting')
    main_config_map = conf_maps['config']
    rec_config_map = conf_maps['config_rec']

    proc = rec_config_map.get('processing', 'auto')
    devices = rec_config_map.get('device', [-1])

    separate = main_config_map.get('separate_scans', False) or main_config_map.get('separate_scan_ranges', False)
    debug = kwargs.get('debug', False)

    # find which library to run it on, default is numpy ('np')
    pkg = com.get_pkg(proc, devices)

    if sys.platform == 'darwin' or pkg == 'np':
        devices = [-1]

    # for multipeak reconstruction divert here
    if 'config_mp' in conf_maps:
        config_map = conf_maps['config_mp']
        config_map.update(main_config_map)
        config_map.update(rec_config_map)
        config_map.update({"save_dir": f"{experiment_dir}/results_phasing"})
        peak_dirs = []
        for dir in os.listdir(experiment_dir):
            if dir.startswith('mp'):
                peak_dirs.append(ut.join(experiment_dir, dir))
        multipeak.reconstruction(pkg, config_map, peak_dirs, devices, **kwargs)
        return

    # exp_dirs_data list hold pairs of data and directory, where the directory is the root of phasing_data/data.tif file, and
    # data is the data.tif file in this directory.
    exp_dirs_data = []

    if separate:
        # experiment may be multi-scan(s) in which case reconstruction will run for each scan, or scan range
        for dir in os.listdir(experiment_dir):
            if dir.startswith('scan'):
                datafile = ut.join(experiment_dir,dir,'phasing_data', 'data.tif')
                if os.path.isfile(datafile):
                    exp_dirs_data.append((datafile, ut.join(experiment_dir, dir)))
    else:
        # in typical scenario data_dir is not configured, and it is defaulted to <experiment_dir>/data
        # the data_dir is ignored in multi-scan scenario
        data_dir = rec_config_map.get('data_dir', ut.join(experiment_dir, 'phasing_data'))
        datafile = ut.join(data_dir, 'data.tif')
        if os.path.isfile(datafile):
            exp_dirs_data.append((datafile, experiment_dir))

    no_scan_ranges = len(exp_dirs_data)
    if no_scan_ranges == 0:
        raise ValueError('did not find data.tif file(s) for phasing. ')

    if rec_id is None:
        conf_file = ut.join(experiment_dir, 'conf', 'config_rec')
    else:
        conf_file = ut.join(experiment_dir, 'conf', f'config_rec_{rec_id}')

    ga_method = None
    if 'ga_generations' in rec_config_map and rec_config_map['ga_generations'] > 1:
        if 'ga_fast' in rec_config_map and rec_config_map['ga_fast']:
            ga_method = 'ga_fast'
        else:
            ga_method = 'populous'
    reconstructions = rec_config_map.get('reconstructions', 1)

    # number of wanted devices to accommodate all reconstructions is a product of no_scan_ranges and reconstructions
    want_dev_no = no_scan_ranges * reconstructions

    # This is the simplest case, i.e. one scan range, single reconstruction, no GA
    if want_dev_no == 1:
        datafile, dir = exp_dirs_data[0]
        if rec_config_map['device'] == 'all':
            print('configure device as list of int(s) for simple case')
            return

        dev = [rec_config_map['device'][0]]
        reconstruction_single(pkg, conf_file, datafile, dir, dev, **kwargs)

        print('finished reconstruction')
        return

    hostfile = None
    # if device is [-1] it will be run on cpu
    if devices == [-1]:
        # for now run locally on cpu, will be enhanced to support cluster conf
        picked_devs, avail_jobs, hostfile = devices * want_dev_no, want_dev_no, None
    else:
        # based on configured devices find what is available
        # this code below assigns jobs for GPUs
        data_size = ut.read_tif(exp_dirs_data[0][0]).size
        job_size = get_job_size(data_size, ga_method, 'pc' in rec_config_map['algorithm_sequence'])
        picked_devs, avail_jobs, hostfile = balancer.get_gpu_use(devices, want_dev_no, job_size)

    if hostfile is not None:
        picked_devs = sum(picked_devs, [])
    kwargs['hostfile'] = hostfile

    # if fast_ga and there is not enough available devices, exit
    if ga_method == 'ga_fast' and avail_jobs < want_dev_no:
        raise ValueError(f'requested {want_dev_no} reconstructions but only {avail_jobs} is available')

    if no_scan_ranges == 1:
        datafile, dir = exp_dirs_data[0]
        process_scan_range(ga_method, pkg, conf_file, datafile, dir, picked_devs, hostfile, None, debug)
    else: # multiple scans or scan ranges
        q = None
        if avail_jobs >= want_dev_no:
            # there is enough resources to run all reconstructions in parallel
            # assign the resources to scans
            parallel_scan_ranges = no_scan_ranges
        else:
            # not enough resources to run all reconstructions in parallel, recycle devices
            parallel_scan_ranges = int(avail_jobs // reconstructions)
            if parallel_scan_ranges == 0:
                # there is not enough devices to run reconstruction for one scan range
                # try to run one scan range with all picked devices
                parallel_scan_ranges = 1
            else:
                # otherwise pick the devices
                picked_devs = picked_devs[: parallel_scan_ranges * reconstructions]

            # create queue to reuse devices/hostfiles
            q = Queue()

        scan_picked_devs = [picked_devs[x:x+reconstructions] for x in range(0, len(picked_devs), reconstructions)]
        if hostfile is None:
            hostfiles = [None] * no_scan_ranges
        else:
            hostfiles = split_resources(hostfile, reconstructions, parallel_scan_ranges)

        pr = {}
        for i in range(parallel_scan_ranges):
            datafile, dir = exp_dirs_data[i]
            # run parallel
            p = Process(target=process_scan_range,
                        args=(ga_method, pkg, conf_file, datafile, dir, scan_picked_devs[i], hostfiles[i], q))
            p.start()
            pr[p.pid] = p

        i += 1
        while i < no_scan_ranges:
            datafile, dir = exp_dirs_data[i]
            i += 1
            pid, devs, hostfile = q.get()
            if pid in pr.keys():
               del pr[pid]
            p = Process(target=process_scan_range,
                        args=(ga_method, pkg, conf_file, datafile, dir, devs, hostfile, q))
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
    if hostfile is not None:
        os.remove(hostfile)

    print('finished reconstruction')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory.")
    parser.add_argument("--rec_id", action="store", help="reconstruction id, a postfix to 'results_phasing_' directory")
    parser.add_argument("--no_verify", action="store_true",
                        help="if True the verifier has no effect on processing, error is always printed when incorrect configuration")
    parser.add_argument("--debug", action="store_true",
                        help="if True the exceptions are not handled")
    args = parser.parse_args()
    manage_reconstruction(args.experiment_dir, rec_id=args.rec_id, no_verify=args.no_verify, debug=args.debug)


if __name__ == "__main__":
    main()
