# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.utils
=================

This module returns available, balanced devices suited for given job.
"""
import os
import ast
import GPUtil
from functools import reduce
import cohere_core.utilities as ut


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
           'estimate_no_proc',
           'get_avail_gpu_runs',
           'get_gpu_use',
           'get_one_dev',
           ]

def estimate_no_proc(arr_size, factor):
    """
    Estimates number of processes the prep can be run on. Determined by number of available cpus and size
    of array.
    Parameters
    ----------
    arr_size : int
        size of array
    factor : int
        an estimate of how much memory is required to process comparing to array size
    Returns
    -------
    int
        number of processes
    """
    from multiprocessing import cpu_count
    import psutil

    ncpu = cpu_count()
    freemem = psutil.virtual_memory().available
    nmem = freemem / (factor * arr_size)
    # decide what limits, ncpu or nmem
    if nmem > ncpu:
        return ncpu
    else:
        return int(nmem)


def get_avail_gpu_runs(devices, run_mem):
    """
    Finds how many jobs of run_mem size can run on configured GPUs on local host.

    :param devices: list or string
        list of GPU IDs or 'all' if configured to use all available GPUs
    :param run_mem: int
        size of GPU memory (in MB) needed for one job
    :return: dict
        pairs of GPU IDs, number of available jobs
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpus = GPUtil.getGPUs()
    available = {}

    for gpu in gpus:
        if devices == 'all' or gpu.id in devices:
            available[gpu.id] = gpu.memoryFree // run_mem

    return available


def get_avail_hosts_gpu_runs(devices, run_mem):
    """
    This function is called in a cluster configuration case, i.e. devices parameter is configured as dictionary of hostnames and GPU IDs (either list of the IDs or 'all' for all GPUs per host).
    It starts mpi subprocess that targets each of the configured host. The subprocess returns tuples with hostname and available GPUs. The tuples are converted into dictionary and returned.

    :param devices:
    :param run_mem:
    :return:
    """
    hosts = ','.join(devices.keys())
    script = ut.join(os.path.realpath(os.path.dirname(__file__)), 'host_utils.py')
    command = ['mpiexec', '-n', str(len(devices)), '--host', hosts, 'python', script, str(devices), str(run_mem)]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True).stdout
    mem_map = {}
    for entry in result.splitlines():
        host_devs = ast.literal_eval(entry)
        mem_map[host_devs[0]] = host_devs[1]
    return mem_map


def get_balanced_load(avail_runs, runs):
    """
    This function distributes the runs proportionally to the GPUs availability.
    If number of available runs is less or equal to the requested runs, the input parameter avail_runs becomes load.
    The function also returns number of available runs.

    :param avail_runs: dict
        keys are GPU IDs, and values are available runs
        for cluster configuration the keys are prepended with the hostnames
    :param runs: int
        number of requested jobs
    :return: dict, int
        a dictionary with the same structure as avail_runs input parameter, but with values indicating runs modified to achieve balanced distribution.
    """
    if len(avail_runs) == 0:
        return {}

    # if total number of available runs is less or equal runs, return the avail_runs,
    # and total number of available jobs
    total_available = reduce((lambda x, y: x + y), avail_runs.values())
    if total_available <= runs:
        return avail_runs, total_available

    # initialize variables for calculations
    need_runs = runs
    available = total_available
    load = {}

    # add one run from each available
    for k, v in avail_runs.items():
        if v > 0:
            load[k] = 1
            avail_runs[k] = v - 1
            need_runs -= 1
            if need_runs == 0:
                return load, runs
            available -= 1

    # use proportionally from available
    distributed = 0
    ratio = need_runs / available
    for k, v in avail_runs.items():
        if v > 0:
            share = int(v * ratio)
            load[k] = load[k] + share
            avail_runs[k] = v - share
            distributed += share
    need_runs -= distributed
    available -= distributed

    if need_runs > 0:
        # need to add the few remaining
        for k, v in avail_runs.items():
            if v > 0:
                load[k] = load[k] + 1
                need_runs -= 1
                if need_runs == 0:
                    break

    return load, runs


def get_gpu_use(devices, no_jobs, job_size):
    """
    Determines available GPUs that match configured devices, and selects the optimal distribution of jobs on available devices. If devices is configured as dict (i.e. cluster configuration) then a file "hosts" is created in the running directory. This file contains hosts names and number of jobs to run on that host.
    Parameters
    ----------
    devices : list or dict or 'all'
        Configured parameter. list of GPU ids to use for jobs or 'all' if all GPUs should be used. If cluster configuration, then
        it is dict with keys being host names.
    no_jobs : int
        wanted number of jobs
    job_size : float
        a GPU memory requirement to run one job
    Returns
    -------
    picked_devs : list or list of lists(if cluster conf)
        list of GPU ids that were selected for the jobs
    available jobs : int
        number of jobs allocated on all GPUs
    cluster_conf : boolean
        True is cluster configuration
    """

    def unpack_load(load):
        picked_devs = []
        for ds in [[k] * int(v) for k, v in load.items()]:
            picked_devs.extend(ds)
        return picked_devs

    if type(devices) != dict:  # a configuration for local host
        hostfile_name = None
        avail_jobs = get_avail_gpu_runs(devices, job_size)
        balanced_load, avail_jobs_no = get_balanced_load(avail_jobs, no_jobs)
        picked_devs = unpack_load(balanced_load)
    else:  # cluster configuration
        hosts_avail_jobs = get_avail_hosts_gpu_runs(devices, job_size)
        avail_jobs = {}
        # collapse the host dict into one dict by adding hostname in front of key (gpu id)
        for k, v in hosts_avail_jobs.items():
            host_runs = {(f'{k}_{str(kv)}'): vv for kv, vv in v.items()}
            avail_jobs.update(host_runs)
        balanced_load, avail_jobs_no = get_balanced_load(avail_jobs, no_jobs)

        # un-collapse the balanced load by hosts
        host_balanced_load = {}
        for k, v in balanced_load.items():
            idx = k.rfind('_')
            host = k[:idx]
            if host not in host_balanced_load:
                host_balanced_load[host] = {}
            host_balanced_load[host].update({int(k[idx + 1:]): v})

        # create hosts file and return corresponding picked devices
        hosts_picked_devs = [(k, unpack_load(v)) for k, v in host_balanced_load.items()]

        picked_devs = []
        hostfile_name = f'hostfile_{os.getpid()}'
        host_file = open(hostfile_name, mode='w+')
        linesep = os.linesep
        for h, ds in hosts_picked_devs:
            host_file.write(f'{h}:{str(len(ds))}{linesep}')
            picked_devs.append(ds)
        host_file.close()

    return picked_devs, int(min(avail_jobs_no, no_jobs)), hostfile_name


def get_one_dev(ids):
    """
    Returns GPU ID that is included in the configuration, is on a local node, and has the most available memory.

    :param ids: list or string or dict
        list of gpu ids, or string 'all' indicating all GPUs included, or dict by hostname
    :return: int
        selected GPU ID
    """
    import socket

    # if cluster configuration, look only at devices on local machine
    if issubclass(type(ids), dict):  # a dict with cluster configuration
        ids = ids[socket.gethostname()]  # configured devices on local host

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpus = GPUtil.getGPUs()
    dev = -1
    max_mem = 0
    # select one with the highest availbale memory
    for gpu in gpus:
        if ids == 'all' or gpu.id in ids:
            free_mem = gpu.memoryFree
            if free_mem > max_mem:
                dev = gpu.id
                max_mem = free_mem
    return dev
