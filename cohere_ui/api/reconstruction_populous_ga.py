# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.reconstruction_GA
=============================

This module controls a reconstructions using genetic algorithm (GA).
Refer to cohere_core-ui suite for use cases. The reconstruction can be started from GUI x or using command line scripts x.
"""

import numpy as np
import os
import cohere_core.utilities.utils as ut
import cohere_core.utilities.ga_utils as gaut
from multiprocessing import Queue
import shutil
from multiprocessing import Process
import argparse
from cohere_ui.api.reconstruction_populous import multi_rec


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['reconstruction']


def order_dirs(tracing, dirs, evals, metric_type):
    """
    Orders results in generation directory in subdirectories numbered from 0 and up, the best result stored in the '0' subdirectory. The ranking is done by numbers in evals list, which are the results of the generation's metric to the image array.

    Parameters
    ----------
    tracing : object
        Tracing object that keeps fields related to tracing
    dirs : list
        list of directories where the reconstruction results files are saved
    evals : list
        list of evaluations of the results saved in the directories matching dirs list. The evaluations are dict
        <metric type> : <eval result>
    metric_type : metric type to be applied for evaluation

    Returns
    -------
    list :
        a list of directories where results are saved ordered from best to worst
    dict :
        evaluations of the best results
    """
    # ranks keeps indexes of reconstructions from best to worst
    # for most of the metric types the minimum of the metric is best, but for
    # 'summed_phase' and 'area' it is opposite, so reversing the order
    metric_evals = [e[metric_type] for e in evals]
    ranks = np.argsort(metric_evals).tolist()
    if metric_type == 'summed_phase' or metric_type == 'area':
        ranks.reverse()
    best_metrics = evals[ranks[0]]

    # Add tracing for the generation results
    gen_ranks = {}
    for i in range(len(evals)):
        gen_ranks[ranks[i]] = (i, evals[ranks[i]])
    tracing.append_gen(gen_ranks)

    # find how the directories based on ranking, map to the initial start
    prev_map = tracing.map
    map = {}
    for i in range(len(ranks)):
        prev_seq = int(os.path.basename(dirs[ranks[i]]))
        inx = prev_map[prev_seq]
        map[i] = inx
    prev_map.clear()
    tracing.set_map(map)

    # order directories by rank
    rank_dirs = []
    # append "_<rank>" to each result directory name
    for i in range(len(ranks)):
        dest = f'{dirs[ranks[i]]}_{str(i)}'
        src = dirs[ranks[i]]
        shutil.move(src, dest)
        rank_dirs.append(dest)

    # remove the number preceding rank from each directory name, so the directories are numbered
    # according to rank
    current_dirs = []
    for dir in rank_dirs:
        last_sub = os.path.basename(dir)
        parent_dir = os.path.dirname(dir)
        dest = ut.join(parent_dir, last_sub.split('_')[-1])
        shutil.move(dir, dest)
        current_dirs.append(dest)
    return current_dirs, best_metrics


def cull(lst, no_left):
    if len(lst) <= no_left:
        return lst
    else:
        return lst[0:no_left]


def reconstruction(pkg, conf_file, datafile, dir, devices):
    """
    Controls reconstruction that employs genetic algorith (GA).

    This script is typically started with cohere_core-ui helper functions. The 'init_guess' parameter in the
    configuration file defines whether it is a random guess, AI algorithm determined (one reconstruction,
    the rest random), or starting from some saved state. It will set the initial guess accordingly and start
    GA algorithm. It will run multiple reconstructions for each generation in a loop. After each generation
    the best reconstruction, alpha is identified, and used for breeding. For each generation the results will
    be saved in g_x subdirectory, where x is the generation number, in configured 'save_dir' parameter or in
    'results_phasing' subdirectory if 'save_dir' is not defined.

    Parameters
    ----------
    pkg : str
        library acronym to use for reconstruction. Supported:
        np - to use numpy,
        cp - to use cupy,

    conf_file : str
        configuration file name

    datafile : str
        data file name

    dir : str
        a parent directory that holds the reconstructions. For example experiment directory or scan directory.

    devices : list
        list of GPUs available for this reconstructions

    """
    pars = ut.read_config(conf_file)
    pars = gaut.set_ga_defaults(pars)

    if pars['ga_generations'] < 2:
        print("number of generations must be greater than 1")
        return

    if pars['reconstructions'] < 2:
        print("GA not implemented for a single reconstruction")
        return

    if pars['ga_fast']:
        reconstructions = min(pars['reconstructions'], len(devices))
    else:
        reconstructions = pars['reconstructions']

    if 'ga_cullings' in pars:
        cull_sum = sum(pars['ga_cullings'])
        if reconstructions - cull_sum < 2:
            raise ValueError ("At least two reconstructions should be left after culling. Number of starting reconstructions is", reconstructions, "but ga_cullings adds to", cull_sum)

    if pars['init_guess'] == 'AI_guess':
        # run AI part first
        import cohere_core.controller.AI_guess as ai
        ai_dir = ai.start_AI(pars, datafile, dir)
        if ai_dir is None:
            return

    if 'save_dir' in pars:
        save_dir = pars['save_dir']
    else:
        # the config_rec might be an alternate configuration with a postfix that will be included in save_dir
        filename = conf_file.split('/')[-1]
        save_dir = ut.join(dir, filename.replace('config_rec', 'results_phasing'))

    # create alpha dir and placeholder for the alpha's metrics
    alpha_dir = ut.join(save_dir, 'alpha')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(alpha_dir):
        os.mkdir(alpha_dir)

    tracing = gaut.Tracing(reconstructions, pars, dir)

    q = Queue()
    prev_dirs = tracing.init_dirs
    tracing.set_map({i:i for i in range(len(prev_dirs))})
    for g in range(pars['ga_generations']):
        # delete previous-previous generation
        if g > 1:
            shutil.rmtree(ut.join(save_dir, f'g_{str(g-2)}'))
        print ('starting generation', g)
        gen_save_dir = ut.join(save_dir, f'g_{str(g)}')
        metric_type = pars['ga_metrics'][g]
        reconstructions = len(prev_dirs)
        p = Process(target=multi_rec, args=(pkg, gen_save_dir, devices, reconstructions, pars, datafile, prev_dirs, g, alpha_dir, q))
        p.start()
        p.join()

        results = q.get()
        if results == 'failed':
            print('reconstruction failed')
            return

        evals = []
        temp_dirs = []
        for r in results:
            eval = r[0]
            if eval is not None:
                evals.append(eval)
                temp_dirs.append(r[1])

        # results are saved in a list of directories - save_dir
        # it will be ranked, and moved to temporary ranked directories
        current_dirs, best_metrics = order_dirs(tracing, temp_dirs, evals, metric_type)
        reconstructions = pars['ga_reconstructions'][g]
        current_dirs = cull(current_dirs, reconstructions)
        prev_dirs = current_dirs

        # compare current alpha and previous. If previous is better, set it as alpha.
        # no need to set alpha  for last generation
        if (g == 0
                or
                best_metrics[metric_type] >= alpha_metrics[metric_type] and
                (metric_type == 'summed_phase' or metric_type == 'area')
                or
                best_metrics[metric_type] < alpha_metrics[metric_type] and
                (metric_type == 'chi' or metric_type == 'sharpness')):
            shutil.copyfile(ut.join(current_dirs[0], 'image.npy'), ut.join(alpha_dir, 'image.npy'))
            alpha_metrics = best_metrics

    # remove the previous gen
    shutil.rmtree(ut.join(save_dir, 'g_' + str(pars['ga_generations'] - 2)))
    shutil.rmtree(ut.join(save_dir, 'alpha'))
    # move results from the best reconstruction to save_dir
    source_dir = ut.join(save_dir, 'g_' + str(pars['ga_generations'] - 1), '0')
    for file_name in os.listdir(source_dir):
        try:
            os.remove(ut.join(save_dir, file_name))
        except OSError:
            pass
        shutil.move(ut.join(source_dir, file_name), save_dir)
    # remove the last gen
    shutil.rmtree(ut.join(save_dir, 'g_' + str(pars['ga_generations'] - 1)))

    tracing.save(save_dir)

    print('done gen')


def main():
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("lib", help="lib")
    parser.add_argument("conf_file", help="conf_file")
    parser.add_argument("datafile", help="datafile")
    parser.add_argument('dir', help='dir')
    parser.add_argument('dev', help='dev')

    args = parser.parse_args()
    dev = ast.literal_eval(args.dev)
    reconstruction(args.lib, args.conf_file, args.datafile, args.dir, dev)


if __name__ == "__main__":
    exit(main())