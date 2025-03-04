import os
import argparse
import cohere_core.utilities.utils as ut
from cohere_core.controller.phasing import TeRec
from mpi4py import MPI
import time


# def time_evolving_rec():
#     import ast
#     parser = argparse.ArgumentParser()
# #    parser.add_argument("conf", help="conf")
#     parser.add_argument("exp_dir", help="directory with datafiles")
#     args = parser.parse_args()
#     exp_dir = args.exp_dir
#
#     dfiles = []
#     for scan_dir in os.listdir(exp_dir):
#         if scan_dir.startswith('scan'):
#             dfiles.append(ut.join(exp_dir, scan_dir, 'preprocessed_data', 'prep_data.tif'))
#             print('scan, size', scan_dir, os.path.getsize(ut.join(exp_dir, scan_dir, 'preprocessed_data', 'prep_data.tif')))
#     # assuming the first scan is full, followed by n low density scan, and so on.
#     full_size = os.path.getsize(dfiles[0])
#     small_size = os.path.getsize(dfiles[1])
#     # find ratio r, which means the pattern: full_size, (r - 1) small_size
#     r = int(full_size / small_size + .5)
#
#     conf = ut.join(exp_dir, 'conf', 'config_rec')
#     params = ut.read_config(conf)
#     params['weight'] = 0.1
#
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#
#     # data_files = ast.literal_eval(args.datafile_dir)
#     datafile = dfiles[rank]
#     worker = TeRec(params, datafile, 'cp', comm)
# #    worker.adjust_data()
#     comm.Barrier()
#
#     ret_code = worker.init_dev(rank % 2)  # when running on Polaris no args, otherwise pass dev id
#                                           # two GPU on machine that is used now
#     if ret_code < 0:
#         print ('reconstruction failed, check algorithm sequence and triggers in configuration', rank)
#         return
#
#     ret_code = worker.init_iter_loop()
#     if ret_code < 0:
#         print ('reconstruction failed, check algorithm sequence and triggers in configuration')
#         return
#
#     ret_code = worker.iterate()
#     if ret_code < 0:
#         print ('reconstruction failed during iterations')
#         return
#
#     if 'save_dir' in params:
#         save_dir = params['save_dir']
#     else:
#         save_dir, filename = os.path.split(datafile)
#         save_dir = save_dir.replace('phasing_data', 'results_phasing')
#     worker.save_res(save_dir)


def time_evolving_rec():
    import ast
    parser = argparse.ArgumentParser()
#    parser.add_argument("conf", help="conf")
    parser.add_argument("exp_dir", help="directory with datafiles")
    args = parser.parse_args()
    exp_dir = args.exp_dir

    dfiles = []
    for scan_dir in os.listdir(exp_dir):
        if scan_dir.startswith('scan'):
            dfiles.append(ut.join(exp_dir, scan_dir, 'phasing_data', 'data.npy'))

    conf = ut.join(exp_dir, 'conf', 'config_rec')
    params = ut.read_config(conf)
    params['weight'] = 0.1

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # data_files = ast.literal_eval(args.datafile_dir)
    datafile = dfiles[rank]
    worker = TeRec(params, datafile, 'cp', comm)

    ret_code = worker.init_dev(rank % 2)  # when running on Polaris no args, otherwise pass dev id
                                          # two GPU on machine that is used now
    if ret_code < 0:
        print ('reconstruction failed, check algorithm sequence and triggers in configuration', rank)
        return

    worker.exchange_data_info()

    ret_code = worker.init_iter_loop()
    if ret_code < 0:
        print ('reconstruction failed, check algorithm sequence and triggers in configuration')
        return

    ret_code = worker.iterate()
    if ret_code < 0:
        print ('reconstruction failed during iterations')
        return

    if 'save_dir' in params:
        save_dir = params['save_dir']
    else:
        save_dir, filename = os.path.split(datafile)
        save_dir = save_dir.replace('phasing_data', 'results_phasing')
    worker.save_res(save_dir)

if __name__ == "__main__":
    st = time.time()
    exit_code = time_evolving_rec()
    en = time.time()
    print(f'reconstruction took {st - en} seconds.')
    exit(exit_code)

# mpiexec -n 18 python cohere-scripts/te_rec.py <experiment_dir>