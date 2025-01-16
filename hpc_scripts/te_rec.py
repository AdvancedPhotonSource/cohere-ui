import os
import argparse
import cohere_core.utilities.utils as ut
from cohere_core.controller.phasing import TeRec
from mpi4py import MPI


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
            dfiles.append(ut.join(exp_dir, scan_dir, 'phasing_data', 'data.tif'))

    conf = ut.join(exp_dir, 'conf', 'config_rec')
    params = ut.read_config(conf)
    params['weight'] = 0.1

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # data_files = ast.literal_eval(args.datafile_dir)
    datafile = dfiles[rank]
    worker = TeRec(params, datafile, 'cp', comm)
#    worker.adjust_data()
    comm.Barrier()

    ret_code = worker.init_dev(rank % 2)  # when running on Polaris no args, otherwise pass dev id
                                          # two GPU on machine that is used now
    if ret_code < 0:
        print ('reconstruction failed, check algorithm sequence and triggers in configuration', rank)
        return

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
    exit(time_evolving_rec())

# mpiexec -n 18 python cohere-scripts/te_rec.py <experiment_dir>