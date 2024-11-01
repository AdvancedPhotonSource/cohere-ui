import time
import os
import subprocess
from multiprocessing import Process
import argparse


def show_log(file):
    file = open(file, 'r')
    while True:
        time.sleep(1)
        lines = file.readlines()
        if len(lines) > 0:
            print(lines[-1])


def run_with_mpi(ga_method, lib, conf_file, datafile, dir, devices, hostfile=None):
    """

    :param ga_method: str
        info about what kind of case is requested
    :param lib: str
        defines package used in reconstruction
    :param conf_file: str
        a configuration file with reconstruction parameters
    :param datafile: str
        name of data file
    :param dir: str
        directory tree that holds the related files, can be experiment directory or scan directory
    :param devices: list
        list of integer defining device ids used for processing
    :param hostfile: str
        given when cluster configuration is used
    :return:
    """
    p = None
    start_time = time.time()
    if ga_method is None:
        script = '/reconstruction_multi.py'
    else:
        script = '/reconstruction_GA.py'
        # start process that will monitor log file and print progress
        log_file = 'ga.log'
        open(log_file, 'w')
        p = Process(target=show_log, args=(log_file,))
        p.start()


    script = os.path.realpath(os.path.dirname(__file__)).replace(os.sep, '/') + script
    if hostfile is None:
        command = ['mpiexec', '-n', str(len(devices)), 'python', script, lib, conf_file, datafile, dir, str(devices)]
    else:
        command = ['mpiexec', '-n', str(len(devices)), '--hostfile', hostfile, 'python', script, lib, conf_file, datafile, dir, str(devices)]

    subprocess.run(command, check=True, capture_output=True)
    run_time = time.time() - start_time

    # The process p was created to monitor log file and print progress info, i.e. the start of generation
    if p is not None:
        p.terminate()
        while p.is_alive():
            time.sleep(.1)
        p.close()
        os.remove(log_file)

    if ga_method is None:
        print(f'multiple reconstructions took {run_time} seconds')
    else:   # fast GA
        print(f'GA reconstruction took {run_time} seconds')



def main():
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("lib", help="lib")
    parser.add_argument("conf_file", help="conf_file")
    parser.add_argument("datafile", help="datafile")
    parser.add_argument('dir', help='dir')
    parser.add_argument('dev', help='dev')
    parser.add_argument('hostfile', action="store", help='hostfile')

    args = parser.parse_args()
    dev = ast.literal_eval(args.dev)
    run_with_mpi(args.lib, args.conf_file, args.datafile, args.dir, dev, args.hostfile)


if __name__ == "__main__":
    exit(main())
