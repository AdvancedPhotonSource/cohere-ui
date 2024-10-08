import argparse
import cohere_core.controller as rec
import os


def reconstruction(datafile):
        datafile = datafile.replace(os.sep, '/')
        rec.phasing.reconstruction(datafile,
                                   device=[0],
                                   algorithm_sequence='3*(20*ER+180*HIO)+20*ER',
                                   shrink_wrap_trigger=[1, 1],
                                   shrink_wrap_type="GAUSS",
                                   shrink_wrap_threshold=0.1,
                                   shrink_wrap_gauss_sigma=1.0,
                                   twin_trigger=[2],
                                   progress_trigger=[0, 20],
                                   save_dir="results")


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("datafile", help="data file name. It should be either tif file or numpy.")
        args = parser.parse_args()
        reconstruction(args.datafile)


if __name__ == "__main__":
    exit(main())
