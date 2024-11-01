import argparse
import cohere_core.controller as rec
import os


def reconstruction(datafile, **kwargs):
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
                                   save_dir="results",
                                   **kwargs)


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("datafile", help="data file name. It should be either tif file or numpy.")
        parser.add_argument("--no_verify", action="store_true",
                        help="if True the verifier has no effect on processing, error is always printed when incorrect configuration")
        parser.add_argument("--debug", action="store_true",
                            help="if True the exceptions are not handled")
        args = parser.parse_args()
        reconstruction(args.datafile, no_verify=args.no_verify, degug=args.debug)


if __name__ == "__main__":
    exit(main())
