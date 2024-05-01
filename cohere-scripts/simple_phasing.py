import argparse
import cohere_core as cohere
import os


def reconstruction(datafile):
        datafile = datafile.replace(os.sep, '/')
        cohere.phasing.reconstruction(datafile,
                algorithm_sequence='3*(20*ER+180*HIO)+20*ER',
                device=[0],
                shrink_wrap_trigger=[1, 1],
                shrink_wrap_type="GAUSS",
                shrink_wrap_threshold=0.1,
                shrink_wrap_gauss_sigma=1.0,
                twin_trigger=[2],
                progress_trigger=[0, 20],
                save_dir="results" )


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("datafile", help="data file name. It should be either tif file or numpy.")
        args = parser.parse_args()
        reconstruction(args.datafile)


if __name__ == "__main__":
    exit(main())
