import cohere_core.utilities as ut
import argparse
import os
import numpy as np

def diet_data(dfile):
    dta = ut.read_tif(dfile)
    shape = dta.shape

    dta = np.take(dta, range(0, shape[-1], 3), axis=2)

    ut.save_tif(dta, dfile)


def tedata(exp_dir):
    dfiles = []
    for scan_dir in os.listdir(exp_dir):
        if scan_dir.startswith('scan'):
            dfiles.append(ut.join(exp_dir, scan_dir, 'preprocessed_data', 'prep_data.tif'))

    for i in range(len(dfiles)):
        if i != len(dfiles) - 1 and i % 3 != 0:
            diet_data(dfiles[i])

    for f in dfiles:
        d = ut.read_tif(f)
        print('shape', d.shape, np.unravel_index(np.argmax(d), d.shape), np.max(d))


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("exp_dir", help="experiment dir.")
        args = parser.parse_args()
        tedata(args.exp_dir)


if __name__ == "__main__":
    exit(main())
