import numpy as np
import util.util as vut
import argparse
import sys
from scipy.interpolate import interpn

Vi = interpn((x,y,z), V, np.array([xi,yi,zi]).T)

def interpolate(imfile, supfile):
    image = np.load(imfile)
    support = np.load(supfile)
    image, support = vut.center(image, support)
    image = vut.remove_ramp(image)



def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("imfile", help="image file name")
    parser.add_argument("supfile", help="support file name")
    args = parser.parse_args()
    interpolate(args.imfile, args.supfile)


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
