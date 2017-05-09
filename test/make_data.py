import numpy as np
import argparse
import sys

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, help="Number of datapoints", required=True)
    p.add_argument("-d", type=int, help="Number of dimensions", required=True)
    p.add_argument("-o", type=argparse.FileType('w'), default=sys.stdout, help="Outfile. Defaults to stdout.")
    args = p.parse_args()
    centers = [np.random.random((args.d,)), np.random.random((args.d,))]
    while np.linalg.norm(centers[0] - centers[1]) < .5:
        centers = [np.random.random((args.d,)), np.random.random((args.d,))]
    datafirsthalf = [np.random.random((args.d,)) / 10 + centers[0] for i in range(args.n >> 1)]
    datasecondhalf = [np.random.random((args.d,)) / 10 + centers[1] for i in range(args.n - len(datafirsthalf))]
    data = datafirsthalf + datasecondhalf
    labels = [1] * len(datafirsthalf) + [-1] * len(datasecondhalf)
    fpw = args.o.write
    for label, datapoints in zip(labels, data):
        fpw(" ".join(str(dp) for dp in datapoints) + " %i\n" % label)
