import numpy as np
import sys

if __name__ == "__main__":
    argv = sys.argv
    nsamples = int(argv[1]) if argv[1:] else 100000
    ndims = int(argv[2]) if argv[2:] else 200
    datapoints = np.random.random([nsamples, ndims])
    vals = np.random.randint(2, size=nsamples)
    vals[vals==0] = -1
    with open(argv[3], "w") if argv[3:] else sys.stdout as f:
        fw = f.write
        for i in range(nsamples):
            fw("".join("%f\t" % j for j in datapoints[i,:]) + "\t%i\n" % vals[i])
