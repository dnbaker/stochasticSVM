#!/usr/bin/env python
import numpy as np


def usage():
    import sys
    sys.stderr.write("%s <in.dat> <out.dat> (stdout if "
                     "out.dat omitted.)\n" % sys.argv[0])
    return 1


def get_max_index(path):
    mx = 0
    for line in open(path):
        if line[0] == "#":
            continue
        for tok in line.split()[1:]:
            ind = int(tok.split(":")[0])
            if ind > mx:
                mx = ind
    return mx


if __name__ == "__main__":
    import sys
    from sys import argv, stderr
    if len(argv) < 2:
        sys.exit(usage())
    d = get_max_index(argv[1])
    data = np.zeros([d + 1], dtype=np.double)
    # print("Shape of data: %s" % data.shape)
    out = open(argv[2], "w") if argv[2:] else sys.stdout
    with open(argv[1]) as infile:
        out.write("#Unsparsified data. %i dimensions\n" % d)
        for line in infile:
            if line[0] == "#":
                continue
            data[:] = 0
            toks = line.split()
            label = toks[0]
            for tok in toks[1:]:
                ind, val = tok.split(":")
                data[int(ind) - 1] = float(val)
            for item in data:
                out.write("%f " % item)
            out.write(" %i\n" % (int(label)))
