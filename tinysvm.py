import numpy as np
import math
import sys

def parse_fn(fn, sparse_dims=-1):
    if sparse_dims >= 0:
        return sparse_parse(fn, sparse_dims)
    labels = []
    vals = []
    for line in open(fn):
        if line[0] in "#\n": continue
        toks = line.strip().split()
        labels.append(int(toks[-1]))
        vals.append(list(map(float, toks[:-1])))
    labelset = set(labels)
    if len(labelset) != 2:
        raise RuntimeError("Wrong number of labels: %i" % len(labelset))
    sortlabelset = sorted(labelset)
    convdict = {sortlabelset[0]: -1, sortlabelset[1]: 1}
    for i in range(len(labels)):
        labels[i] = convdict[labels[i]]
    return np.array(labels), np.array(vals)


def sparse_parse(fn, ndims):
    labels = []
    vals = []
    for line in open(fn):
        if line[0] in "#\n": continue
        toks = line.strip().split()
        labels.append(int(toks[0]))
        row = np.zeros((ndims,))
        for tok in toks[1:]:
            subtoks = tok.split(":")[:2]
            row[int(subtoks[0]) - 1] = float(subtoks[1])
        vals.append(row)
    assert len(labels) == len(vals)
    labelset = set(labels)
    if len(labelset) != 2:
        raise RuntimeError("Wrong number of labels: %i" % len(labelset))
    sortlabelset = sorted(labelset)
    convdict = {sortlabelset[0]: -1, sortlabelset[1]: 1}
    for i in range(len(labels)):
        labels[i] = convdict[labels[i]]
    return np.array(labels), np.array(vals)


class SVM(object):
    def __init__(self, path, lb, fixed_eta=-1, rescale=True, project=True, init_from_data=True, sparse_dims=-1):
        self.lb = lb
        self.t  = 0
        self.labels, self.data = parse_fn(path, sparse_dims)
        self.w = np.zeros((len(self.data[0]),))
        self.fixed_eta = fixed_eta
        self.rescale = rescale
        self.project = project
        if init_from_data:
            for label, dp in zip(self.labels, self.data):
                self.w += dp * label * np.random.random()
        print("Rescaling" if rescale else "Not rescaling")
        print("Projecting" if project else "Not projecting")
        print("init from data" if init_from_data else "Not init from data")

    def add(self, bs=1):
        if self.fixed_eta < 0:
            eta = 1 / (self.lb * (self.t + 1))
        else:
            eta = self.fixed_eta
        # print(eta)
        tmp = np.zeros((len(self.w),))
        for i in range(bs):
            index = np.random.randint(0, len(self.labels))
            if np.dot(self.w, self.data[index,:]) * self.labels[index] < 1.:
                 tmp += self.data[index,:] * self.labels[index]
        tmp *= eta / bs
        self.w += tmp
        if self.rescale:
            scale = 1 - eta * self.lb;
            self.w *= scale
        if self.project:
            wnorm = np.linalg.norm(self.w)
            if wnorm > 0.:
                frac = 1 / math.sqrt(self.lb) / np.linalg.norm(self.w)
                if frac < 1:
                    self.w *= frac
        self.t += 1

    def loss(self):
        return 1. * sum(np.dot(self.w, self.data[i,:]) * self.labels[i] < 0
                        for i in range(len(self.labels))) / len(self.labels)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lb", "-l", type=float, default=1.)
    parser.add_argument("path")
    parser.add_argument("--no-rescale", action='store_true')
    parser.add_argument("--no-project", action='store_true')
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--inner", type=int, default=1000)
    parser.add_argument("--fixed-eta", type=float, default=-1)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--init-from-data", action='store_true')
    parser.add_argument("--sparse-dims", "-s", type=int, default=-1)
    parser.add_argument("--test", "-t")
    args = parser.parse_args()
    svm = SVM(args.path, args.lb, fixed_eta=args.fixed_eta,
              rescale=not args.no_rescale, project=not args.no_project,
              init_from_data=args.init_from_data, sparse_dims=args.sparse_dims)
    for j in range(args.outer):
        for i in range(args.inner):
           svm.add(args.bs)
        print("After %i iterations, loss: %f. wnorm squared: %f"
              % ((j + 1) * args.inner, svm.loss(), np.linalg.norm(svm.w)))
    print("weights: ", svm.w)
    sys.stderr.write("Final loss: %f. Total iterations: %i.\n" %
                     (svm.loss(), args.outer * args.inner))
    if args.test:
        test_labels, test_data = parse_fn(args.test, args.sparse_dims)
        loss = 1. * sum(np.dot(svm.w, test_data[i,:]) * test_labels[i] < 0
                        for i in range(len(test_labels))) / len(test_labels)
        sys.stderr.write("Test loss: %f.\n" % loss)
        
    sys.exit(0)
