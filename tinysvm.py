import numpy as np
import math

def parse_fn(fn):
    labels = []
    vals = []
    for line in open(fn):
        if line[0] in "#\n": continue
        toks = line.strip().split()
        labels.append(int(toks[-1]))
        vals.append(list(map(float, toks[:-1])))
    return np.array(labels), np.array(vals)

class SVM(object):
    def __init__(self, path, lb, fixed_eta=-1, rescale=True, project=True, init_from_data=True):
        self.lb = lb
        self.t  = 0
        self.labels, self.data = parse_fn(path)
        self.w = np.zeros((len(self.data[0]),))
        self.fixed_eta = fixed_eta
        self.rescale = rescale
        self.project = project
        if init_from_data:
            for label, dp in zip(self.labels, self.data):
                self.w += dp * label * np.random.random()
        print("Rescaling" if rescale else "Not rescaling")
        print("Projecting" if project else "Not projecting")

    def add(self, bs=1):
        if self.fixed_eta < 0:
            eta = 1 / (self.lb * self.t + 1))
        else:
            eta = self.fixed_eta
        # print(eta)
        tmp = np.zeros((len(self.w),))
        for i in range(bs):
            index = np.random.randint(0, len(self.labels))
            if np.dot(self.w, self.data[index,:]) * self.labels[index] < 1.:
                 tmp += self.data[index,:] * self.labels[index]
        tmp *= eta / bs
        # print("tmp: ", tmp)
        self.w += tmp
        if self.rescale:
            scale = 1 - eta * self.lb;
            # print("scale: ", scale)
            self.w *= scale
        if self.project:
            wnorm = np.linalg.norm(self.w)
            if wnorm > 0.:
                frac = 1 / math.sqrt(self.lb) / np.linalg.norm(self.w)
                if frac < 1:
                    self.w *= frac
        self.t += 1

    def loss(self):
        return 1. * sum(np.dot(self.w, self.data[i,:]) * self.labels[i] < 0 for i in range(len(self.labels))) / len(self.labels)
                



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lb", "-l", type=float, default=1.)
    parser.add_argument("path")
    parser.add_argument("--rescale", action='store_false')
    parser.add_argument("--project", action='store_false')
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--inner", type=int, default=1000)
    parser.add_argument("--fixed-eta", type=float, default=-1)
    parser.add_argument("--bs", type=int, default=1)
    args = parser.parse_args()
    svm = SVM(args.path, args.lb, args.fixed_eta, args.rescale, args.project)
    for j in range(args.outer):
        for i in range(args.inner):
           svm.add(args.bs)
        print("loss: %f" % svm.loss())
        print("wnorm: %f" % np.linalg.norm(svm.w))
    
