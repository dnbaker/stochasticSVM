import sys
import os
import itertools
from subprocess import check_output as co, CalledProcessError

def filter_call(cstr, fp):
    instr = co(cstr, shell=True, stderr=fp, executable="/bin/bash").decode()
    return [line for line in instr.split('\n') if "error" in line.lower()]


def rbf_hyperparameters():
    A8A_FILES = ("test/a8a.txt", "test/a8a.test")
    A8A_LAMBDAS = [0.0001, 0.001, 0.025, 0.05, 0.1, 1.]
    A8A_BATCHSZ = [32]
    A8A_GAMMAS  = [0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 2.5]

    BRCA_FILES = ("test/brcafix.train", "test/brcafix.test")
    BRCA_LAMBDAS = [0.0001, 0.001, 0.025, 0.05, 0.1, 1.]
    BRCA_GAMMAS  = [0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 2.5]
    BRCA_BATCHSZ = [32]
    a8a_combs = [A8A_FILES, itertools.chain.from_iterable(
        [[[(lb, batch, gamma) for lb in A8A_LAMBDAS] for
          batch in A8A_BATCHSZ] for gamma in A8A_GAMMAS]), 123]
    brca_combs = [BRCA_FILES, itertools.chain.from_iterable(
        [[[(lb, batch, gamma) for lb in BRCA_LAMBDAS] for
          batch in BRCA_BATCHSZ] for gamma in BRCA_GAMMAS]), 10]
    devnull = open(os.devnull, 'w')
    for settings in [a8a_combs, brca_combs]:
        sys.stderr.write("Processing %s, %s\n" % (settings[0][0],
                                                  settings[0][1]))
        results = []
        for lb, batch_size, gamma in settings[1]:
                cstr = ("./train_rbf -g%f -s%i -b%i -M10000 -l%f %s %s" %
                        (gamma, settings[2], batch_size, lb,
                         settings[0][0], settings[0][1]))
                try:
                    output = filter_call(cstr, devnull)
                except CalledProcessError:
                    output = filter_call(cstr, devnull)
                for line in output:
                    if "test" in line.lower():
                        testout = float(line.split(":")[1][:-1])
                    if "train" in line.lower():
                        trainout = float(line.split(":")[1][:-1])
                results.append((testout, trainout, lb, batch_size))
        results.sort(key=lambda x: x[0] * 10 + x[1])
        sys.stdout.write("Best parameters for %s (test %f, "
                         "train %f): {lambda: %f, bs: %i}\n" %
                         (settings[0][0], results[0][0], results[0][1],
                          results[0][2], results[0][3]))


def linear_hyperparameters():
    A8A_FILES = ("test/a8a.txt", "test/a8a.test")
    A8A_LAMBDAS = [0.0001, 0.001, 0.025, 0.05, 0.1, 1.]
    A8A_BATCHSZ = [32, 64, 128, 256, 512, 1024]
    BRCA_FILES = ("test/brcafix.train", "test/brcafix.test")
    BRCA_LAMBDAS = [0.0001, 0.001, 0.025, 0.05, 0.1, 1.]
    BRCA_BATCHSZ = [32, 64, 128, 256, 512, 1024]
    HYPERPAR_SETTINGS = [(A8A_FILES, A8A_LAMBDAS, A8A_BATCHSZ, 123),
                         (BRCA_FILES, BRCA_LAMBDAS, BRCA_BATCHSZ, 10)]
    devnull = open(os.devnull, 'w')
    for settings in HYPERPAR_SETTINGS[::-1]:
        sys.stderr.write("Processing %s, %s\n" % (settings[0][0],
                                                  settings[0][1]))
        results = []
        for batch_size in settings[2]:
            for lb in settings[1]:
                tmpstr = ""
                cstr = ("./train_linear -s%i -b%i -l%f %s %s" %
                        (settings[3], batch_size, lb,
                         settings[0][0], settings[0][1]))
                try:
                    output = filter_call(cstr, devnull)
                except CalledProcessError:
                    output = filter_call(cstr, devnull)
                for line in output:
                    if "test" in line.lower():
                        testout = float(line.split(":")[1][:-1])
                    if "train" in line.lower():
                        trainout = float(line.split(":")[1][:-1])
                results.append((testout, trainout, lb, batch_size))
        results.sort(key=lambda x: x[0] * 10 + x[1])
        sys.stdout.write("Best parameters for %s (test %f, "
                         "train %f): {lambda: %f, bs: %i}\n" %
                         (settings[0][0], results[0][0], results[0][1],
                          results[0][2], results[0][3]))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-linear", action='store_true')
    p.add_argument("--run-rbf",    action='store_true')
    args = p.parse_args()
    if args.run_linear:
        linear_hyperparameters()
    if args.run_rbf:
        rbf_hyperparameters()
