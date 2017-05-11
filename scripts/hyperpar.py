import sys
import os
from subprocess import check_output as co, CalledProcessError

A8A_FILES    = ("test/a8a.txt", "test/a8a.test")
A8A_LAMBDAS  = [0.0001, 0.001, 0.025, 0.05, 0.1, 1.]
A8A_BATCHSZ  = [32, 64, 128, 256, 512, 1024]
BRCA_FILES   = ("test/brcafix.train", "test/brcafix.test")
BRCA_LAMBDAS = [0.0001, 0.001, 0.025, 0.05, 0.1, 1.]
BRCA_BATCHSZ = [32, 64, 128, 256, 512, 1024]

HYPERPAR_SETTINGS = [(A8A_FILES, A8A_LAMBDAS, A8A_BATCHSZ, 123),
                     (BRCA_FILES, BRCA_LAMBDAS, BRCA_BATCHSZ, 10)]

if __name__ == "__main__":
    devnull = open(os.devnull, 'w')
    for settings in HYPERPAR_SETTINGS[::-1]:
        train_results, test_results = [], []
        for batch_size in settings[2]:
            for lb in settings[1]:
                tmpstr = ""
                cstr = ("./train_linear -s%i -b%i -l%f %s %s" %
                        (settings[3], batch_size, lb, settings[0][0], settings[0][1]))
                try:
                    output = [line for line in co(cstr, shell=True,
                                                  stderr=devnull,
                                                  executable="/bin/bash").decode().split('\n')
                              if "error" in line.lower()]
                except CalledProcessError:
                    print("Retrying %s" % cstr)
                    output = [line for line in co(cstr, shell=True, stderr=devnull).decode().split('\n')
                              if "error" in line.lower()]
                for line in output:
                    print(line)
                    if "test" in line.lower():
                        testout = float(line.split(":")[1][:-1])
                    if "train" in line.lower():
                        trainout = float(line.split(":")[1][:-1])
                train_results.append((trainout, lb, batch_size))
                test_results.append((trainout, lb, batch_size))
        train_results.sort()
        test_results.sort()
        sys.stdout.write("Best parameters for %s (%f): {lambda: %f, bs: %i}\n" %
                         (settings[0][0], test_results[0][0], 
                          test_results[0][1], test_results[0][2]))
        sys.stdout.write("Best parameters for %s (%f): {lambda: %f, bs: %i}\n" %
                         (settings[0][0], train_results[0][0], 
                          train_results[0][1], train_results[0][2]))
