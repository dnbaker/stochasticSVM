import argparse
import sys

METAKERNELS = ("MultiplicativeKernel", "AdditiveKernel")

class Kernel:
    def __init__(self, classname, switchtext="", params="",
                 kernel_args="", usage="", subkernels=[]):
        self.subkernels = subkernels
        self.params = params
        self.classname = classname
        self.switchtext = switchtext
        self.kernel_args = kernel_args
        self.usage = usage
        for kernel in self.subkernels:
            self.usage += kernel.usage
            self.switchtext += kernel.switchtext
        if self.subkernels:
            self.classname = (self.classname +
                              "<FLOAT_TYPE" +
                              ", ".join(i.classname for
                                        i in self.subkernels) + ">")
        else:
            self.classname = self.classname + "<FLOAT_TYPE>"
        if self.params:
            self.classname += "(" + ", ".join(param for i in self.params) + ")"

    def __str__(self):
        return ("#include \"src/train_svm.h\"\n#define SWITCH " +
                self.switchtext + "\n#define KERNEL_PARAMS " + self.params +
                "\n#define KERNEL_INIT " + self.classname + "\n" +
                "DECLARE_KERNEL_SVM(KERNEL_INIT, SWITCH, KERNEL_PARAMS, " +
                self.usage + ")\n")

def main():
    parser = argparse.ArgumentParser()
    raise NotImplementedError("This is in draft stage.")

if __name__ == "__main__":
    sys.exit(main())
