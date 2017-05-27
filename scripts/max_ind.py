import sys

def main():
    if not sys.argv[1:]:
        print("Usage: python %s <input.txt>")
        return 1
    mx = 0
    for line in open(sys.argv[1]):
        toks = line.strip().split()
        ind = int(toks[-1].split(":")[0])
        if ind > mx:
            mx = ind
    sys.stdout.write("Max index: %i\n" % mx)
    return 0

if __name__ == "__main__":
    sys.exit(main())
