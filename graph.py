import matplotlib.pyplot as plt
import re
import sys

def parse(file):
    pattern = re.compile('^(\d+) Train: (\d+\.\d+) Val: (\d+.\d+)$')
    test = []
    train = []
    with open(file) as f:
        for line in f:
            match = pattern.match(line)
            if match:
                train.append(float(match.group(2)))
                test.append(float(match.group(3)))
    plt.plot(test)
    plt.plot(train)
    plt.show()
    print(test)

if __name__ == "__main__":
    parse(sys.argv[1])
