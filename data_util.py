from decimal import Decimal
import numpy as np

def read_data():
    """
    Read data in ./data
    :return: numpy matrix
    """
    text_file = open("./data/ex3x.dat", "r")
    raw_string = text_file.read()
    lines = raw_string.split('\n')
    x0 = []
    x1 = []
    for line in lines:
        words = line.split()
        if (len(words)<1):
            break
        x0.append(float(words[0]))
        x1.append(float(words[1]))

    text_file = open("./data/ex3y.dat", "r")
    raw_string = text_file.read()
    lines = raw_string.split('\n')
    y = []
    for line in lines:
        words = line.split()
        if (len(words) < 1):
            break
        y.append(float(words[0]))

    x = np.asarray([x0, x1])
    x = x.transpose()
    y = np.asarray(y)
    return x, y



