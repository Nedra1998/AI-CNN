import numpy as np
import matplotlib.pyplot as plt
import os
import random

import data


def view_image(X, Y, names, id):
    rgb = X[:, id]
    img = rgb.reshape(3, 32, 32).transpose([1, 2, 0])
    plt.imshow(img)
    plt.title(str(names[id].decode('utf-8')) + "::{}".format(Y[:, id]))
    plt.show()


def main():
    x, y, name = data.load_data([1, 2, 3, 4])
    print("X:", x.shape)
    print("Y:", y.shape)
    print("N:", name.shape)
    for i in range(1):
        view_image(x.T,y.T,name, random.randint(1, 10000))


if __name__ == "__main__":
    main()
