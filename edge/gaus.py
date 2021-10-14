import numpy as np
import matplotlib.pyplot as plt
import utils


def gausDerivative(sigma, x, y):
    return -1 / (2 * np.pi * (sigma ** 4)) * y * np.exp(-(x * x + y * y) / (2 * sigma * sigma))


def gausDerivativeFilter(sigma):
    size = 31
    filter = np.zeros((size, size))
    mid = (size - 1) // 2
    for i in range(size):
        for j in range(size):
            filter[i, j] = gausDerivative(sigma, i - mid, j - mid)

    return filter


kernel = [
    gausDerivativeFilter(5),
    gausDerivativeFilter(2),
    gausDerivativeFilter(1),
]

mag = []
for i in range(3):
    mag.append(utils.logMagnitude(kernel[i]))

plt.figure(figsize=(30, 20))
for i in range(3):
    plt.subplot(2, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(kernel[i], cmap="bwr")
for i in range(3):
    plt.subplot(2, 3, i + 4)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(mag[i], cmap="gray")

plt.savefig(r"image\gaus.png")
plt.show()
