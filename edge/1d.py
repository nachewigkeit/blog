import numpy as np
import matplotlib.pyplot as plt
import utils


def bigKernel(kernel):
    size = 101
    filter = np.zeros((size, size))

    for i in range(5):
        pos = i - 2
        mid = (size - 1) // 2
        filter[mid, mid + pos] = kernel[i]

    return filter


kernel = [
    [0, -1, 1, 0, 0],
    [0, -1, 0, 1, 0],
    [-1, 0, 0, 0, 1],
    [-1, -0.5, 0, 0.5, 1]
]

mag = []
for i in range(4):
    mag.append(utils.logMagnitude(bigKernel(kernel[i])))

plt.figure(figsize=(40, 20))
for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow([kernel[i]], cmap="bwr")
for i in range(4):
    plt.subplot(2, 4, i + 5)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(mag[i], cmap="gray")

plt.savefig(r"image/1d.png")
plt.show()
