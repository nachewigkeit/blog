import numpy as np
import matplotlib.pyplot as plt
import utils


def bigKernel(kernel):
    size = 101
    filter = np.zeros((size, size))

    mid = (size - 1) // 2
    filter[mid - 1:mid + 2, mid - 1:mid + 2] = kernel

    return filter


kernel = [
    [[0, 0, 0], [-1, 1, 0], [-1, 1, 0]],
    [[0, 0, 0], [-1, 0, 1], [-1, 0, 1]],
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
]

mag = []
for i in range(4):
    mag.append(utils.logMagnitude(bigKernel(kernel[i])))

plt.figure(figsize=(40, 20))
for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    if i==1:
        plt.title("France", fontsize=36)
    elif i == 2:
        plt.title("Prewitt", fontsize=36)
    elif i == 3:
        plt.title("Sobel", fontsize=36)
    plt.imshow(kernel[i], cmap="bwr")
for i in range(4):
    plt.subplot(2, 4, i + 5)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(mag[i], cmap="gray")

plt.savefig(r"image/2d.png")
plt.show()
