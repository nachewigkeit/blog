import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import utils

edge = np.array(Image.open(r"image/gt.jpg"))
origin = np.array(Image.open(r"image/origin.jpg"))
freqEdge = utils.logMagnitude(edge)
freqOrigin = utils.logMagnitude(origin)
images = [origin, edge, freqOrigin, freqEdge, ]

plt.figure(figsize=(30, 20))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], cmap="gray")
plt.savefig(r"image/edge.png")
plt.show()
