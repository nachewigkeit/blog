from scipy.ndimage import sobel
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def myImshow(img, c='gray'):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=c)


def manyImages(mag, text):
    length = len(mag)
    plt.figure(figsize=(10 * length, 15))
    for i in range(length):
        plt.subplot(2, length, i + 1)
        plt.title(text[i], fontsize=24)
        myImshow(mag[i], c='Reds')
        plt.subplot(4, length, i + 2 * length + 1)
        plt.xticks([])
        plt.yticks([])
        plt.hist(mag[i].flatten())


colorImg = Image.open(r"image\cake.jpg")
img = colorImg.convert('L')
img = np.array(img, dtype='float32') / 255

xEdge = sobel(img, 0)
yEdge = sobel(img, 1)
square = xEdge * xEdge + yEdge * yEdge

plt.figure(figsize=(30, 10))
plt.subplot(131)
plt.title("Original Image", fontsize=24)
myImshow(img)
plt.subplot(132)
plt.title("Magnitude(No Root)", fontsize=24)
mag = square
print("No Root")
print("max:", mag.max())
print("median:", np.median(mag))
myImshow(mag, c='Reds')
plt.subplot(133)
plt.title("Magnitude(Square Root)", fontsize=24)
mag = np.sqrt(square)
print("Square Root")
print("max:", mag.max())
print("median:", np.median(mag))
myImshow(mag, c='Reds')
plt.show()

mag = [
    np.sqrt(square),
    np.power(square, 1 / 3),
    np.power(square, 1 / 4)]
text = [
    "Square Root",
    "Cubic Root",
    "Quartic Root"
]
manyImages(mag, text)
plt.show()

square = np.sqrt(square)
square /= square.max()
mag = [
    np.clip(square, 0, 0.9),
    np.clip(square, 0, 0.3),
    np.clip(square, 0, 0.1)]
text = [
    "0.9",
    "0.3",
    "0.1"
]
manyImages(mag, text)
plt.show()
