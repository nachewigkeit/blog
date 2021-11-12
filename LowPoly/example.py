from PIL import Image
import numpy as np
import lowpoly
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = Image.open(r"E:\images\\头像1.jpg")
gray = img.convert('L')
img = np.array(img).transpose((1, 0, 2))
gray = np.array(gray, dtype='float64').T / 255

points = lowpoly.getEdgeDrawingAnchor(gray, gausSigma=1, edgeThresh=0.5, anchorThresh=0, freeNum=1000)
canvas = lowpoly.points2canvas(points, img)
mpimg.imsave(r"image/lena.png", canvas)


plt.xticks([])
plt.yticks([])
plt.imshow(canvas)
plt.show()
