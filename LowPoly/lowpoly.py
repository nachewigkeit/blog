import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter, sobel


def getEdgeDrawingAnchor(gray, gausSigma=1, edgeThresh=0, anchorThresh=0, freeNum=1000):
    gaus = gaussian_filter(gray, gausSigma)

    gx = sobel(gaus, 0)
    gy = sobel(gaus, 1)

    mag = abs(gx) + abs(gy)
    mag = mag / mag.max()
    w, h = mag.shape
    direct = abs(gx) > abs(gy)

    xs, ys = np.where(mag > edgeThresh)
    anchor = []
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        if 0 < x < w - 1 and 0 < y < h - 1:
            if (direct[x, y]) and mag[x, y] - mag[x - 1, y] > anchorThresh and mag[x, y] - mag[
                x + 1, y] > anchorThresh:
                anchor.append((x, y))
            if (not direct[x, y]) and mag[x, y] - mag[x, y - 1] > anchorThresh and mag[x, y] - mag[
                x + 1, y] > anchorThresh:
                anchor.append((x, y))

    xs = np.random.randint(0, w, size=freeNum)
    ys = np.random.randint(0, h, size=freeNum)
    value = mag[xs, ys]
    accept = np.random.random(size=freeNum) > value
    freePoints = np.array([xs[accept], ys[accept]])

    points = np.concatenate((anchor, freePoints.T))

    return points


def points2canvas(points, image):
    # 添加四角
    w, h, _ = image.shape
    corners = np.array([(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)])
    points = np.concatenate((corners, points))
    print("Points Number:", len(points))

    # 三角剖分
    tri = Delaunay(points)

    # 绘制
    canvas = Image.fromarray(np.zeros((image.shape[1], image.shape[0], image.shape[2]), dtype='uint8'))
    draw = ImageDraw.Draw(canvas)
    for simplex in tri.simplices:
        vertices = tri.points[simplex].astype(int)
        mid = vertices.mean(axis=0).astype(int)
        color = image[mid[0], mid[1], :]
        draw.polygon(vertices.flatten().tolist(), fill=tuple(color.astype(int)))

    return np.array(canvas)
