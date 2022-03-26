from scipy.signal import czt
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import numpy as np
import matplotlib.pyplot as plt


def czt2(signal, mx, my):
    """
    Compute the 2D chirp-z transform using the scipy.signal.czt function.
    :param signal: The signal to transform.
    :param mx: The number of points in axis 0.
    :param my: The number of points in axis 1.
    :return: The transformed signal.
    """
    return czt(czt(signal, mx, axis=0), my, axis=1)


plt.figure(figsize=(6, 6))

plt.subplot(221)
plt.xticks([])
plt.yticks([])
plt.title("signal", fontsize=12)
signal = np.zeros((110, 110))
signal[50:61, 50:61] = 1
plt.imshow(signal, cmap='gray')

plt.subplot(222)
plt.xticks([])
plt.yticks([])
plt.title("fft result", fontsize=12)
plt.imshow(np.abs(fftshift(fft2(signal))), cmap='gray')

plt.subplot(223)
plt.xticks([])
plt.yticks([])
plt.title("czt result", fontsize=12)
signal = np.ones((11, 11))
freq = czt2(signal, 110, 110)
plt.imshow(abs(fftshift(freq)), cmap='gray')

plt.subplot(224)
plt.xticks([])
plt.yticks([])
plt.title("ifft from czt result", fontsize=12)
plt.imshow(ifftshift(np.abs(ifft2(freq))), cmap='gray')

plt.savefig("image/n_d.png")
