import numpy as np
from numpy import fft


def logMagnitude(filter):
    freq = fft.fft2(filter)
    transfreq = fft.fftshift(freq)
    return np.log(1 + abs(transfreq))
