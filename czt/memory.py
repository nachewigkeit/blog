from scipy.signal import czt
from scipy.fft import fft
from memory_profiler import profile
import numpy as np


@profile
def czt_test():
    signal = np.ones(11)
    czt(signal, 2 ** 28)


@profile
def fft_test():
    signal = np.zeros(2 ** 28)
    signal[:11] = 1
    fft(signal)


if __name__ == "__main__":
    czt_test()
    fft_test()
