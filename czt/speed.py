from scipy.signal import czt
from scipy.fft import fft
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

x = list(range(20, 28))
length_list = [2 ** i for i in x]

czt_time = []
signal = np.ones(11)
for length in tqdm(length_list):
    start = time()
    czt(signal, length)
    end = time() - start
    czt_time.append(end)

fft_time = []
for length in tqdm(length_list):
    signal = np.zeros(length)
    signal[:11] = 1
    start = time()
    fft(signal)
    end = time() - start
    fft_time.append(end)

plt.plot(x, fft_time, label='fft')
plt.plot(x, czt_time, label='czt')
plt.legend()
plt.xticks(x, ["2^" + str(i) for i in x])
plt.xlabel("Length of signal")
plt.ylabel("Time (s)")
plt.savefig("image/speed.png")
