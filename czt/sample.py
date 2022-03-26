from scipy.signal import czt
from scipy.fft import fft, fftshift
import numpy as np
import matplotlib.pyplot as plt

signal = np.ones(11)
freq_fft = fft(signal)
freq_czt = czt(signal, 110)

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.xticks([])
plt.title("signal(11)", fontsize=12)
plt.stem(range(signal.shape[0]), signal)
plt.subplot(132)
plt.xticks([])
plt.title("fft(11)", fontsize=12)
plt.stem(range(freq_fft.shape[0]), fftshift(abs(freq_fft)))
plt.subplot(133)
plt.xticks([])
plt.title("czt(110)", fontsize=12)
plt.scatter(range(5, 10 * freq_fft.shape[0], 10), fftshift(abs(freq_fft)))
plt.plot(range(freq_czt.shape[0]), fftshift(abs(freq_czt)))
plt.savefig("image/sample.png")
