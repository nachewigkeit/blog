from scipy.fft import fft, fftshift
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))

signal = np.zeros(110)
signal[:11] = 1
freq_fft = fft(signal)
freq_fft_shift = fftshift(freq_fft)

plt.subplot(221)
plt.xticks([])
plt.title("signal(110)", fontsize=12)
plt.plot(range(signal.shape[0]), signal)
plt.subplot(222)
plt.xticks([])
plt.title("fft(110)", fontsize=12)
plt.plot(range(freq_fft.shape[0]), abs(freq_fft_shift))

signal = np.zeros(110)
signal[50:61] = 1
freq_fft = fft(signal)
freq_fft_shift = fftshift(freq_fft)

plt.subplot(223)
plt.xticks([])
plt.title("signal(110)", fontsize=12)
plt.plot(range(signal.shape[0]), signal)
plt.subplot(224)
plt.xticks([])
plt.title("fft(110)", fontsize=12)
plt.plot(range(freq_fft.shape[0]), abs(freq_fft_shift))

plt.savefig("image/padding.png")
