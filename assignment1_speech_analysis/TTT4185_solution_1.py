# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # TTT4185 Machine learning for Signal processing
# ## Computer assignment 1: Speech analysis
# ## Suggested Solution
#

# ### Problem 1:
#
# According to the command which are provided in the assignment, we load the .wav data which includes the sampling rate (Fs) and the waveform data (data).

# +
# %matplotlib inline
import numpy as np
import scipy as sp
import scipy.io.wavfile
from matplotlib import pyplot as plt

Fs, data = scipy.io.wavfile.read('stry.wav')
# normalize the samples between -1 and +1 (not indespensable)
data = data / np.max(np.abs(data))
# get number of samples
n_samples = data.shape[0]
# define time scale corresponding to the samples
time = np.arange(0,n_samples)/Fs
fig = plt.figure(figsize=(16, 8))
plt.plot(time,data)
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.title('Figure 1: Waveform plot for the "stry.wav" file')
# find which sample corresponds to an arbitrary point in the vowel
np.where(time==0.375)
# -

# (a) The speech waveform is shown in the Figure 1. We can see the characteristics of
# the sounds.
# - $\bf/s/$ : Noise like, no periodicity.
# - $\bf/t/$: Silence, then a burst. Typical for a plosive. Silence represents the closure in the oral cavity, while the burst indicates the opening.
# - $\bf/r/$ and $\bf/iy/$: We can see that we have periodicity here. The transition from $\bf/r/$ to $\bf/iy/$ could be difficult to find by just looking at the waveform.
#
# Plot of the $\bf/iy/$ part of speech file is shown in Figure 2.
#     We find $f_0$ by looking at Figure 2. We can se that the distance from one top to the next in the figure is approximately 5.7 ms. This gives us a $f_0$ equal 175Hz. Alternatively, count the number of periods in the segment: # periods = 7 / lenght of segment = 0.04 sec = 175Hz. The accuracy of this measurement improves the longer the segment, provided that $f_0$ is constant.

N = int(0.040 * Fs) #number of samples for 40ms
# define the indices of the window
win = 6000 + np.arange(N)
vowel = data[win] # selecting a part from the vowel /iy/
vtime = time[win] # same for the time scale
fig = plt.figure(figsize=(16, 8))
plt.plot(vtime,vowel)
plt.xlabel('Time (sec)')
plt.title('Figure 2: 40 ms plot for the "/iy/" vowel')

# (b) The spectrum estimates when rectangular window and with Hamming window are used are shown in Figure 3 and Figure 4 respectively. The two plots have similar spectral envelope. But it is easier to identify the different harmonics in the case of Hamming window. This is due to lower sidelobes in the spectral response of Hamming window.
# We find $f_0$,$F_1$ and $F_2$ from Figure 4. We find $f_0$ by looking at the distance between two tops of the spectrum. $F_1$ and $F_2$ are found by finding the two first main tops in the spectral envelope.
# We find $f_0=175$Hz as in Problem 1a, and $F_1=350$Hz and $F_2=2300$Hz. From the textbook we find $F_1$ and $F_2$ as 300Hz and 2300Hz.

# define lenght of the FFT
nfft = 1024
vowelSpectrumRect = np.fft.fft(vowel, n=nfft) # frequency response of the vowel with the rectangular window
# define the corresponding frequency scale. Note that we go up to the Nyquist frequency (Fs/2)
freq = (np.arange(nfft)*Fs/nfft)[:nfft//2]
fig = plt.figure(figsize=(16, 8))
plt.plot(freq, 20*np.log10(np.abs(vowelSpectrumRect))[:nfft//2])
plt.xlabel('frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.title('Figure 3: spectrum for vowel /iy/ (with rectagular window)')

# scale samples with hamming window
vowelHamming = vowel*np.hamming(N)
vowelSpectrumHamming = np.fft.fft(vowelHamming, n=nfft) # frequency response of the vowel with the Hamming window
# we save this for later
vowelSpectrumHammingDB = 20*np.log10(np.abs(vowelFreqHamming))[:nfft//2]
fig = plt.figure(figsize=(16, 8))
plt.plot(freq, vowelSpectrumHammingDB)
plt.xlabel('frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.title('Figure 4: spectrum for vowel /iy/ (with hamming window)')

# (c) We use the lpc from the pysptk library to find the coefficients to the AR
# models. Then use the command freqz from scipy library to estimate the spectrum. The
# plots for the spectrum for the different AR orders are shown in Figure 5
# to Figure 8. We can see that the low order model isn't able to model the
# spectral envelope. When the model order gets too high the estimate
# starts to follows the individual harmonics, instead of just the spectral
# envelope that we want to model. So a model order of 8 or 16 is best here.
# We can not easily estimate $f_0$ here because the LPC coefficients only model the filter and not the source signal, that contains the information on the fundamental frequency. In reality the estimation of the LPC coefficients is not entirely robust to variation in $f_0$. For this reason, we might still be able to find a correlation between LPCs and $f_0$, but this is only a side effect of the estimation.

# +
from pysptk.sptk import lpc
from scipy.signal import freqz
fig = plt.figure(figsize=(16, 8))
plt.plot(freq, vowelFreqHammingDB)
# check the documentation of sptk to find out that the output of the lpc function
# includes the gain and the lpc coefficients. In the documentation for the levdur
# method you will find the definition of the corresponding all-pole digital filter.
# Then check the documentation of the freqz function to know how to define this
# filter in terms of the b and a input parameters.
lpcOrd4 = lpc(vowelHamming, order=4)
# this function returns both the frequency scale and the actual smoothed spectrum
lpcSpectrumOrd4 = freqz(b=lpcOrd4[0], a=np.concatenate((np.ones(1,), lpcOrd4[1:])), fs=Fs)
plt.plot(lpcSpectrumOrd4[0], 20*np.log10(lpcSpectrumOrd4[1]),'r')
plt.title('Figure 5: spectrum based on AR with model order 4 for vowel /iy/', fontsize=16)

fig = plt.figure(figsize=(16, 8))
plt.plot(freq, vowelFreqHammingDB)
lpcOrd8 = lpc(vowelHamming, order=8)
lpcSpectrumOrd8 = freqz(b=lpcOrd8[0], a=np.concatenate((np.ones(1,), lpcOrd8[1:])), fs=Fs)
plt.plot(lpcSpectrumOrd8[0], 20*np.log10(lpcSpectrumOrd8[1]),'r')
plt.title('Figure 6: spectrum based on AR with model order 8 for vowel /iy/', fontsize=16)

fig = plt.figure(figsize=(16, 8))
plt.plot(freq, vowelFreqHammingDB)
lpcOrd16 = lpc(vowelHamming, order=16)
lpcSpectrumOrd16 = freqz(b=lpcOrd16[0], a=np.concatenate((np.ones(1,), lpcOrd16[1:])), fs=Fs)
plt.plot(lpcSpectrumOrd16[0], 20*np.log10(lpcSpectrumOrd16[1]),'r')
plt.title('Figure 7: spectrum based on AR with model order 16 for vowel /iy/', fontsize=16)

fig = plt.figure(figsize=(16, 8))
plt.plot(freq, vowelFreqHammingDB)
lpcOrd50 = lpc(vowelHamming, order=50)
lpcSpectrumOrd50 = freqz(b=lpcOrd50[0], a=np.concatenate((np.ones(1,), lpcOrd50[1:])), fs=Fs)
plt.plot(lpcSpectrumOrd50[0], 20*np.log10(lpcSpectrumOrd50[1]),'r')
plt.title('Figure 8: spectrum based on AR with model order 50 for vowel /iy/', fontsize=16)
# -

# (d) The spectrograms for both narrow-band and wide-band are plotted in Figure 9.
# We can observe that the /s/ sound has energy in higher frequencies. The /t/ sound has a segment with
# no energy, then a burst with energy in higher frequencies. This indicates
# no periodicity for the /s/ and /t/ sound. For the /r/ sound we can see that it has energy in certain frequencies, indicating periodicity. For the /iy/ sound we can see that there is energy in the frequencies corresponding to the formant frequencies.

# +
from scipy.signal import spectrogram

fig, ax = plot.subplots(1, 2, figsize=(16*0.8, 9*0.8))
fig.suptitle("Figure 9: Spectogram for the speech signal", fontsize=16)
fNB, tNB, SxxNB = spectrogram(data, Fs, nperseg=2048, noverlap=2000) # choosing a long enough window
ax[0].set_title("Narrow-band")
ax[0].pcolormesh(tNB, fNB, 20*np.log10(SxxNB))
fWB, tWB, SxxWB = spectrogram(data, Fs, nperseg=256, noverlap=250)
ax[1].set_title("Wide-band")
ax[1].pcolormesh(tWB, fWB, 20*np.log10(SxxWB))
# -

# A homomorphic transformation is a transformation that converts a convolution into a sum. Given
# \begin{align*}
#      x(n) &= e(n)*h(n) \\
#     |X(\omega)| &= |E(\omega)|\cdot|H(\omega)| .
# \end{align*}
# The real cepstrum gives
# \begin{align*}
# c_x(n) &= \frac{1}{2\pi}\int_{-\pi}^{\pi}{ln(|E(\omega)|\cdot|H(\omega)|)e^{j\omega n}d\omega} \\
# &=\frac{1}{2\pi}\int_{-\pi}^{\pi}{ln(|E(\omega)|)+ln(|H(\omega)|)e^{j\omega n}d\omega} .
# \end{align*}
# This gives
# \[c_c(n)=c_e(n)+c_h(n),\]
# We calculate the real cepstrum according to formula $c_x(n)=\frac{1}{2\pi}\int_{-\pi}^{\pi}{ln(|X(\omega)|)e^{j\omega n}d\omega}$. The plot of the cepstrum is given in Figure 10.
#
# When we calculate the cepstrum we try to separate the source and the filter. The source will then appear as a pulse train with pitch period $1/F_0$. We can therefore find $F_o$ by localizing the first peak in the plot of the ceptrum. Here we find the distance 92. We then have  $F_0$ as $T_0=T_s*N \Rightarrow 1/F_0=N/F_s \Rightarrow F_0=F_s/92 = 16~000 \mathrm{Hz} / 92 = 177$Hz.

# cepstrum is the inverse Fourier transform of the logarithm of the absolute value of the Fourier transform:
cepstr = np.fft.ifft(np.log(np.abs(vowelFreqHamming)))
plt.figure(figsize=(16, 8))
plt.title('Figure 10: real cepstrum')
plt.plot(np.array(range(0,cepstr.shape[0]))/Fs, np.real(cepstr))
plt.xlabel('Quefrency(sec)')
plt.ylabel('Amplitude')
plt.grid(True)

# The spectral envelope is ploted in Figure 11. The part of the cepstrum that includes information about the spectral envelope is the first few ceptral parameters. We extract the first 20 coefficients (and the 20 last. Remember that the cepstrum is symmetric). The FFT-spectrum is plotted in the same figure. If we compare the estimated spectral envelope here with the FFT and the AR-spectrum we can see that the envelope here is as the AR-spectrum and follows the envelope of the FFT spectrum. In this case, if we increase the number of non-zero cepstral coefficients we start modelling the contributions in the spectrum given by the glottal pulse (partials).

# +
#cepstrEnv = np.zeros(cepstr.shape)
#cepstrEnv = cepstr
# the above commands only copy by reference (the cepstr array gets modified)
# choose a number of cepstral coefficients to keep
n_cepstr = 20
truncated_cepstr = np.copy(cepstr)
# zero all the coefficients above n_cepstr
truncated_cepstr[n_cepstr:-n_cepstr] = 0
ReconSpecCeps = np.abs(np.exp(np.fft.fft(truncated_cepstr)))

plt.figure(figsize=(16, 8))
plt.title('Figure 11: spectral envalope')
plt.plot(freq, vowelFreqHammingDB)
plt.plot(freq, 20*np.log10(ReconSpecCeps[:nfft//2]),'r')

# -
