import numpy as np
from thinkdsp import Wave, UncorrelatedUniformNoise
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import numpy as np

def create_wave(framerate, duration, amp, norm_amp=1):
    """
    Creates a wave with uncorrelated uniform noise.

    Args:
    framerate (int): The number of samples per second.
    duration (float): The duration of the wave in seconds.
    amp (float): The amplitude of the noise.

    Returns:
    Wave: A Wave object representing uncorrelated uniform noise.
    """

    # Create an instance of UncorrelatedUniformNoise
    noise = UncorrelatedUniformNoise(amp)

    # Generate time samples
    ts = np.arange(0, duration, 1/framerate)

    # Evaluate the noise signal at these times
    ys = noise.evaluate(ts)

    # Create a Wave object
    wave = Wave(ys, ts, framerate)

    # Normalize the wave - shifting minimum to 0 and scaling to [0, max_amp] range
    wave.ys = wave.ys - np.min(wave.ys)  # Shift to start at 0
    wave.ys = wave.ys / np.max(wave.ys) * norm_amp  # Scale to max at max_amp

    # Calculate power in both domains
    pow_signal = signal_power(wave)

    plt.figure(figsize=(10, 4))
    plt.plot(ts, ys, "-o")
    plt.title("Uncorrelated Uniform Noise")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    return wave, pow_signal

def signal_power(wave):
    """
    Calculates the power of a wave in both the time and frequency domain.

    Args:
    wave (Wave): The input wave.

    Returns:
    tuple: (float, float) Power calculated in time domain and frequency domain.
    """
    # Time domain calculation
    power_time_domain = np.mean(wave.ys ** 2)

    # Frequency domain calculation using FFT
    n = len(wave.ys)
    d = 1 / wave.framerate
    hs = fft(wave.ys)
    psd = np.abs(hs) ** 2
    power_freq_domain = np.sum(psd) * d / n  # Parseval's theorem

    return power_freq_domain

def noise_power(snr_db, p_signal):
    """
    Calculate the noise power given SNR in dB and signal power.

    Args:
    snr_db (float): SNR in decibels (dB).
    p_signal (float): Power of the signal.

    Returns:
    float: Calculated noise power.
    """
    # Convert SNR from dB to linear ratio and compute noise power
    p_noise = p_signal / (10 ** (snr_db / 10))
    return p_noise

# Parameters for the wave
framerate = 1000
duration = 1.0  # 1 second
amplitude = 1.0
normalization_amp = 1

# Create the wave
#noise_wave, pow = create_wave(framerate, duration, amplitude, normalization_amp)

# Plot the wave
#plt.figure(figsize=(10, 4))
#plt.plot(noise_wave.ts, noise_wave.ys, "-o")
#plt.title("Uncorrelated Uniform Noise")
#plt.xlabel("Time (s)")
#plt.ylabel("Amplitude")
#plt.grid(True)
#plt.show()

